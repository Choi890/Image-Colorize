import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from modelscope.outputs import OutputKeys
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ALPHA_EXTS = {".png", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="High-quality CUDA image colorization with DDColor (ModelScope)."
    )
    parser.add_argument("--input", type=str, required=True, help="Input file or directory path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--model",
        type=str,
        default="damo/cv_ddcolor_image-colorization",
        help="ModelScope model id",
    )
    parser.add_argument("--gpu", type=int, default=0, help="CUDA GPU index")
    parser.add_argument(
        "--infer-max-side",
        type=int,
        default=1024,
        help="Resize long side for inference to reduce VRAM usage (0 to disable)",
    )
    parser.add_argument(
        "--tta",
        action="store_true",
        help="Enable test-time augmentation (horizontal flip ensemble)",
    )
    parser.add_argument(
        "--preserve-luma",
        action="store_true",
        help="Preserve original luminance detail using LAB merge",
    )
    parser.add_argument(
        "--luma-strength",
        type=float,
        default=1.0,
        help="0~1, used when --preserve-luma is enabled",
    )
    parser.add_argument(
        "--chroma-boost",
        type=float,
        default=1.0,
        help="Color vividness multiplier (>1.0 is more saturated)",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Apply light bilateral denoise after colorization",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_colorized",
        help="Suffix for output filenames",
    )
    parser.add_argument(
        "--keep-alpha",
        action="store_true",
        help="Keep original alpha channel for PNG/WEBP/TIFF input",
    )
    return parser.parse_args()


def list_input_images(input_path: Path) -> List[Path]:
    # Accept either a single image or a directory tree so batch runs share one CLI path.
    if input_path.is_file():
        if input_path.suffix.lower() not in VALID_EXTS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    files = [p for p in sorted(input_path.rglob("*")) if p.suffix.lower() in VALID_EXTS]
    if not files:
        raise ValueError(f"No image files found under: {input_path}")
    return files


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img

    if img.dtype == np.uint16:
        return (img / 257.0).astype(np.uint8)

    if np.issubdtype(img.dtype, np.floating):
        max_val = float(np.nanmax(img))
        if max_val <= 1.0:
            img = img * 255.0
        return np.clip(img, 0, 255).astype(np.uint8)

    return np.clip(img, 0, 255).astype(np.uint8)


def load_image(image_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    # Normalize every supported source into BGR plus optional alpha before model inference.
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to decode image: {image_path}")

    alpha = None
    if img.ndim == 2:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] == 4:
        bgr = img[..., :3]
        alpha = img[..., 3]
    elif img.ndim == 3 and img.shape[2] == 3:
        bgr = img
    elif img.ndim == 3 and img.shape[2] == 1:
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    bgr = normalize_to_uint8(bgr)
    if alpha is not None:
        alpha = normalize_to_uint8(alpha)
    return bgr, alpha


def resize_for_inference(img: np.ndarray, infer_max_side: int) -> np.ndarray:
    if infer_max_side <= 0:
        return img
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side <= infer_max_side:
        return img

    scale = infer_max_side / float(long_side)
    new_w = max(32, int(round(w * scale)))
    new_h = max(32, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def run_model(colorizer, bgr: np.ndarray) -> np.ndarray:
    result = colorizer(bgr)
    if OutputKeys.OUTPUT_IMG not in result:
        raise KeyError(f"Pipeline output missing '{OutputKeys.OUTPUT_IMG}'. keys={list(result.keys())}")

    out = result[OutputKeys.OUTPUT_IMG]
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(f"Unexpected model output shape: {out.shape}")
    return out


def colorize_with_tta(colorizer, bgr: np.ndarray, use_tta: bool) -> np.ndarray:
    # Horizontal flip TTA blends two predictions to reduce one-sided color artifacts.
    base = run_model(colorizer, bgr)
    if not use_tta:
        return base

    flipped_in = cv2.flip(bgr, 1)
    flipped_out = run_model(colorizer, flipped_in)
    flipped_out = cv2.flip(flipped_out, 1)
    return cv2.addWeighted(base, 0.55, flipped_out, 0.45, 0.0)


def merge_luma(original_bgr: np.ndarray, color_bgr: np.ndarray, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))

    orig_lab = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2LAB)
    color_lab = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2LAB)

    merged_l = (
        orig_lab[..., 0].astype(np.float32) * strength
        + color_lab[..., 0].astype(np.float32) * (1.0 - strength)
    )
    merged_l = np.clip(merged_l, 0, 255).astype(np.uint8)

    merged_lab = np.dstack((merged_l, color_lab[..., 1], color_lab[..., 2]))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)


def boost_chroma(bgr: np.ndarray, factor: float) -> np.ndarray:
    if abs(factor - 1.0) < 1e-6:
        return bgr

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1].astype(np.float32) * factor
    hsv[..., 1] = np.clip(s, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def maybe_denoise(bgr: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return bgr
    return cv2.bilateralFilter(bgr, d=0, sigmaColor=20, sigmaSpace=5)


def maybe_attach_alpha(
    bgr: np.ndarray,
    alpha: Optional[np.ndarray],
    enabled: bool,
    src_suffix: str,
) -> np.ndarray:
    if not enabled or alpha is None:
        return bgr
    if src_suffix.lower() not in ALPHA_EXTS:
        return bgr
    if alpha.shape != bgr.shape[:2]:
        alpha = cv2.resize(alpha, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    bgra = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    bgra[..., 3] = alpha
    return bgra


def build_output_path_with_structure(
    output_dir: Path,
    src: Path,
    input_root: Optional[Path],
    suffix: str,
) -> Path:
    if input_root is not None:
        rel = src.relative_to(input_root)
        dst_dir = output_dir / rel.parent
        dst_name = f"{rel.stem}{suffix}{rel.suffix}"
    else:
        dst_dir = output_dir
        dst_name = f"{src.stem}{suffix}{src.suffix}"

    dst_dir.mkdir(parents=True, exist_ok=True)
    return dst_dir / dst_name


def build_colorizer(model_id: str, gpu_idx: int):
    candidates = [f"cuda:{gpu_idx}", "cuda", f"gpu:{gpu_idx}", "gpu"]
    seen = set()
    unique_candidates = [d for d in candidates if not (d in seen or seen.add(d))]
    last_error = None

    for device in unique_candidates:
        try:
            colorizer = pipeline(task=Tasks.image_colorization, model=model_id, device=device)
            print(f"Using device: {device}")
            return colorizer
        except Exception as exc:  # noqa: BLE001
            last_error = exc

    raise RuntimeError(
        "ModelScope pipeline device initialization failed. "
        f"attempted={unique_candidates}, last_error={last_error}"
    )


def main() -> None:
    args = parse_args()
    # Validate user-facing knobs before touching CUDA so failures are quick and explicit.
    if not 0.0 <= args.luma_strength <= 1.0:
        raise ValueError(f"--luma-strength must be in [0, 1], got {args.luma_strength}")
    if args.chroma_boost <= 0:
        raise ValueError(f"--chroma-boost must be > 0, got {args.chroma_boost}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU was not found. Check NVIDIA driver/CUDA/PyTorch(CUDA) setup.")
    if args.gpu < 0 or args.gpu >= torch.cuda.device_count():
        raise ValueError(f"--gpu is out of range. gpu_count={torch.cuda.device_count()}, input={args.gpu}")

    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    colorizer = build_colorizer(args.model, args.gpu)

    input_path = Path(args.input)
    output_dir = Path(args.output)
    images = list_input_images(input_path)
    input_root = input_path if input_path.is_dir() else None
    is_single_input = input_path.is_file()
    failures: List[Tuple[Path, str]] = []

    for image_path in tqdm(images, desc="Colorizing"):
        try:
            bgr, alpha = load_image(image_path)
            src_h, src_w = bgr.shape[:2]

            infer_in = resize_for_inference(bgr, args.infer_max_side)
            infer_out = colorize_with_tta(colorizer, infer_in, args.tta)

            if infer_out.shape[:2] != (src_h, src_w):
                infer_out = cv2.resize(infer_out, (src_w, src_h), interpolation=cv2.INTER_CUBIC)

            out = infer_out
            if args.preserve_luma:
                out = merge_luma(bgr, out, args.luma_strength)

            out = boost_chroma(out, args.chroma_boost)
            out = maybe_denoise(out, args.denoise)
            out = maybe_attach_alpha(out, alpha, args.keep_alpha, image_path.suffix)

            out_path = build_output_path_with_structure(output_dir, image_path, input_root, args.suffix)
            ok = cv2.imwrite(str(out_path), out)
            if not ok:
                raise IOError(f"Failed to save image: {out_path}")
        except Exception as exc:  # noqa: BLE001
            if is_single_input:
                raise
            failures.append((image_path, str(exc)))
            tqdm.write(f"[WARN] {image_path}: {exc}")

    done_count = len(images) - len(failures)
    print(f"Done. Saved {done_count}/{len(images)} file(s) to: {output_dir}")
    if failures:
        print("Failed files:")
        for p, err in failures:
            print(f"- {p}: {err}")


if __name__ == "__main__":
    main()
