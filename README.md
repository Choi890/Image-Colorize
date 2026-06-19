# Image Colorization

`DDColor`(ModelScope)를 사용해 흑백/저채도 이미지를 고품질로 컬러라이징합니다.

## 1) 설치

> CUDA가 활성화된 PyTorch를 먼저 설치하세요. (아래는 CUDA 12.1 예시)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt`는 NumPy/Scipy/Scikit-learn 호환 버전을 고정해 ModelScope import 충돌을 피하도록 설정되어 있습니다.

## 2) 단일 이미지 실행

```powershell
python colorize_cuda.py --input .\input\old_photo.jpg --output .\output --tta --preserve-luma --luma-strength 0.9 --chroma-boost 1.05
```

## 3) 폴더 일괄 처리

```powershell
python colorize_cuda.py --input .\input --output .\output --tta --preserve-luma --infer-max-side 1280
```

## 4) 주피터 노트북 실행

```powershell
jupyter notebook .\Colorized.ipynb
```

노트북 안에서 `INPUT_PATH`, `OUTPUT_DIR`, `GPU_INDEX`만 먼저 바꾼 뒤 셀을 위에서부터 실행하면 됩니다.

## 주요 옵션

- `--gpu 0`: 사용할 NVIDIA GPU 인덱스
- `--tta`: 좌우 반전 앙상블로 안정적인 색감
- `--preserve-luma`: 원본 명도 디테일 유지
- `--luma-strength`: 명도 유지 강도(0~1)
- `--chroma-boost`: 채도 보정 배율
- `--denoise`: 후처리 노이즈 완화
- `--infer-max-side`: 추론 해상도 상한(메모리 절감)
- `--keep-alpha`: PNG/WEBP/TIFF 알파 채널 유지

## 품질 팁

- 얼굴/인물 사진: `--tta --preserve-luma --luma-strength 0.85`
- 풍경/자연 사진: `--tta --chroma-boost 1.1`
- 매우 큰 이미지(4K+): `--infer-max-side 1024` 또는 `1280`
