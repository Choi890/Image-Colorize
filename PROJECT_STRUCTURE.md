# Image Colorize 프로젝트 구조 설명

## 프로젝트 한줄 설명

Python 이미지 컬러라이징 실험 프로젝트입니다. OpenCV/DNN 기반 모델을 사용해 흑백 이미지를 컬러 이미지로 변환하는 노트북과 실행 스크립트가 들어 있습니다.

## 기본 작동 흐름

- requirements.txt의 Python 패키지를 설치합니다.
- Colorized.ipynb 또는 colorize_cuda.py가 입력 이미지를 읽고 컬러라이징 모델 추론을 실행합니다.
- 결과 이미지는 원본 대비 색상 복원 결과를 확인하는 데 사용됩니다.

## 문서 기준

- 아래 목록은 `git ls-files`로 확인되는 Git 추적 파일을 기준으로 작성했습니다.
- `.git`, `node_modules`, `build`, `.gradle`, 임시 업로드/출력물처럼 Git이 관리하지 않는 폴더는 제외했습니다.
- 폴더 표는 코드와 자산이 어떤 책임으로 나뉘는지, 파일 표는 각 파일이 실제로 무엇을 담당하는지 설명합니다.

## 폴더별 설명 (1개)

| 폴더 | 설명 |
| --- | --- |
| `.` | 프로젝트 루트입니다. 실행/빌드 설정, README, 전체 구조 문서, 최상위 진입 파일이 모여 있습니다. |

## 파일별 설명 (6개)

| 파일 | 설명 |
| --- | --- |
| `.gitignore` | Git에 올리지 않을 빌드 산출물, 캐시, 개인 환경 파일을 지정하는 설정 파일입니다. 저장소에는 필요한 소스/자산만 남기도록 도와줍니다. |
| `colorize_cuda.py` | CUDA/GPU 사용 가능 환경에서 이미지 컬러라이징 모델 추론을 실행하는 Python 스크립트입니다. |
| `Colorized.ipynb` | 이미지 컬러라이징 과정을 셀 단위로 실행하고 결과를 확인하는 Jupyter Notebook입니다. |
| `PROJECT_STRUCTURE.md` | 프로젝트의 모든 주요 폴더와 Git 추적 파일을 한글로 설명하는 구조 문서입니다. 처음 보는 사람이 경로별 역할을 빠르게 파악하기 위해 추가했습니다. |
| `README.md` | 프로젝트 개요, 실행 방법, 주요 기능을 설명하는 기본 안내 문서입니다. |
| `requirements.txt` | Python 실행에 필요한 기본 패키지 목록입니다. `pip install -r requirements.txt`로 설치합니다. |

## 읽는 방법

- 먼저 폴더별 설명에서 큰 기능 묶음을 확인한 다음, 파일별 설명에서 실제 구현 파일을 찾으면 됩니다.
- Android 프로젝트는 `app/src/main/java` 아래 Kotlin 파일이 핵심 코드이고, `app/src/main/res`와 `app/src/main/assets`는 화면/모델/오디오 자산입니다.
- 웹 프로젝트는 `index.html`, `styles.css`, `script.js` 또는 `app.js`가 화면 구조, 스타일, 동작을 나눠 담당합니다.
- Python 프로젝트는 루트의 실행 스크립트와 `src`, `backend`, `scripts`, `tests` 폴더를 함께 보면 처리 흐름을 이해할 수 있습니다.
