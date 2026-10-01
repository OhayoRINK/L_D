# 실행 노트북

노트북은 단계별로 실행합니다. 세 기준 모델의 경향성 학습은 각각의 노트북에서 `RUN_FULL_TRAINING = True`일 때 시작됩니다. 학습 셀은 기본 꺼짐 상태이며 각 모델은 100 epoch입니다. `02_final_test.ipynb`는 설정을 확정한 최종 평가 단계 전까지 실행하지 않습니다.

## 로컬 실행

1. 프로젝트 폴더를 VS Code에서 열고 `Python (CSI_SLLM_RESEARCH)` 또는 `.venv`를 Python 커널로 선택합니다.
2. `00_setup_and_data.ipynb`를 위에서 아래로 실행합니다.
3. `01_csinet_baseline.ipynb`에서 짧은 CsiNet smoke check를 실행하고, 이미 학습했다면 다시 전체 학습할 필요는 없습니다.
4. CRNet과 TransNet은 `03_crnet_100ep.ipynb`, `04_transnet_100ep.ipynb`를 각각 열어 shape 확인 후 전체 학습 설정을 켭니다.
5. 세 모델 학습 후 `05_compare_validation.ipynb`에서 validation best 결과를 확인합니다. 결과와 체크포인트는 `runs/` 아래에 저장됩니다.

## Google Colab 실행

1. `CSI_SLLM_RESEARCH` 폴더 전체를 Google Drive의 `MyDrive/CSI_SLLM_RESEARCH`에 둡니다. `Data/`에는 공식 MAT 파일이 있어야 합니다.
2. 실행할 각 노트북을 Drive에서 열고 런타임 유형에서 GPU를 선택합니다. `03_crnet_100ep.ipynb`와 `04_transnet_100ep.ipynb`는 각각 독립 실행됩니다.
3. 기본 프로젝트 경로 `/content/drive/MyDrive/CSI_SLLM_RESEARCH`를 확인하고 위에서 아래로 셀을 실행합니다.
4. 필요한 train/validation 데이터만 Drive에서 세션 로컬 저장소로 복사합니다. 학습 결과와 로그는 Drive 프로젝트의 `runs/`로 저장합니다.

Lite-DPAS V2는 [`08_lite_dpas_v2_experiment.ipynb`](08_lite_dpas_v2_experiment.ipynb)를 사용하세요. 프로젝트 전체를 `MyDrive/CSI_SLLM_RESEARCH`에 두고 노트북을 실행하면 Drive를 mount한 뒤 train/validation MAT만 세션 로컬로 복사합니다. 모델 코드·`runs/` 체크포인트와 로그는 Drive에 남습니다. 설정에서 `SAVE_EVERY = 100`이 켜져 있으며, 연결이 끊겼을 때는 `RESUME_FROM`을 Drive의 마지막 `checkpoint_epoch_XXXX.pt`로 지정하고 학습 셀을 다시 실행합니다. 목표 총 epoch는 그대로 두세요(예: epoch 100 checkpoint에서 1,000까지 이어서면 `EPOCHS = 1000`).

Colab은 런타임에 기본 제공되는 PyTorch를 사용합니다. 로컬용 CUDA 휠이 지정된 `requirements-baseline.txt`를 Colab에서 설치하지 마세요. 런타임이 초기화되면 셀을 처음부터 다시 실행해야 합니다.

현재 PyTorch CsiNet은 저자 Keras 구조를 바탕으로 새로 옮긴 초기 포트입니다. 저자 제공 체크포인트와 수치적으로 동등한지 검증되기 전에는 결과를 “저자 공식 구현 재현”으로 표현하지 말고, 우선 자체 포트 baseline으로 기록하세요.

## 노트북 순서

- `06_router_feasibility.ipynb`: 고정 CR=1/4 validation에서 CsiNet·CRNet·TransNet별 샘플 NMSE와 batch-1 latency를 측정하고 oracle upper bound를 계산합니다. 정답 CSI를 쓰는 oracle은 실현 가능한 정책이 아닙니다. 결과는 정책 연구 진행 여부를 판단하는 feasibility 자료이며 test를 읽지 않습니다.

## Lite-DPAS 테스트 빌드 (새 경량 모델)

Lite-DPAS V2는 기존 이중 경로와 주파수 보정을 유지하면서 학습형 stride-2 다운샘플링과 PixelShuffle 업샘플링을 사용합니다. V1 실험과 체크포인트는 그대로 보존됩니다.

대화형 실험은 [`08_lite_dpas_v2_experiment.ipynb`](08_lite_dpas_v2_experiment.ipynb)를 열어 순서대로 실행하세요. 기본 상태에서는 모델 동작 확인만 하며 학습은 시작하지 않습니다. 짧은 학습은 `RUN_TRAINING = True`와 `EPOCHS = 1`로 실행하고, 본 실험 전 목표 epoch로 설정하세요. 기본 저장 주기는 100 epoch이며 final epoch checkpoint도 저장합니다.

```powershell
.\.venv\Scripts\python.exe -m src.train_lite_dpas --compression-ratio 4 --epochs 100 --batch-size 200 --seed 42
```

체크포인트와 validation 학습 이력은 실행 태그가 붙은 새 폴더(예: `runs/litedpas_v2_indoor_cr4_100ep_20260928_120000/`)에 생성됩니다. 기존 결과 폴더를 덮어쓰지 않습니다. 짧은 동작 확인에는 `--epochs 1 --run-tag v2_try_01`을 지정하세요. GPU 메모리가 부족하면 batch size를 64 또는 32로 낮춥니다. 지원 압축률은 1/4, 1/16, 1/32, 1/64입니다.

학습이 끝난 뒤 validation 일부 샘플의 NMSE, 파라미터 수, Conv/Linear 연산량 추정치, batch-1 지연을 측정합니다. 학습 명령이 출력한 checkpoint 경로의 run-tag를 아래 경로에 넣으세요.

```powershell
.\.venv\Scripts\python.exe -m src.profile_lite_dpas --checkpoint runs/litedpas_v2_indoor_cr4_100ep_<실행태그>/best.pt --samples 512
```

`profile_validation.json`은 같은 run 폴더에 저장됩니다. MAC 추정은 Conv2d와 Linear만 세며 FFT·활성화·정규화·보간 비용을 제외합니다. 보고서의 latency 실측과 함께 해석하세요. 이 빌드는 아직 결과가 검증된 논문 모델이 아니라, 사용자가 성능과 비용을 시험하기 위한 출발점입니다.

- `00_setup_and_data.ipynb`: 환경, 데이터 경로, 검증 split만 점검합니다. 학습과 test 평가는 하지 않습니다.
- `01_csinet_baseline.ipynb`: CsiNet smoke check와 기준선 학습입니다. test split을 열지 않습니다.
- `03_crnet_100ep.ipynb`, `04_transnet_100ep.ipynb`: 저자 모델 구조에 공통 train/validation 전용 학습기를 적용합니다. 저자 공식 스크립트가 epoch마다 test를 사용하므로, 이 노트북은 test 누수를 피하도록 별도 구현했습니다.
- `05_compare_validation.ipynb`: 세 모델의 validation best score를 표로 읽습니다. 모델별 학습 recipe가 달라 초기 경향 확인에만 사용합니다.
- `02_final_test.ipynb`: 선택된 체크포인트의 최종 test 평가입니다. 설정을 확정하기 전에 실행하지 않습니다.

