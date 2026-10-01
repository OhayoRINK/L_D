# 공개 기준선 저장소 점검

## 가져온 저장소

| 기준선 | 공식 저장소 | clone commit | 공식 저장소의 학습/실행 조건 |
|---|---|---|---|
| CsiNet | [sydney222/Python_CsiNet](https://github.com/sydney222/Python_CsiNet) | `9b921a9098594b0eeec102e83055e0da500d1834` | 저자 코드 README는 Python 3.5/3.6, Keras 2.1.1 이상, TensorFlow 1.4 이상을 적고 있음. Indoor/outdoor 512차원 저장 모델 파일이 저장소에 포함됨. |
| CRNet | [Kylin9511/CRNet](https://github.com/Kylin9511/CRNet) | `22c9abd61490af46af63d737ca470f0943efa973` | README 학습 재현은 cosine annealing + warm-up 2,500 epochs, batch 200. 문서는 PyTorch 1.2–1.6을 요구함. |
| TransNet | [Treedy2020/TransNet](https://github.com/Treedy2020/TransNet) | `f362e33174cadd9ceae437b23512d20a1881f2a6` | README 저자 결과는 constant LR 1e-4, 400/1,000 epochs. 포함된 환경은 Python 3.8, PyTorch 1.6, CUDA 10.1 기반 Linux 환경임. |

## COST2100 입력 변환

- CsiNet 저자 학습 코드: `HT`를 float32로 바꾸고 `[N, 2, 32, 32]`로 reshape
- CRNet/TransNet `dataset/cost2100.py`: 같은 `HT` 변환을 PyTorch `.view(N, 2, 32, 32)`로 수행
- 세 구현 모두 COST2100 angular-delay 입력과 0.5 중심 이동을 사용합니다.
- CRNet/TransNet test loader는 추가로 `DATA_HtestF*_all.mat`의 `HF_all`을 불러와 주파수 영역 `rho` 평가에 씁니다. 이 프로젝트는 해당 파일도 복사해 SHA-256을 확인했습니다.

## NMSE 구현

CRNet과 TransNet의 `utils/statics.py`는 prediction/ground truth 각 real·imaginary 채널에서 0.5를 빼고, sample별 `sum(error power)/sum(reference power)`를 구한 뒤 평균하고 `10*log10`합니다. CsiNet 저자 evaluation도 sample별 delay-domain error/power 비율의 평균을 보고합니다. 새 평가 코드에서 동일 공식을 사용합니다.

## 새 프로젝트 실험 어댑터 상태

- CRNet/TransNet 원본 clone을 수정하지 않고, 원 모델 정의를 새 어댑터에서 불러오는 forward/backward 확인을 마쳤습니다. 두 모델 모두 입력 `[2,2,32,32]`에 대해 같은 출력 shape를 반환합니다.
- 저자 `main.py`는 매 epoch test loader를 평가하므로 경향성 노트북은 재사용하지 않습니다. 새 공통 runner는 train/validation만 읽고 test는 최종 평가 노트북에 남깁니다.
- CRNet 100 epoch 경향성 조건은 constant LR `1e-3`, TransNet은 constant LR `1e-4`입니다. 이는 각각 저자 스크립트의 constant scheduler 설정을 따르며, CRNet README의 2500 epoch warmup/cosine 최종 recipe와 같지 않습니다.
- CsiNet PyTorch 구현은 저자 Keras 구조를 바탕으로 만든 초기 포트입니다. 원 체크포인트와 예측 수치 동등성을 확인하지 않았으므로 저자 공식 재현과 구분합니다.
- 원 저장소 clone은 upstream 참조용으로 보존하며 실행용 파일은 프로젝트 `src/`와 `notebooks/`에 따로 둡니다.
- 아직 확인할 일: 각 저장소의 LICENSE 및 재배포 조건을 기록하고, 공식 checkpoint 기반 결과와 자체 학습 결과를 구분합니다.

## README 수치의 취급

저자 README의 NMSE/latency는 저장소 저자 보고 결과입니다. 이 프로젝트에서 같은 데이터와 평가 규약으로 실행하기 전까지 우리 baseline 결과로 쓰지 않습니다.
