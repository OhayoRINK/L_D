# CSI 피드백용 경량 복원 모델 연구

## 현재 연구 방향

DPAS-CsiNet의 공간·지연·스펙트럼 정보 활용 아이디어를 유지하면서, depthwise 이중축 블록과 저차원 latent를 이용해 **같은 feedback budget에서 복원 정확도와 계산 비용을 함께 개선**하는 방향을 우선 검증합니다.

첫 연구 질문은 다음과 같습니다.

> 같은 CSI 피드백 차원에서 Lite-DPAS가 CsiNet 및 DPAS-CsiNet보다 낮은 복원 NMSE와 적은 추론 비용을 달성할 수 있는가?

새 모델의 성능은 아직 검증되지 않았습니다. 평균 NMSE와 함께 파라미터 수, Conv/Linear MAC 추정치, batch-1 실측 지연을 기록하고, 이후 CRNet·TransNet 및 미학습 채널 조건으로 평가를 확장합니다.

## 시작 범위

- 데이터: COST2100 공식 benchmark
- 첫 환경: Indoor, 압축률 CR=1/4
- 첫 기준선: CsiNet 및 기존 DPAS-CsiNet
- 첫 모델: Lite-DPAS, COST2100 indoor, CR=1/4
- 다음 전이 대상: CR=1/16·1/32·1/64, outdoor, CRNet, TransNet
- 1차 품질 지표: 복소수 NMSE (dB)
- 1차 시스템 지표: batch-1 복원 지연, 파라미터 수, 연산량

NMSE 개선, 파라미터 감소, 연산량 감소, 지연 감소를 각각 측정합니다. 한 지표의 변화만으로 다른 지표까지 개선됐다고 주장하지 않습니다.

## Lite-DPAS 테스트 빌드

대화형 실험은 [Lite-DPAS V2 노트북](notebooks/08_lite_dpas_v2_experiment.ipynb)에서 설정·학습·validation 프로파일 순서로 진행할 수 있습니다. 학습은 100 epoch 간격으로 재개 가능한 checkpoint를 저장하며, Colab 설정은 [실험 README](notebooks/README.md#google-colab-실행)에 있습니다. V2는 learned down/up-sampling 구조를 시험하는 새 버전이며, 결과가 확인되기 전에는 성능 우위나 경량성을 연구 결과로 간주하지 않습니다.

## 진행 순서

데이터 출처와 복사 검증 결과는 [DATA_MANIFEST.md](DATA_MANIFEST.md), 기준 모델과 loader 점검은 [BASELINE_AUDIT.md](BASELINE_AUDIT.md), 고정 평가 규칙은 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)에 있습니다. sLLM 관련 문서는 별도 기존 연구 계획으로 남아 있습니다.

## 실험 실행

기준선 노트북은 [notebooks/README.md](notebooks/README.md)에 안내되어 있습니다. Lite-DPAS는 `python -m src.train_lite_dpas`로 실행하며 학습 중 test split을 읽지 않습니다. `02_final_test.ipynb`는 설정을 확정한 후 최종 단계에서만 실행합니다.

## 현재 상태

새 연구 폴더를 만들고 COST2100 `.mat` 파일을 복사했습니다. 복사본 12개의 SHA-256이 원본과 모두 일치합니다. 이전 프로젝트의 모델 체크포인트·oracle 결과·sLLM 학습자료는 이 폴더에 포함하지 않았습니다. 기준 모델 소스와 새 실행 환경을 고정한 뒤, 깨끗한 baseline부터 시작합니다.
