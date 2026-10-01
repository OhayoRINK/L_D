# 실행 기록

> 새 연구 폴더 최초 작성: 2026-09-28

## 현재 단계

새 연구 초점의 feasibility gate — 고정 CR=1/4에서 heterogeneous decoder routing의 oracle 여지 검증

## 완료

- [x] 기존 폴더와 분리된 `CSI_SLLM_RESEARCH` 생성
- [x] 기존 COST2100 `.mat` 12개를 복사. 원본 파일은 이동/수정하지 않음
- [x] 원본과 복사본 12개 SHA-256이 모두 일치함을 확인
- [x] 기존 체크포인트, oracle labels, JSONL 학습자료는 새 프로젝트에 포함하지 않음
- [x] sLLM 연구 목표를 “복원 모델 대체”에서 “품질–비용 적응 정책”으로 명시
- [x] 데이터·평가·단계별 연구 문서를 새로 작성
- [x] CsiNet, CRNet, TransNet 저자 저장소를 `baselines/`에 새로 clone
- [x] 세 저장소의 현재 commit SHA 및 COST2100 reshape/NMSE 코드를 대조
- [x] 로컬/Colab 공통 `.ipynb` 실행 경로 초안 작성 (환경 점검, CsiNet smoke/full training, 최종 test 분리)
- [x] 전체 학습은 노트북 기본값에서 꺼져 있고 test 평가는 별도 노트북으로 격리
- [x] 노트북 JSON과 Python 셀 문법 파싱 확인
- [x] 두 LLM 논문 외 CSI adaptive rate/model routing·LLM CSI 선행 조사 및 novelty 위험 기록
- [x] 연구 질문을 고정 feedback rate에서의 heterogeneous CSI decoder compute routing 및 architecture transfer로 구체화
- [x] validation 전용 routing feasibility 노트북 작성 (sample NMSE, batch-1 latency, oracle upper bound; test 미사용)

## 아직 안 한 일

- [ ] baseline 저장소의 LICENSE/사용 조건을 기록하고 실행 변경은 upstream과 분리
- [x] 각 원 코드가 같은 `HT`/COST2100 split을 어떤 입력으로 바꾸는지 대조
- [x] 새 프로젝트용 Python/CUDA 환경 고정 (로컬 RTX 3070에서 CUDA 인식)
- [x] 로컬 `.venv`에 ipykernel 설치 및 `Python (CSI_SLLM_RESEARCH)` 커널 등록
- [x] 저자 CRNet/TransNet 구조를 불러오는 새 모델 adapter 작성 (원본 clone은 변경하지 않음)
- [x] CRNet/TransNet 각각 100 epoch 노트북 및 train/validation 전용 공통 runner 작성
- [x] CRNet/TransNet의 무작위 입력 forward/backward 확인: 출력 모두 `[2, 2, 32, 32]`, loss gradient 생성 확인
- [x] 세 모델 validation checkpoint 요약 노트북 작성
- [ ] Colab에서 Drive 마운트 및 GPU 런타임으로 노트북 수동 실행
- [ ] 저자 baseline 재현과 자체 baseline 결과를 구분해 산출
- [ ] 기존 폴더에 있던 점수나 학습자료를 새 실험의 결과로 사용하지 않음

## 다음 작업

1. 사용자가 `notebooks/06_router_feasibility.ipynb`를 위에서 아래로 실행합니다. 필요한 세 checkpoint가 있는지 확인한 뒤 validation 전용 결과를 만듭니다.
2. `runs/router_feasibility_indoor_cr4/oracle_summary.json`의 oracle gain과 모델별 latency/품질 편차를 확인합니다. oracle 이득이 작으면 다음 정책 모델 학습으로 넘어가지 않고 행동 공간을 다시 설계합니다.
3. oracle 여지가 충분하면 규칙/LUT 및 작은 MLP를 먼저 구현하고, 입력 정보 누출·validation 재사용을 통제하는 정책 평가 split을 설계합니다.
4. sLLM은 비언어 기준선의 end-to-end 결과를 확인한 뒤 동일 조건에서 비교합니다.
5. 모델 recipe를 공정하게 재학습/선택하고 일반화 평가를 확정하기 전 test split을 실행하지 않습니다.
