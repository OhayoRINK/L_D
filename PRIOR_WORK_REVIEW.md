# CSI·LLM 연구 선행연구 조사 (초기 스코핑 리뷰)

조사일: 2026-09-28  
대상 프로젝트: COST2100 기반 CSI 피드백/복원 및 sLLM 아이디어  
목적: 현재 아이디어와 가까운 연구를 찾아 중복 위험을 평가하고 다음 연구 질문을 좁힌다.

> **범위와 한계:** 아래는 IEEE/arXiv/학회·저널 페이지에서 제목·초록·공개 본문을 검색해 만든 초기 스코핑 리뷰다. 모든 학술 데이터베이스의 검색식을 사전 등록하고 중복 제거·전문 심사를 수행한 체계적 문헌고찰은 아니다. 특히 매우 최근 arXiv 글은 peer review 상태, 정식 출판판, 인용/후속 연구를 추가로 확인해야 한다. 그러므로 novelty를 “확정”하는 보고서가 아니라, 현재 제안의 중복 위험을 찾고 연구 설계를 보완하는 보고서다.

## 핵심 결론

1. **CSI 복원기에 LLM을 직접 적용하는 아이디어는 이미 복수의 연구가 다뤘다.** 2025년 LLM 기반 CSI 피드백 연구, 2026년 LLMCsiNet, LLM4FB, CoFi-CLM 등이 있다. “상용 사전학습 LLM을 CSI에 맞게 fine-tuning한다” 또는 “CSI 토큰을 Transformer에 넣어 복원한다”는 큰 방향만으로 novelty를 주장하기 어렵다.
2. **학습된 모델을 상태에 따라 바꾸거나 피드백률을 조절하는 CSI 연구도 존재한다.** 서로 다른 압축률 모델 선택, 한 모델의 목표 피드백 비트 조절, adaptive rate CSI 피드백이 선행돼 있다. 따라서 프로젝트의 기존 가설인 “작은 정책 모델이 기존 CSI 복원기의 실행을 조절한다”도 넓은 표현만으로는 독창성이 보장되지 않는다.
3. 확인한 선행과 구체적으로 구별할 후보는 **피드백 bit rate를 고르는 게 아니라, 서로 다른 기성 CSI 복원기/실행경로 사이의 계산을 샘플별로 배분하고, 보지 못한 아키텍처에도 정책이 전이되는지 검증하는 문제**다. 다만 이 후보도 일반적인 adaptive inference·model routing과 구별되는 알고리즘 기여가 있어야 한다.
4. sLLM 자체를 기여로 내세우기보다, “작은 언어모델이 MLP/규칙보다 실제로 나은가?”를 실험 질문으로 취급해야 한다. 모델이 무거워져 end-to-end 비용이 악화되면 sLLM이 필요하지 않다는 결과를 받아들여야 한다.

## 선행연구 지도

| 흐름 | 대표 연구 | 이미 다룬 내용 | 현재 아이디어에 대한 위험 |
|---|---|---|---|
| 사전학습 LLM 기반 CSI 복원 | Cui et al., 2025; LLMCsiNet, 2026; LLM4FB, 2026 | CSI 전처리/임베딩, LLM 또는 GPT 계열로 CSI 복원/예측, 제한된 fine-tuning | “LLM을 CSI decoder에 붙인다”만으로는 매우 높은 중복 |
| CSI용 토큰 언어모델 | CoFi-CLM, 2026 | coarse/fine CSI codebook token, 제한 bit feedback, 기지국 Transformer의 누락 fine token 예측, dual path fusion | discrete CSI tokens→언어모델 복원과 직접 중복 |
| 여러 LLM의 결합 | Cui et al., 2026 preprint의 MoA | 다수 LLM의 CSI 복원을 경량 집계기로 결합 | “여러 LLM 중 고르거나 결과를 합친다”와 가까움 |
| 적응형 CSI bit/rate 제어 | ABLNet, WideNLNet-CA, Multi-Rate Variable-Length CSI Compression | 다양한 CSI 길이/bit 수 처리, 목표 피드백률 조정, 한 모델의 CR 적응 | 동적 압축률/bit budget 조정만으로는 중복 |
| CSI 모델 뱅크 선택과 시스템 제어 | RL-Guided Autoencoder Switching, 2026 | 미리 학습된 multi-rate autoencoder 중 사용자별 CR 선택, system reward에 overhead·SINR·adaptation cost 포함 | “상태 보고 정책기가 여러 CSI 모델을 고른다”와 높은 위험 |
| 경량화·증류 | Cui et al., TCOM 2024; codeword mimic learning, 2022 | 큰 CSI teacher에서 작은 student로 출력/표현 지식 전달, 추론 비용 절감 | 일반적인 LLM teacher→작은 CSI student는 그 자체로 새롭지 않을 수 있음 |
| 환경 전이/도메인 적응 | CSI-TransNet, 2022; near-real-time domain adaptation, 2026 | 새 채널환경 적응, 작은 변환 모듈, 데이터 부족/분포 이동 대응 | “상황 변화에 알아서 대응”이라는 넓은 주장과 겹침 |
| 범용 적응 추론 | early-exit/latency-aware inference 등 | 입력별 계산량·깊이·exit를 조정해 품질-지연시간 trade-off | CSI 전용으로 옮기는 것만으론 약함; 도메인 기여가 필요 |

## 세부 검토한 주요 연구

### A. LLM을 CSI 복원에 직접 사용하는 연구

- **Exploring the Potential of Large Language Models for Massive MIMO CSI Feedback** (Cui et al., 2025, GLOBECOM 2025로 기록): CSI 신호용 전처리·임베딩·후처리를 두고 사전학습 LLM의 denoising 능력을 활용한다. 현재 확인한 arXiv 초록 및 학회 메타데이터 기준으로, LLM을 CSI 피드백 복원에 쓰는 선행이 LLMCsiNet보다 먼저 있다. [arXiv](https://arxiv.org/abs/2501.10630) · [GLOBECOM 2025 기록](https://dblp.org/rec/conf/globecom/CuiGWJT25.html)
- **LLMCsiNet** (Wu et al., 2026): 중요한 CSI 원소만 보내는 self-information 마스킹/선택과 GPT-2 계열 Transformer 층 기반 복원을 제안한다. 단말은 작게, 기지국은 수천만 이상의 파라미터가 될 수 있는 비대칭 설계다. [arXiv](https://arxiv.org/abs/2603.02686)
- **LLM4FB: A One-Sided CSI Feedback and Prediction Framework for Lightweight UEs via Large Language Models** (Sensors, 2026): 경량 UE를 전제로 사전학습 LLM을 이용한 CSI feedback/prediction을 제안하며, pretrained LLM의 일부 파라미터를 미세조정하는 접근을 설명한다. 따라서 “한쪽에만 LLM을 두어 가벼운 UE를 만든다”도 구별점으로 약하다. [저널 페이지](https://www.mdpi.com/1424-8220/26/2/691)
- **Leveraging Pre-Trained Large Language Models for CSI Feedback in Massive MIMO Systems** (Cui et al., 공개 preprint): CSI 맞춤 모듈과 MoA(여러 LLM 결과를 합치는 접근)를 다룬다. 여러 LLM의 상호 보완성, 미관측 환경 일반화 등을 주장하므로 LLM ensemble/aggregation도 단순 차별점이 되기 어렵다. [공개 PDF](https://d197for5662m48.cloudfront.net/documents/publicationstatus/269336/preprint_pdf/6d1280e4225f4423ce54ffb1328e47b4.pdf)
- **CoFi-CLM** (Zhang et al., 2026 preprint): UE의 거친 양자화 CSI 토큰을 조건으로 BS CLM이 미전송 fine token을 예측하고, 두 복원경로를 결합한다. “coarse CSI token→fine token을 LLM이 생성”은 매우 직접적인 중복이다. [arXiv](https://arxiv.org/abs/2609.15865)

**판정:** 완전히 새로운 sLLM을 CSI 복원 전용으로 사전학습/fine-tuning하는 것은 가능하지만, 그 자체가 빈 연구 공백은 아니다. 성능이 더 좋고 시스템 총비용이 줄어드는 명확한 원인과 공정한 비교가 필요하다.

### B. 동적 CSI 피드백·모델 선택

- **An Adaptive CSI Feedback Model Based on BiLSTM for Massive MIMO-OFDM Systems** (2024): CSI 길이와 피드백 bit 수가 달라지는 상황을 지원하고, feedback bit control unit 및 목표 성능 기반 bit 조절 알고리즘을 둔다. [arXiv](https://arxiv.org/abs/2408.06359)
- **Multi-Rate Variable-Length CSI Compression for FDD Massive MIMO** (Park et al., ICASSP 2024): 다중률·가변길이 CSI 압축을 다룬다. 즉 “한 모델이 여러 압축률을 지원한다”는 제안도 이미 있다. [ICASSP 프로그램/논문 정보](https://cmsworkshops.com/ICASSP2024/view_paper.php?PaperNum=9956) · [DOI](https://doi.org/10.1109/ICASSP48485.2024.10448212)
- **Deep Learning-Based Rate-Adaptive CSI Feedback for Wideband XL-MIMO Systems in the Near-Field Domain** (2025): 하나의 모델에서 목표 압축률에 따라 feature selection을 조정하는 모듈을 다룬다. 데이터/채널 도메인은 COST2100과 다르지만 rate adaptation 자체가 새로운 것은 아님을 보여준다. [arXiv](https://arxiv.org/abs/2508.00626)
- **System-Aware Adaptive CSI Feedback via RL-Guided Autoencoder Switching in Multi-User MIMO System** (Ansarifard et al., 2026 preprint): 미리 학습한 여러 rate의 AE 중 RL agent가 사용자별 모델을 선택한다. reward에 SINR, feedback overhead, 모델 adaptation 계산비용을 함께 포함한다. 현재 연구계획의 “CSI 상황을 보고 정책이 기존 모델을 고른다”와 가장 가까운 추가 선행이다. [arXiv](https://arxiv.org/abs/2607.25588)

**판정:** 기존 `RESEARCH_PLAN.md`의 정책기 방향은 LLMCsiNet/CoFi-CLM과는 목적이 다르지만, RL-guided multi-rate AE switching과는 일부 핵심이 겹친다. “모델 선택”만을 novelty로 주장하지 말아야 한다.

### C. 경량화·증류·환경 적응

- **Lightweight Neural Network With Knowledge Distillation for CSI Feedback** (Cui et al., IEEE TCOM 2024): 큰 teacher의 CSI 복원 결과를 작은 student에 증류하는 방법을 연구한다. 파라미터·FLOPs·추론시간을 함께 고려한다. 따라서 “LLM을 teacher로 쓰고 작은 CSI 모델을 만든다”는 단순 구상도 기존 CSI distillation과 세밀히 비교해야 한다. [arXiv](https://arxiv.org/abs/2210.17113) · [저자 소속 연구 포털](https://researchportal.hkust.edu.hk/en/publications/lightweight-neural-network-with-knowledge-distillation-for-csi-fe/)
- **Better Lightweight Network for Free: Codeword Mimic Learning for Massive MIMO CSI feedback** (Lu et al., 2022): 경량 feedback network의 codeword를 교사 학습에 활용하고, 추론비용 증가 없이 성능을 높이려는 CSI 전용 증류를 제안한다. [arXiv](https://arxiv.org/abs/2210.16544)
- **Deep Learning for Efficient CSI Feedback in Massive MIMO: Adapting to New Environments and Small Datasets** (Liu et al., 2022): 새 환경에 pretrained CSI 모델을 효율적으로 적응시키는 plug-in 변환 모듈과 소규모 데이터 증강을 제안한다. [arXiv](https://arxiv.org/abs/2211.14785)
- **Compressed-CSI Feedback With Near Real-Time Domain Adaptation** (IEEE TCOM 2026): 변화한 채널 환경에서 UE가 새 CSI로 sparse linear codebook을 재구축해 BS에 갱신하는 domain adaptation을 제시한다. [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11543360)

## 연구 아이디어와 중복 위험 재평가

| 제안 | 현재 문헌과의 관계 | 예비 평가 |
|---|---|---|
| 사전학습 LLM을 CSI decoder로 붙이기 | 2025 LLM CSI, LLMCsiNet, LLM4FB와 직접 유사 | **중복 위험 매우 큼** |
| UE는 작고 BS에서 LLM이 fine CSI를 복원 | LLMCsiNet·CoFi-CLM·LLM4FB가 가까움 | **중복 위험 매우 큼** |
| CSI discrete token을 sLLM이 coarse-to-fine 생성 | CoFi-CLM에 직접 가까움 | **중복 위험 매우 큼** |
| LLM 여러 개를 ensemble/결합 | preprint MoA가 이미 다룸 | **중복 위험 큼** |
| 적응형 CR/피드백 비트 선택 | ABLNet, WideNLNet-CA, RL AE switching 등 존재 | **중복 위험 큼** |
| 임의의 기성 CSI decoder 간 샘플별 계산량 배분 | CSI rate/model switching 및 범용 adaptive inference와 겹치는 영역이 있음 | **가능성은 있으나 차별 기여를 더 좁혀야 함** |
| 작은 sLLM이 CsiNet·CRNet·TransNet 등 *학습 시 보지 않은 구조*로 일반화하는 decoder-agnostic controller | 확인된 RL CSI 연구는 multi-rate AE bank/CR 선택이 중심. 아키텍처 간 전이와 제약 아래 cross-model generalization은 별도 확인 필요 | **현재 가장 구체적인 빈틈 후보; 아직 novelty 미확정** |
| CSI 도메인 LLM이 unseen 채널에서 신뢰도 보정/실패 감지 후 검증된 fallback 결정 | domain adaptation과 adaptive inference에 근접; CSI task-specific risk/constraint 설계 가능 | **탐색 후보; 강한 문헌조사 필요** |

## 권장 연구 질문(가설 수준)

> **학습에 포함되지 않은 CSI 복원 아키텍처와 채널 조건에서도, 작은 정책기가 현재 입력의 난이도/신뢰도와 전체 시스템 예산을 이용해 복원 경로를 선택할 수 있는가? 이 정책이 고정 경로, CR 선택형 정책, LUT, MLP보다 품질-지연시간 Pareto 성능을 높이는가?**

이는 단순히 “모델을 바꿔 준다”가 아니라 아래 세 가지를 묶어야 의미가 생긴다.

1. **행동 공간 차별:** bit rate/CR 선택만 하지 않고, CsiNet·CRNet·TransNet 등 구조가 다른 복원기와 그 안의 선택적 계산 경로를 다룬다. 기존 모델을 전부 실행한 후 최선값을 고르는 측정은 배포 지연시간에서 제외하지 않는다.
2. **전이성 평가:** 어떤 모델에서 정책을 학습하고 다른 모델을 보지 않은 채 평가하는 leave-one-architecture-out 실험. 사용자/환경 split도 분리해 입력 누출을 막는다.
3. **리스크·비용을 포함한 선택:** 예상 NMSE만 아니라 측정한 end-to-end 지연시간, 메모리, UE/BS 구분, 피드백 bit budget을 함께 고려한다. 낮은 confidence/OOD/정책 timeout 때 안전한 기본 복원기로 되돌아간다.

LLM/sLLM은 **검증할 후보 정책기**다. 똑같은 입력을 받는 규칙, LUT, 선형/작은 MLP, 경량 Transformer 또는 RL baseline을 둬서 LLM 사용의 추가 가치를 분리한다. 특히 작은 구조화 입력에서 MLP가 동등하면, LLM보다 MLP가 합리적이고 그 자체가 중요한 결론이다.

## 다음 조사·실험 순서

### 문헌을 더 확인할 항목

1. RL-guided autoencoder switching 원문: 사용 데이터, 모델 뱅크, 관측값, action/reward, 계산량 계측, random seed와 baseline을 전부 확인한다.
2. LLM4FB 및 MoA 논문 정식판/부록: 어떤 pretrained weights와 backbone인지, full-system parameter/latency, COST2100 protocol, 비교 기준을 확인한다.
3. CSI adaptive bitrate / multiple compression rate / rate-distortion / variable-length feedback을 IEEE Xplore·arXiv·Google Scholar에서 동의어까지 확장 검색한다.
4. neural network adaptive inference, early exit, mixture-of-experts, cascade routing, model selection의 peer-reviewed 연구를 확인하고 CSI 도메인에 실제로 적용한 논문을 citation chasing한다.
5. 모든 논문의 상태(preprint, peer-reviewed publication), 날짜, 버전, 공개 코드와 결과 재현성 표기.

### 프로젝트 실험의 gate

1. 행동 선택지가 실제로 있는지 확인: 행동별 단일 샘플 NMSE, batch-1 latency, peak memory를 측정한다.
2. oracle 상한 계산: 정답 CSI를 이용해 학습 split에서만 샘플별 최선 행동을 표시하고, 얻을 수 있는 이득 자체가 있는지 본다.
3. 작은 baseline부터: fixed fast/accurate, LUT, MLP. 정책을 추가해도 품질-비용 Pareto 이득이 없으면 LLM 구현 전에 질문을 재검토한다.
4. adaptive feedback/rate 연구와 정면 비교 가능한 경우 그 정책을 baseline으로 추가한다.
5. sLLM 후보는 마지막에 같은 행동·입력·데이터·예산으로 비교하고 end-to-end 비용을 보고한다.
6. COST2100 외부 도메인 데이터 없이 “환경 일반화”를 주장하지 않는다. COST2100 내 split은 외부 무선 환경 일반화의 대체가 아니다.

## 참고 문헌 / 링크

1. Cui et al., *Exploring the Potential of Large Language Models for Massive MIMO CSI Feedback*, arXiv:2501.10630; GLOBECOM 2025. https://arxiv.org/abs/2501.10630
2. Wu et al., *Large Language Model Empowered CSI Feedback in Massive MIMO Systems*, arXiv:2603.02686. https://arxiv.org/abs/2603.02686
3. Zhang et al., *CoFi-CLM: A Coarse-to-Fine Channel Language Model for Finite-Bit CSI Feedback*, arXiv:2609.15865. https://arxiv.org/abs/2609.15865
4. *LLM4FB: A One-Sided CSI Feedback and Prediction Framework for Lightweight UEs via Large Language Models*, Sensors 26(2), 691 (2026). https://www.mdpi.com/1424-8220/26/2/691
5. Cui et al., *Leveraging Pre-Trained Large Language Models for CSI Feedback in Massive MIMO Systems*, 공개 preprint. https://d197for5662m48.cloudfront.net/documents/publicationstatus/269336/preprint_pdf/6d1280e4225f4423ce54ffb1328e47b4.pdf
6. Ansarifard et al., *System-Aware Adaptive CSI Feedback via RL-Guided Autoencoder Switching in Multi-User MIMO System*, arXiv:2607.25588. https://arxiv.org/abs/2607.25588
7. Shen et al., *An Adaptive CSI Feedback Model Based on BiLSTM for Massive MIMO-OFDM Systems*, arXiv:2408.06359. https://arxiv.org/abs/2408.06359
8. Park, Do, Lee, *Multi-Rate Variable-Length CSI Compression for FDD Massive MIMO*, ICASSP 2024, DOI:10.1109/ICASSP48485.2024.10448212. https://doi.org/10.1109/ICASSP48485.2024.10448212
9. Liu et al., *Deep Learning-Based Rate-Adaptive CSI Feedback for Wideband XL-MIMO Systems in the Near-Field Domain*, arXiv:2508.00626. https://arxiv.org/abs/2508.00626
10. Cui et al., *Lightweight Neural Network With Knowledge Distillation for CSI Feedback*, IEEE TCOM (2024), preprint arXiv:2210.17113. https://arxiv.org/abs/2210.17113
11. Lu et al., *Better Lightweight Network for Free: Codeword Mimic Learning for Massive MIMO CSI feedback*, arXiv:2210.16544. https://arxiv.org/abs/2210.16544
12. Liu et al., *Deep Learning for Efficient CSI Feedback in Massive MIMO: Adapting to New Environments and Small Datasets*, arXiv:2211.14785. https://arxiv.org/abs/2211.14785
13. *Compressed-CSI Feedback With Near Real-Time Domain Adaptation*, IEEE Transactions on Communications 74 (2026). https://ieeexplore.ieee.org/abstract/document/11543360
14. Tan et al., *Empowering Adaptive Early-Exit Inference with Latency Awareness*, AAAI 2021. https://ojs.aaai.org/index.php/AAAI/article/view/17181
