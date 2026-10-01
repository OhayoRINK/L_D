# Lite-DPAS V5 실험 및 비교 정리

> **목적**: DPAS의 공간·지연·스펙트럼 활용 아이디어를 유지하면서, 동일 피드백 예산에서 CSI 복원 성능을 확보하고 모델 크기와 추론 비용을 낮추는 경량 모델을 검토한다.
>
> **버전 기준**: 이 문서는 프로젝트의 `LiteDPASCsiNetV5` 구현과 사용자 대화에서 공유된 실험 수치를 중심으로 정리했다. 아직 정식 논문용 동일 조건 비교가 완료되지 않았으므로, 측정값·추정값·문헌 보고값을 구분한다.

## 1. 핵심 요약

- Lite-DPAS V5는 DPAS의 **공간/지연 이중 축 처리**와 **주파수 정보 활용**을 가벼운 depthwise 연산, 저해상도 특징 처리, 대역별 spectral gate로 재구성했다.
- CR=1/4 Indoor에서 약 **−19.4 dB**, CR=1/64 Indoor에서 약 **−6.85 dB**가 공유되었다. Outdoor CR=1/64에서는 약 400 epoch 시점 **−2.301 dB**가 공유되었다.
- Outdoor CR=1/4 약 **−10 dB**는 현재 **예상값**이지 실측값이 아니다.
- 코드 구조로 계산한 Lite-DPAS V5 파라미터는 CR=1/4에서 **1,057,700개**다. CsiNet 포트의 **2,103,116개**보다 약 **49.7% 적다**.
- 현재 성능은 CsiNet보다 나은 실험 결과를 보이지만, 문헌의 CRNet, CsiNet+, DCRNet, TransNet 및 원본 DPAS 보고값과 같은 조건에서 비교한 결과는 아니다.

## 2. 모델 구조와 계승 관계

| DPAS 구성 | DPAS의 기능 | Lite-DPAS V5 구현 | 관계 |
|---|---|---|---|
| Dual-Path | 3×3 공간 경로와 1×W 지연 경로를 병렬 처리 후 융합 | depthwise 3×3 및 depthwise 1×15를 병렬 처리하고 1×1 convolution으로 융합 | 이중 축 처리 아이디어를 유지하고 convolution을 경량화 |
| AFG | FFT 크기 특징으로 입력에 따라 feature를 주파수 측면에서 재가중 | 지연축 FFT 특징을 4개 대역으로 모아 bandwise gate를 생성 | FFT 기반 적응 게이팅 목적을 대역 게이팅으로 재구성 |
| SRB | decoder에서 FFT 영역 feature를 convolution으로 정제하고 IFFT/잔차 결합 | 저해상도 decoder의 bandwise spectral gate와 Lite block을 사용 | spectral 보정 목적은 일부 계승하지만 SRB의 복소수 convolution은 그대로 구현하지 않음 |
| 압축 | FC 변환으로 제한된 길이의 codeword 생성 | feature를 4×16×16으로 축소한 뒤 FC로 codeword 생성 | CR별 codeword 예산은 유지하고 bottleneck 이전 feature 크기를 줄임 |
| Decoder 해상도 | 32×32 전 해상도에서 refinement 중심 | 16×16에서 대부분 정제 후 PixelShuffle로 32×32 복원 | 더 작은 해상도에서 처리해 계산량 절감 |

### Lite-DPAS V5 데이터 흐름

`CSI 2×32×32 → stem/dual-axis blocks → learned downsample 16×16 → encoder spectral gate → 4×16×16 latent → FC codeword → decoder Lite block + spectral gate → PixelShuffle upsample → CSI 2×32×32`

CR=1/4, 1/16, 1/32, 1/64에 해당하는 실수 codeword 차원은 각각 512, 128, 64, 32다.

## 3. Lite-DPAS 실험 기록

### 3.1 공유된 성능값

| 데이터셋 | 압축률 | Lite-DPAS V5 NMSE | 상태/주의 |
|---|---:|---:|---|
| Indoor | 1/4 | 약 −19.4 dB | 사용자 공유 실험값. epoch 및 평가 split 세부 정보 미기록 |
| Indoor | 1/64 | 약 −6.85 dB | 사용자 공유 실험값. epoch 및 평가 split 세부 정보 미기록 |
| Outdoor | 1/64 | 약 −2.301 dB | 약 400 epoch 시점 사용자 공유값 |
| Outdoor | 1/4 | 약 −10 dB | 사용자 예상값. **실측 결과가 아님** |
| Indoor/Outdoor | 1/16, 1/32 | — | Lite-DPAS 실측값 미공유 |

### 3.2 과거 버전의 진행 기록

| 시점/버전 | 학습 정보 | 보고된 validation NMSE |
|---|---|---:|
| Lite-DPAS 초기 버전 | CR=1/4, 100 epoch | 약 −12.8 dB |
| Lite-DPAS 중간 버전 | 약 430 epoch 시점 | 약 −17.507 dB |
| Lite-DPAS V5 | 버전명으로 공유 | 약 −19.4 dB |

버전별 구조·학습 설정이 다르므로 이 값들은 하나의 연속 학습곡선으로 해석하지 않는다.

### 3.3 모델 크기 및 계산량

프로젝트의 CsiNet 포트 및 Lite-DPAS V5 구조로 계산한 파라미터 수:

| 압축률 | codeword 차원 | Lite-DPAS V5 parameters | CsiNet 포트 parameters | 차이 |
|---:|---:|---:|---:|---:|
| 1/4 | 512 | 1,057,700 | 2,103,116 | Lite-DPAS 약 49.7% 감소 |
| 1/16 | 128 | 270,500 | 529,868 | Lite-DPAS 약 48.9% 감소 |
| 1/32 | 64 | 139,300 | 267,660 | Lite-DPAS 약 48.0% 감소 |
| 1/64 | 32 | 73,700 | 136,556 | Lite-DPAS 약 46.0% 감소 |

Lite-DPAS V5 Conv2d/Linear MACs는 입력 1개 기준 약 3.983M, 3.197M, 3.066M, 3.000M이다(CR=1/4, 1/16, 1/32, 1/64 순). 이전 표에서 FLOPs는 MAC당 2 FLOPs로 환산해 각각 7.967M, 6.394M, 6.132M, 6.001M으로 제시했다. 이 집계는 FFT, 정규화, 활성화 함수와 일부 elementwise 연산을 포함하지 않는다. 따라서 다른 profiler의 값과 직접 동일시하지 않는다.

### 3.4 기준 모델과 복잡도 비교

아래 문헌 수치는 논문에서 FLOPs로 보고한 값이고, Lite-DPAS 및 DPAS-CsiNet V5 수치는 코드 구조에서 Conv2d/Linear MAC을 분석적으로 계산한 값이다. 문헌의 FLOPs 집계가 MAC을 1 operation으로 세는 관례와 가까워 나란히 제시했지만, 엄밀한 동일 profiler 비교는 아니다. Lite-DPAS/DPAS V5 수치는 FFT, BN, activation 및 기타 elementwise 연산을 제외한다. FLOPs를 표준적인 `2 × MACs`로 환산하면 Lite-DPAS는 7.967M, 6.394M, 6.132M, 6.001M이고 DPAS-CsiNet V5는 61.165M, 46.995M, 44.633M, 43.453M이다.

| 모델 | CR=1/4 | CR=1/16 | CR=1/32 | CR=1/64 | 출처/집계 방식 |
|---|---:|---:|---:|---:|---|
| CsiNet | 5.41M | 3.84M | 3.58M | 3.45M | 논문 보고 FLOPs |
| CRNet | 5.12M | 3.55M | 3.28M | 3.16M | 논문 보고 FLOPs |
| CsiNet+ | 24.57M | 23.00M | 22.74M | — | 논문 보고 FLOPs; 1/64 미보고 |
| DCRNet-1× | 4.01M | 2.44M | 2.18M | — | 논문 보고 FLOPs; 1/64 미보고 |
| TransNet | 35.72M | 34.14M | 33.88M | 33.75M | 공개 구현의 1000-epoch 결과표 FLOPs |
| DPAS-CsiNet V5 | 30.582M | 23.497M | 22.317M | 21.726M | 저장소 V5 코드로 Conv/Linear MAC 계산; FFT 제외 |
| **Lite-DPAS V5** | **3.983M** | **3.197M** | **3.066M** | **3.000M** | 프로젝트 V5 코드로 Conv/Linear MAC 계산; FFT 제외 |

| 모델 | CR=1/4 parameters | CR=1/16 parameters | CR=1/32 parameters | CR=1/64 parameters | 출처/집계 방식 |
|---|---:|---:|---:|---:|---|
| CsiNet | 2.103M | 0.530M | 0.268M | 0.137M | 프로젝트 CsiNet 포트 구조에서 산출 |
| CRNet | 2.103M | 0.530M | 0.268M | 0.137M | 프로젝트 CRNet 구현 구조에서 산출 |
| CsiNet+ | 2.122M | 0.549M | 0.286M | — | DCRNet 논문 비교표; 1/64 미보고 |
| DCRNet-1× | 2.102M | 0.528M | 0.266M | — | 원 논문 비교표; 1/64 미보고 |
| TransNet | 약 2.679M | 약 1.106M | 약 0.843M | 약 0.712M | 프로젝트 구현 설정으로 계산한 추정치; 반복 Transformer layer의 weight sharing 포함 |
| DPAS-CsiNet V5 | 약 9.470M | 약 2.385M | 약 1.204M | 약 0.614M | 저장소 V5 코드 구조로 산출; auxiliary branch 포함 |
| **Lite-DPAS V5** | **1.058M** | **0.271M** | **0.139M** | **0.074M** | 프로젝트 V5 코드에서 산출 |

CR=1/4에서 Lite-DPAS V5는 프로젝트 CsiNet 포트보다 parameters가 약 49.7% 적고, Conv/Linear MAC이 약 26.4% 적다. 저장소 DPAS-CsiNet V5 구현과 비교하면 parameters는 약 88.8%, Conv/Linear MAC은 약 87.0% 적게 산출된다. 이 수치는 코드 구조를 바탕으로 계산한 것이므로 실제 장치 latency나 FFT 전체 비용을 대신하지 않는다.

## 4. 문헌 성능 비교

아래는 문헌 및 공개 구현의 보고값이다. Lite-DPAS는 사용자가 공유한 결과만 넣었고, 미측정 압축률은 비워 두었다. **학습 epoch·데이터 분할·재현 코드가 다르므로 참고용 비교**다.

### Indoor NMSE (dB)

| 모델 | 1/4 | 1/16 | 1/32 | 1/64 |
|---|---:|---:|---:|---:|
| CsiNet | −17.36 | −8.65 | −6.24 | −5.84 |
| CRNet | −26.99 | −11.35 | −8.93 | 미보고 |
| CsiNet+ | −27.37 | −14.14 | −10.43 | 미보고 |
| DCRNet-1× | −28.04 | −11.74 | −9.05 | 미보고 |
| TransNet (1000 epoch) | −32.38 | −15.00 | −10.49 | −6.08 |
| DPAS-CsiNet (저장소 보고값) | −24.72 | −13.52 | −9.35 | −6.64 |
| **Lite-DPAS V5 (사용자 실험)** | **약 −19.4** | — | — | **약 −6.85** |

### Outdoor NMSE (dB)

| 모델 | 1/4 | 1/16 | 1/32 | 1/64 |
|---|---:|---:|---:|---:|
| CsiNet | −8.75 | −4.51 | −2.81 | −1.93 |
| CRNet | −12.70 | −5.44 | −3.51 | 미보고 |
| CsiNet+ | −12.40 | −5.73 | −3.40 | 미보고 |
| DCRNet-1× | −12.58 | −5.60 | −3.47 | 미보고 |
| TransNet (1000 epoch) | −14.86 | −7.82 | −4.13 | −2.62 |
| DPAS-CsiNet (저장소 보고값) | −15.15 | −6.56 | −4.37 | −2.49 |
| **Lite-DPAS V5 (사용자 실험/추정)** | **약 −10 (추정)** | — | — | **−2.301** |

DCRNet은 논문의 DCRNet-1× 설정이다. 해당 비교 논문은 CsiNet+, CRNet, DCRNet에 대해 CR=1/4, 1/8, 1/16, 1/32를 보고해 1/64 칸은 미보고로 표시했다. TransNet은 공개 구현의 1000 epoch 결과를 사용했다. DPAS 수치는 GitHub README의 COST2100 표다.

## 5. 현재 결과의 해석

1. **파라미터 감소는 명확한 장점**이다. CR=1/4에서 CsiNet 포트 대비 약 절반의 학습 파라미터를 사용한다.
2. **Indoor CR=1/4 성능은 개선 여지가 크다.** 공유된 약 −19.4 dB는 CsiNet 보고값 −17.36 dB보다 좋지만, CRNet·CsiNet+·DCRNet·TransNet·원본 DPAS 보고값보다 낮다.
3. **Indoor CR=1/64 약 −6.85 dB**는 표의 CsiNet·CRNet·TransNet·원본 DPAS 보고값과 비교해 경쟁력 있는 편이다. 다만 학습·평가 조건 통일 전에는 우위로 단정하지 않는다.
4. **Outdoor CR=1/4 약 −10 dB는 아직 가설**이다. 실제 결과를 받은 뒤 실측치로 교체한다.
5. Outdoor CR=1/64의 약 −2.301 dB는 약 400 epoch 시점 사용자 공유값이다.

## 6. 논문용 비교를 위해 남은 실험

- Outdoor CR=1/4 학습 완료 후 best validation 및 고정 test split NMSE를 기록한다.
- Lite-DPAS Outdoor −2.301 dB 실험의 seed, validation/test 여부를 run log에서 확인한다.
- Lite-DPAS 1/16·1/32를 실험해 압축률별 곡선을 완성한다.
- 비교 모델들을 동일한 COST2100 전처리, train/validation/test 분할, loss, epoch 또는 수렴 기준으로 재학습한다.
- 가능하면 seed를 3개 이상 바꿔 평균과 표준편차를 보고한다.
- 파라미터 수, Conv/Linear MACs, 전체 FFT 포함 FLOPs, batch-1 지연시간을 별도 정의로 측정하고 하드웨어·도구 버전을 함께 적는다.

## 참고 자료

1. DPAS-CsiNet 저장소 및 모델/성능 설명: https://github.com/OhayoRINK/DPAS-CsiNet
2. CsiNet 원 논문: https://arxiv.org/abs/1712.08919
3. CRNet 원 논문: https://arxiv.org/abs/1910.14322
4. DCRNet 원 논문: https://arxiv.org/abs/2106.04043
5. TransNet 공개 구현 및 결과: https://github.com/Treedy2020/TransNet
6. CsiNet+는 DCRNet 논문의 비교표에 인용된 결과를 사용함.
7. Lite-DPAS V5 로컬 구현: `src/models/lite_dpas.py` (프로젝트 기준)

---

**주의**: 본 문서는 사용자가 공유한 학습 로그·수치와 공개 논문/저장소의 보고값을 정리한 실험 노트다. Lite-DPAS의 일부 run metadata(CR, epoch, 평가 split)가 전달되지 않아 해당 칸에는 미확정 또는 추정으로 표기했다. 논문 성능 주장 전에 원시 로그와 test set 평가로 재검증해야 한다.


