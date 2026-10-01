# 평가 규약

## 1차 실험 설정

- Dataset: COST2100 Indoor
- Representation: 32×32 angular-delay CSI
- Compression ratio: CR=1/4
- First backbone: CsiNet
- Primary metric: sample-wise complex NMSE in dB
- System metric: end-to-end latency including policy, data conversion, checks, and reconstruction

## 입력 표현

CsiNet 저자 구현은 `HT` `[N, 2048]`을 `[N, 2, 32, 32]`로 reshape합니다. 두 채널은 real/imag이며 CSI complex array를 만들 때 각 채널에서 `0.5`를 뺍니다. 저자 코드 링크: [CsiNet_train.py](https://github.com/sydney222/Python_CsiNet/blob/master/CsiNet_train.py).

```text
H_i = (X_i[0] - 0.5) + j (X_i[1] - 0.5)
```

CRNet/TransNet의 원 코드가 준비되면 각각의 loader와 정규화가 이 계약과 같은지 확인한 뒤 비교합니다. 모델에 맞춰 임의 전처리를 적용하지 않습니다.

## NMSE aggregation

```text
NMSE_i = ||H_i - Hhat_i||_F² / ||H_i||_F²
NMSE_dB = 10 log10(mean_i(NMSE_i))
```

Numerical protection은 denominator가 0인 경우에만 적용합니다. 모든 방법에 동일 구현을 적용하고 per-sample NMSE도 저장합니다.

## Data split 사용

- Train: 복원 모델 및 정책 학습
- Validation: checkpoint, threshold, 정책 및 하이퍼파라미터 선택
- Test: 선택이 끝난 뒤 최종 평가
- Indoor/Outdoor과 압축률을 분리해 결과를 보고
- 같은 official benchmark 파일을 사용해 비교 가능성을 유지

## 비용 측정

- GPU, driver/runtime, precision, software versions
- model parameter count, peak GPU memory, FLOPs/MACs와 계산 도구
- batch size 1 latency가 주 지표; 고정 batch 조건도 추가 보고
- warm-up 후 다회 반복, median과 p95 사용
- policy 출력 검증/timeout/fallback 포함 end-to-end 측정
- 평가용 배치 처리 시간만으로 실시간 성능을 주장하지 않음

## 결과 유형

저자 설정 재현과 공정 통제 비교를 별도 표로 기록합니다. 저자 저장소 README의 숫자는 우리 실행 결과로 옮기지 않습니다.
