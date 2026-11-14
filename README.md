# X-ray 위해물품 분류: 영상 + 추정 밀도 다중 입력 모델 (학술지 논문)

융합보안논문지 제26권 제1호(2026.02) 실험에 사용한 Google Colab 노트북입니다.
실험 당시의 노트북을 2026년 11월에 정리해 올렸고, 코드와 저장된 출력은 실행 당시 그대로입니다.

- 이 브랜치(`poco3858-tech-journal`): 학술지 논문 실험
- 학위논문 실험: [`main` 브랜치](https://github.com/poco3858-tech/X-ray-Image/tree/main)

## 데이터

- AI-Hub "X-ray 다중 객체" 데이터 중 11개 클래스, 12,270장
- 데이터와 라벨 파일은 AI-Hub 이용 약관에 따라 이 저장소에 포함하지 않았습니다.
- 분할: 학습 70% · 검증 20% · 테스트 10%, 클래스 비율을 유지한 계층 분할 (`random_state=42`), 테스트 1,216장

## 추정 밀도

```
D_est = 1 / (그레이스케일 평균 강도 + 1e-5)
```

- 224×224로 줄인 영상에서 계산하고 Min-Max 정규화를 거칩니다.
- 별도 입력(`density_input`)으로 받아, 영상 특징(GlobalAveragePooling + BatchNormalization)과 결합(Concatenate)한 뒤 분류기로 보냅니다.
- 영상 밝기에서 계산한 값이며, 물리적 밀도를 직접 측정한 값은 아닙니다.

## 파일과 결과

하이퍼파라미터는 Dense 1 유닛 / Dropout 1 / Dense 2 유닛 / Dropout 2 / 학습률 순서입니다.

| 파일 | 논문 | 백본 | 입력 | 하이퍼파라미터 결정 | 하이퍼파라미터 | 저장된 테스트 결과 |
|---|---|---|---|---|---|---|
| `VGG16_fin(ENG_nonlabel).ipynb` | Table 3 | VGG16 | 영상 + 추정 밀도 | 고정값 | 256 / 0.3 / 384 / 0.3 / 3.94e-5 | 정확도 0.7870, 가중 F1 0.7891 |
| `VGG16_fin(ENG).ipynb` | Table 3 | VGG16 | 영상 + 추정 밀도 | Keras Tuner Hyperband | 1024 / 0.4 / 384 / 0.4 / 3.94e-5 | 정확도 0.7993, 가중 F1 0.8000 |
| `EfficientNetB0_fin(ENG_nonlabel).ipynb` | Table 4 | EfficientNetB0 | 영상만 | 고정값 | 256 / 0.3 / 384 / 0.3 / 2.14e-4 | 학습 출력 미저장 |
| `EfficientNetB0_fin(ENG).ipynb` | Table 4, Figure 4 | EfficientNetB0 | 영상 + 추정 밀도 | Keras Tuner Hyperband | 1024 / 0.6 / 512 / 0.3 / 2.88e-4 | 정확도 0.7977, 가중 F1 0.7956 |

- 파일 이름의 `nonlabel`은 JSON 라벨에 밀도·색상 메타데이터를 기록하는 단계를 건너뛴 버전입니다. 이 메타데이터는 모델 입력으로 쓰이지 않으며, 모델에 들어가는 밀도는 위의 추정 밀도입니다.
- Keras Tuner 탐색과 학습은 실행할 때마다 결과가 조금씩 달라질 수 있습니다.

## 실행 환경

- Google Colab TPU (v2-8), Colab 기본 런타임(Python 3.12)
- TensorFlow 2.x · Keras · Keras Tuner · scikit-learn · pandas · matplotlib
- 노트북의 경로(`/content/drive/MyDrive/...`)는 실행 당시의 Google Drive 경로입니다. 실행하려면 데이터 위치에 맞게 바꾸어야 합니다.

## 관련 논문

- 추정 밀도 기반 다중 입력 모델을 이용한 X-ray 위해물품 탐지, 융합보안논문지 제26권 제1호, 2026. DOI: 10.33778/kcsa.2026.26.1.091
