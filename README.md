# X-ray 위해물품 분류: 영상 + 추정 밀도 다중 입력 모델 (학위논문)

석사학위논문(2025.02) 실험에 사용한 Google Colab 노트북입니다.
실험 당시의 노트북을 2026년 11월에 정리해 올렸고, 코드와 저장된 출력은 실행 당시 그대로입니다.

- 이 브랜치(`main`): 학위논문 실험
- 학술지 논문 실험: [`poco3858-tech-journal` 브랜치](https://github.com/poco3858-tech/X-ray-Image/tree/poco3858-tech-journal)

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
| `EfficientNetB0.ipynb` | 학위논문 표 4-6 | EfficientNetB0 | 영상 + 추정 밀도 | Keras Tuner Hyperband | 1024 / 0.6 / 384 / 0.3 / 2.85e-4 | 정확도 0.7812, 가중 F1 0.7810 |
| `VGG16.ipynb` | 학위논문 | VGG16 | 영상 + 추정 밀도 | Keras Tuner Hyperband | 1024 / 0.3 / 384 / 0.4 / 1.64e-5 | 정확도 0.7245, 가중 F1 0.7327 (아래 참고) |

- `VGG16.ipynb`의 저장된 출력은 학위논문 작성 이후 다시 실행한 결과입니다. 학위논문 표 4-5의 수치(정확도 0.8043)는 이전 실행의 결과입니다.
- Keras Tuner 탐색과 학습은 실행할 때마다 결과가 조금씩 달라질 수 있습니다.

## 실행 환경

- Google Colab TPU (v2-8), Python 3.10
- TensorFlow 2.x · Keras · Keras Tuner · scikit-learn · pandas · matplotlib
- 노트북의 경로(`/content/drive/MyDrive/...`)는 실행 당시의 Google Drive 경로입니다. 실행하려면 데이터 위치에 맞게 바꾸어야 합니다.

## 관련 논문

- 기계학습 기반 기반시설 위해물품 탐지개선에 관한 연구, 석사학위논문, 2025.
