# Waterbirds — Spurious Correlation 완화 실험

홍익대학교 컴퓨터공학과 **기계학습심화** 기말 대체 프로젝트

📓 **노트북: [waterbirds.ipynb](waterbirds.ipynb)**

## 문제

이미지 분류기가 새 자체가 아니라 **배경**을 보고 판단하는 문제(spurious correlation)를 다룬다.
Waterbirds 데이터에서는 물새가 대부분 물 배경에, 육지새가 대부분 땅 배경에 등장하므로
모델이 "물 배경이면 물새"라는 지름길을 배우기 쉽다.
평가는 (새 종류 × 배경) 4개 그룹 중 가장 낮은 정확도인 **worst-group 정확도**를 기준으로 한다.

## 접근

### 1. 논문 재현 — GroupDRO (E1~E4)

> Sagawa, S., Koh, P. W., Hashimoto, T. B., & Liang, P. (2020).
> *Distributionally Robust Neural Networks for Group Shifts: On the Importance of
> Regularization for Worst-Case Generalization.* ICLR 2020.
> [arXiv:1911.08731](https://arxiv.org/abs/1911.08731)

손실이 큰 그룹에 가중치를 몰아 최악 그룹을 직접 최적화하는 GroupDRO를 구현하고,
논문의 핵심 주장인 "GroupDRO는 강한 L2 정규화와 함께 써야 효과가 난다"를
ERM/GroupDRO × 약한/강한 L2의 2×2 실험으로 재현했다.

### 2. 확장 실험 — 배경 랜덤화 (E5~E6)

논문에는 없는, 직접 설계한 실험이다.
DeepLabV3로 새를 분리한 뒤 무작위 배경에 합성해 학습 데이터의 새-배경 상관 자체를 제거하고,
알고리즘을 고치는 방식(GroupDRO)과 비교했다.

## 결과 (ResNet-50, test set)

| 실험 | 손실 | L2 | 학습 데이터 | 평균 | **Worst** |
|---|---|---|---|---|---|
| E1 | ERM | 약함 | 원본 | 0.857 | 0.678 |
| E2 | GroupDRO | 약함 | 원본 | 0.923 | 0.854 |
| E3 | GroupDRO | 강함 | 원본 | 0.906 | 0.862 |
| E4 | ERM | 강함 | 원본 | 0.765 | 0.184 |
| E5 | ERM | 약함 | 배경 랜덤 | **0.950** | 0.891 |
| E6 | GroupDRO | 강함 | 배경 랜덤 | 0.915 | **0.897** |

- 데이터만 고친 E5가 그룹 라벨을 쓰는 GroupDRO(E3)보다 worst가 높았고, 평균 정확도도 함께 올랐다.
- **왜 좋아졌는지 검증**: 새를 지운 배경 이미지에서 예측이 배경을 따라가는 비율(배경추종률)이
  0.882(E1) → 0.548(E6)로 감소했다. 반면 feature에서 배경을 맞히는 정확도는 세 모델 모두 약 0.92로 같았다.
  즉 배경 정보가 사라진 게 아니라, 모델이 그 정보를 **판단에 덜 쓰게** 된 것이다.

## 한계

- 단일 시드 1회 실행이라 논문의 수치와 직접 비교하기보다는, 경향(강한 L2 + GroupDRO의 효과)을 재현한 것으로 본다.
- 배경 합성 경계가 별도의 단서로 작용했을 가능성을 배제하지 못했다.

## 실행 환경

Google Colab (Tesla T4), PyTorch, WILDS
