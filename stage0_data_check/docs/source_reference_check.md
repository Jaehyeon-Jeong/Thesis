# Source Reference Check

## English

Status: completed for Stage 0-4.

This file records the Stage 1 source references checked before implementation.
No model, training, or data-generation code was implemented in this step.

Scope:
- Stage 1 only: Re-image paper pipeline and the `lich99/Stock_CNN` reference model.
- FiLM paper/repository references are intentionally deferred until the FiLM stage source check.

Checked local sources:
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`

PDF access note:
- The PDF exists locally.
- Direct PDF text extraction tools were not available in the current environment
  (`pypdf`, `PyPDF2`, `fitz`, `pdfplumber`, `pdftotext`, and `mutool` were not available).
- Therefore, page references below come from the local Re-image summary file that already maps
  the paper pages/figures/tables. Before writing Stage 1 code comments, the same page numbers
  should be visually rechecked in the PDF viewer.

GitHub reference:
- Repository: `https://github.com/lich99/Stock_CNN`
- Checked HEAD commit: `415e2acf2a5013afca67e383acd3edc61fced840`
- Reference file checked: `models/baseline.py`
- Local copy saved at: `references/stock_cnn_baseline_415e2ac.py.txt`

Paper source map for Stage 1:

| Topic | Paper / local summary reference | Implementation implication |
| --- | --- | --- |
| Input data | Re-image summary: CRSP daily individual stocks, NYSE/AMEX/NASDAQ, 1993-2019, pages 21-22 | Stage 1 stock reproduction uses stock-level images and stock-level future returns, not portfolio-level labels. |
| OHLC image construction | Re-image summary: image construction pages 8-12; Figure 7 page 18; Table 1 | Black background, white OHLC objects, one trading day = 3 horizontal pixels, per-image vertical rescaling. |
| MA and volume | Re-image summary: image details around Table 1 / Figure 7 | MA window matches image length; volume bars are drawn in the lower image area and scaled by the image-local max volume. |
| Image sizes | Re-image summary: Table 1 and Figure 7 | Use 5-day `32 x 15`, 20-day `64 x 60`, 60-day `96 x 180` as model input shapes, with explicit height/width comments. |
| CNN architecture | Re-image summary: Figure 7 page 18; architecture pages 12-21 | Blocks are Conv -> BN -> LeakyReLU -> MaxPool; filters `5 x 3`; pool `2 x 1`; channels increase `64 -> 128 -> 256 -> 512`. |
| Training setup | Re-image summary: pages 20-22 | Cross-entropy, Adam, learning rate `1e-5`, batch size `128`, dropout `0.5`, early stopping after 2 non-improving validation epochs. |
| Split | Re-image summary: pages 20-22 | Train/validation period 1993-2000; test period 2001-2019; train/validation split is within the 1993-2000 period. |
| Interpretation | Re-image summary: interpretation pages 41-49; user-provided Figure 13 image | Grad-CAM must be treated as a class-discriminative heatmap, not as the raw feature map itself. |

`Stock_CNN` model details confirmed from `models/baseline.py`:

| Component | Reference implementation |
| --- | --- |
| Input reshape | `x.reshape(-1, 1, 64, 60)` |
| Layer 1 | `Conv2d(1, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, BN, LeakyReLU(0.01), MaxPool `(2,1)` |
| Layer 2 | `Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, BN, LeakyReLU(0.01), MaxPool `(2,1)` |
| Layer 3 | `Conv2d(128, 256, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, BN, LeakyReLU(0.01), MaxPool `(2,1)` |
| Classifier | Dropout `p=0.5`, Linear `46080 -> 2` |
| Forward output | Returns logits. `self.softmax` exists but is commented out in `forward`. |

Important mismatch / decision log before Stage 1:

1. Dilation placement:
   - Local paper summary says the paper emphasizes vertical dilation in the first layer,
     with 20-day dilation 2 and 60-day dilation 3.
   - `Stock_CNN/models/baseline.py` applies `dilation=(2,1)` and `padding=(12,1)` to all three I20 convolution layers.
   - Stage 1 implementation should record this mismatch in code comments. Based on the user's latest instruction,
     the I20 model core should follow the GitHub implementation unless the user decides otherwise.

2. Softmax / loss:
   - The paper describes softmax probabilities and a 50% decision threshold.
   - The GitHub implementation returns logits and comments out softmax.
   - Stage 1 should train with `CrossEntropyLoss` on logits, then apply softmax only for evaluation/prediction CSVs.

3. Image dimension ordering:
   - The summary describes Table 1 and Figure 7 with both `60 x 64` and `64 x 60` language depending on width/height convention.
   - The local `.dat` audit confirmed actual I20 tensors reshape to `(N, 64, 60)`.
   - Stage 1 comments must explicitly state tensor convention: `(batch, channel, height=64, width=60)`.

4. Grad-CAM:
   - The user-provided original figure shows Figure 13: `I20R20 Grad-CAM for 20 Images from 2019`.
   - The figure note states that brighter heatmap regions correspond to higher activation, and each panel shows original images followed by Grad-CAM for each CNN layer.
   - Stage 1 must reproduce this style for I20/R20: 10 Up predictions and 10 Down predictions from 2019 when possible.

5. Public data limitation:
   - Current public stock image data is full I20 with MA and volume only.
   - Therefore Stage 1 can directly check I20/R5, I20/R20, and I20/R60 on the provided rendered images.
   - I5, I60, and A/B/C/D image ablations require additional image shards or raw OHLCV regeneration.

Stage 1 code-comment requirement:

When Stage 1 implementation starts, every major module should include compact source comments such as:

```python
# Reference implementation:
#   lich99/Stock_CNN/models/baseline.py
#   commit: 415e2acf2a5013afca67e383acd3edc61fced840
#
# Paper source:
#   Jiang, Kelly, and Xiu, Re-Imagining Price Trends,
#   Figure 7, p.18; training details around pp.20-22.
#
# Tensor convention:
#   images: (batch_size, 1, height=64, width=60)
```

## 한국어

상태: 0-4 완료.

이 파일은 1단계 구현 전에 확인한 근거 자료를 기록합니다.
이번 단계에서는 모델, 학습, 이미지 생성 코드는 구현하지 않았습니다.

범위:
- 1단계만 확인했습니다: Re-image 논문 파이프라인과 `lich99/Stock_CNN` 기준 모델.
- FiLM 논문/깃허브 근거 확인은 4단계 들어갈 때 같은 방식으로 별도 진행합니다.

확인한 로컬 자료:
- `../자료조사/Re-image 요약.md`
- `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`

PDF 접근 메모:
- PDF 파일은 로컬에 존재합니다.
- 현재 환경에는 PDF 텍스트 추출 도구가 없었습니다
  (`pypdf`, `PyPDF2`, `fitz`, `pdfplumber`, `pdftotext`, `mutool` 없음).
- 따라서 아래 page reference는 로컬 Re-image 요약 파일이 정리한 page/figure/table mapping을 기준으로 적었습니다.
  Stage 1 코드 주석을 쓰기 전에는 PDF viewer에서 같은 page를 눈으로 다시 확인해야 합니다.

GitHub 기준:
- Repository: `https://github.com/lich99/Stock_CNN`
- 확인한 HEAD commit: `415e2acf2a5013afca67e383acd3edc61fced840`
- 확인한 파일: `models/baseline.py`
- 로컬 저장본: `references/stock_cnn_baseline_415e2ac.py.txt`

1단계 논문 근거 map:

| 항목 | 논문 / 로컬 요약 근거 | 구현상 의미 |
| --- | --- | --- |
| 입력 데이터 | Re-image 요약: CRSP daily individual stocks, NYSE/AMEX/NASDAQ, 1993-2019, 21-22쪽 | 1단계 재현은 stock-level image와 stock-level future return을 사용합니다. portfolio return 자체를 label로 쓰는 것이 아닙니다. |
| OHLC 이미지 생성 | Re-image 요약: 이미지 생성 8-12쪽; Figure 7 18쪽; Table 1 | 검정 배경, 흰색 OHLC 객체, 하루 가로 3픽셀, 이미지별 vertical rescaling을 사용합니다. |
| MA와 volume | Re-image 요약: Table 1 / Figure 7 주변 | MA window는 image length와 같고, volume bar는 이미지 하단에 이미지 내부 최대 volume 기준으로 scaling합니다. |
| 이미지 크기 | Re-image 요약: Table 1과 Figure 7 | 모델 입력은 5-day `32 x 15`, 20-day `64 x 60`, 60-day `96 x 180`로 관리하되 height/width convention을 코드 주석에 명시해야 합니다. |
| CNN 구조 | Re-image 요약: Figure 7 18쪽; architecture 12-21쪽 | block은 Conv -> BN -> LeakyReLU -> MaxPool, filter `5 x 3`, pool `2 x 1`, channel은 `64 -> 128 -> 256 -> 512`입니다. |
| 학습 설정 | Re-image 요약: 20-22쪽 | Cross-entropy, Adam, learning rate `1e-5`, batch size `128`, dropout `0.5`, validation loss 2 epoch 미개선 시 early stopping입니다. |
| split | Re-image 요약: 20-22쪽 | 1993-2000 train/validation, 2001-2019 test입니다. train/validation split은 1993-2000 내부에서 합니다. |
| 해석 | Re-image 요약: interpretation 41-49쪽; 사용자가 제공한 Figure 13 이미지 | Grad-CAM은 raw feature map이 아니라 class-discriminative heatmap으로 명시해야 합니다. |

`Stock_CNN`의 `models/baseline.py`에서 확인한 모델 디테일:

| 구성요소 | 기준 구현 |
| --- | --- |
| 입력 reshape | `x.reshape(-1, 1, 64, 60)` |
| Layer 1 | `Conv2d(1, 64, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, BN, LeakyReLU(0.01), MaxPool `(2,1)` |
| Layer 2 | `Conv2d(64, 128, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, BN, LeakyReLU(0.01), MaxPool `(2,1)` |
| Layer 3 | `Conv2d(128, 256, kernel_size=(5,3), stride=(3,1), dilation=(2,1), padding=(12,1))`, BN, LeakyReLU(0.01), MaxPool `(2,1)` |
| Classifier | Dropout `p=0.5`, Linear `46080 -> 2` |
| Forward output | logits를 반환합니다. `self.softmax`는 정의되어 있지만 `forward`에서는 주석 처리되어 있습니다. |

1단계 전에 남겨야 할 mismatch / 결정 사항:

1. Dilation 위치:
   - 로컬 논문 요약은 첫 번째 layer의 vertical dilation을 강조하며, 20-day dilation 2, 60-day dilation 3이라고 정리합니다.
   - 반면 `Stock_CNN/models/baseline.py`는 I20의 세 convolution layer 모두에 `dilation=(2,1)`, `padding=(12,1)`을 적용합니다.
   - 사용자 최신 지시에 따르면 모델 핵심은 GitHub 구현을 최대한 따라야 하므로, I20 구현은 GitHub를 우선하되 이 차이를 코드 주석과 문서에 남겨야 합니다.

2. Softmax / loss:
   - 논문은 softmax 상승확률과 50% threshold를 설명합니다.
   - GitHub 구현은 softmax를 `forward`에서 적용하지 않고 logits를 반환합니다.
   - 1단계에서는 `CrossEntropyLoss`에 logits를 넣고, 평가와 prediction CSV 저장 시에만 softmax probability를 계산하는 쪽이 GitHub와 PyTorch 관례에 맞습니다.

3. 이미지 차원 표기:
   - 요약에서는 Table 1과 Figure 7 때문에 `60 x 64`와 `64 x 60` 표현이 섞여 나올 수 있습니다.
   - 로컬 `.dat` 확인 결과 I20은 실제로 `(N, 64, 60)`으로 reshape됩니다.
   - 코드 주석에는 `(batch, channel, height=64, width=60)`로 convention을 고정해야 합니다.

4. Grad-CAM:
   - 사용자가 제공한 원문 Figure 13은 `I20R20 Grad-CAM for 20 Images from 2019`입니다.
   - figure note 기준으로 밝은 heatmap 영역은 higher activation이며, 각 panel은 원본 이미지 다음에 CNN 각 layer의 Grad-CAM을 보여줍니다.
   - 1단계에서는 가능하면 2019년 I20/R20에서 Up 예측 10개와 Down 예측 10개를 골라 같은 형식의 grid를 저장해야 합니다.

5. 공개 데이터 제한:
   - 현재 가지고 있는 공개 stock image data는 MA와 volume이 포함된 full I20 이미지입니다.
   - 따라서 현재 데이터로 직접 확인 가능한 것은 I20/R5, I20/R20, I20/R60입니다.
   - I5, I60, A/B/C/D image ablation은 추가 image shard 또는 raw OHLCV 기반 재생성이 필요합니다.

1단계 코드 주석 원칙:

Stage 1 구현을 시작하면 주요 모듈마다 아래처럼 짧은 source comment를 남깁니다.

```python
# Reference implementation:
#   lich99/Stock_CNN/models/baseline.py
#   commit: 415e2acf2a5013afca67e383acd3edc61fced840
#
# Paper source:
#   Jiang, Kelly, and Xiu, Re-Imagining Price Trends,
#   Figure 7, p.18; training details around pp.20-22.
#
# Tensor convention:
#   images: (batch_size, 1, height=64, width=60)
```
