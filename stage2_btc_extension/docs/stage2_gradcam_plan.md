# Stage 2 BTC Grad-CAM Plan

## English

Status: planning complete for checklist 2-7. Implementation happens later in
`2-I8`.

Purpose:
- Generate BTC Grad-CAM figures for every trained BTC baseline run.
- Keep the same interpretation standard as Stage 1 and the Re-image Figure 13
  style.
- Make the heatmaps comparable across BTC image windows, image specifications,
  return horizons, and later Linear/FiLM stages.

Source basis:
- Root `PLAN.md`, Stage 2 Grad-CAM requirement.
- Re-image Figure 13: original image row followed by Grad-CAM rows for CNN
  layers; brighter heatmap regions indicate stronger activation/contribution.
  Stage 1 source-map records this as Re-image Figure 13 and local summary
  mapping pp.41-49.
- Grad-CAM original paper:
  `자료조사/Grad-CAM요약.md`, method pp.4-6.
  - Use the target class score before softmax.
  - Compute channel weights by spatially averaging gradients.
  - Compute `ReLU(sum_k alpha_k^c A^k)`.
  - Upsample the heatmap to the input image size with bilinear interpolation.
- Stage 1 implementation reference:
  `stage1_reimage_reproduction/src/stage1_reimage/interpretability/gradcam.py`.

Important distinction:
- The saved BTC Grad-CAM figure is not a raw feature map.
- It is a class-discriminative heatmap made from CNN activations and gradients
  with respect to a target class logit.
- A bright region means that region contributed more positively to the selected
  Up or Down class score. It does not prove causal market importance.

## Target Layers by BTC Model

The target layers follow the CNN variant selected by image window:

| BTC model | Input size | CNN blocks | Grad-CAM rows |
| --- | ---: | ---: | --- |
| `stock_cnn_i5` | `32 x 15` | 2 | `block1_conv`, `block2_conv` |
| `stock_cnn_i20` | `64 x 60` | 3 | `block1_conv`, `block2_conv`, `block3_conv` |
| `stock_cnn_i60` | `96 x 180` | 4 | `block1_conv`, `block2_conv`, `block3_conv`, `block4_conv` |

All heatmaps are upsampled to the input image size of the relevant model.

## Sample Selection

Default Figure-13-style selection:
- split: `test`
- per experiment tuple:
  `(image_window, image_spec, return_horizon, model_name, run_seed)`
- choose `10` predicted Up samples and `10` predicted Down samples for final
  report figures.
- choose by prediction confidence:
  - Up panel: highest `prob_up` among `pred_class == 1`.
  - Down panel: highest `prob_down` among `pred_class == 0`.
- quick smoke figures may use `2` per class.

Cross-model comparison selection:
- For comparing baseline, Linear, Gamma-only FiLM, and Full FiLM later, use a
  fixed date list so the same BTC chart dates can be compared across models.
- The fixed list should be saved as:
  `outputs/stage2/{experiment}/gradcam_selected_samples.csv`.

Multiple seeds:
- Per-seed Grad-CAM is computed from each seed checkpoint.
- For averaged prediction reports, select samples from averaged prediction CSV,
  then compute seed-level Grad-CAM heatmaps and average normalized heatmaps
  across available seed checkpoints.
- If only one seed exists, report it explicitly as `single_seed`.

## Output Artifacts

Required outputs:
- `outputs/stage2/{experiment}/figures/gradcam/{split}/figure13_style.png`
- `outputs/stage2/{experiment}/figures/gradcam/{split}/samples/*.png`
- `outputs/stage2/{experiment}/figures/gradcam/{split}/metadata.json`
- optional small report copy:
  `reports/figures/gradcam/stage2_{experiment}_{split}_figure13_style.png`

Metadata must include:
- image window, image spec, return horizon
- model name and run seed
- split
- BTC image start/end date
- label end date
- label, future return, predicted class, `prob_up`, `prob_down`
- target class used for Grad-CAM
- checkpoint path
- prediction CSV path
- target layer names
- heatmap normalization method

## Implementation Contract

Planned code locations:
- `src/stage2_btc/interpretability/gradcam.py`
- `scripts/generate_stage2_gradcam.py`

Implementation rules:
- Use the pre-softmax class logit as target score.
- Do not use softmax probability for backpropagating the heatmap.
- Register hooks only on convolution modules selected by the model variant.
- Remove hooks after each computation.
- Use train-only normalization metadata from the checkpoint/config.
- Save both the rendered input image and Grad-CAM heatmaps.
- Include Korean comments/docstrings explaining tensor shapes and flow.

Validation before full run:
- local or Kaggle smoke check with one Up and one Down sample.
- verify the figure is not blank.
- verify heatmap size equals input image size.
- verify metadata rows match prediction CSV rows.

## 한국어

상태: checklist 2-7 계획 완료. 실제 구현은 이후 `2-I8`에서 합니다.

목적:
- 학습된 모든 BTC baseline run에 대해 BTC Grad-CAM 그림을 생성합니다.
- Stage 1과 Re-image Figure 13의 해석 기준을 유지합니다.
- BTC image window, image specification, return horizon, 그리고 이후 Linear/FiLM
  단계까지 heatmap을 비교 가능하게 만듭니다.

근거:
- Root `PLAN.md`, Stage 2 Grad-CAM 필수 조건.
- Re-image Figure 13: original image row 다음에 CNN layer별 Grad-CAM row를 둡니다.
  밝은 영역은 더 강한 activation/contribution 영역으로 해석합니다. Stage 1
  source-map은 이를 Re-image Figure 13과 로컬 요약 pp.41-49로 기록했습니다.
- Grad-CAM 원전:
  `자료조사/Grad-CAM요약.md`, method pp.4-6.
  - softmax 이전 target class score를 사용합니다.
  - gradient를 spatial average해서 channel weight를 계산합니다.
  - `ReLU(sum_k alpha_k^c A^k)`로 heatmap을 계산합니다.
  - bilinear interpolation으로 input image size까지 upsample합니다.
- Stage 1 구현 참고:
  `stage1_reimage_reproduction/src/stage1_reimage/interpretability/gradcam.py`.

중요한 구분:
- BTC Grad-CAM figure는 raw feature map이 아닙니다.
- target class logit에 대한 CNN activation과 gradient로 만든
  class-discriminative heatmap입니다.
- 밝은 영역은 선택한 Up 또는 Down class score에 더 크게 양의 기여를 한 영역입니다.
  이것이 시장에서 인과적으로 중요하다는 뜻은 아닙니다.

## BTC Model별 Target Layer

Target layer는 image window로 선택되는 CNN variant를 따릅니다.

| BTC model | Input size | CNN blocks | Grad-CAM rows |
| --- | ---: | ---: | --- |
| `stock_cnn_i5` | `32 x 15` | 2 | `block1_conv`, `block2_conv` |
| `stock_cnn_i20` | `64 x 60` | 3 | `block1_conv`, `block2_conv`, `block3_conv` |
| `stock_cnn_i60` | `96 x 180` | 4 | `block1_conv`, `block2_conv`, `block3_conv`, `block4_conv` |

모든 heatmap은 해당 model의 input image size로 upsample합니다.

## Sample Selection

기본 Figure-13-style 선택:
- split: `test`
- experiment tuple:
  `(image_window, image_spec, return_horizon, model_name, run_seed)`
- 최종 보고 figure는 predicted Up `10`개, predicted Down `10`개를 고릅니다.
- prediction confidence 기준으로 고릅니다.
  - Up panel: `pred_class == 1` 중 `prob_up`이 가장 높은 sample.
  - Down panel: `pred_class == 0` 중 `prob_down`이 가장 높은 sample.
- 빠른 smoke figure는 class당 `2`개만 사용할 수 있습니다.

모델 간 비교용 선택:
- 이후 baseline, Linear, Gamma-only FiLM, Full FiLM을 비교할 때는 같은 BTC chart
  date를 놓고 heatmap을 비교할 수 있도록 fixed date list를 저장합니다.
- 저장 위치:
  `outputs/stage2/{experiment}/gradcam_selected_samples.csv`.

여러 seed:
- seed별 checkpoint에서 seed별 Grad-CAM을 계산합니다.
- averaged prediction report에서는 averaged prediction CSV에서 sample을 고르고,
  각 seed checkpoint로 Grad-CAM을 계산한 뒤 normalized heatmap을 평균합니다.
- seed가 하나뿐이면 `single_seed`라고 명시합니다.

## Output Artifacts

필수 output:
- `outputs/stage2/{experiment}/figures/gradcam/{split}/figure13_style.png`
- `outputs/stage2/{experiment}/figures/gradcam/{split}/samples/*.png`
- `outputs/stage2/{experiment}/figures/gradcam/{split}/metadata.json`
- 선택적 작은 report copy:
  `reports/figures/gradcam/stage2_{experiment}_{split}_figure13_style.png`

Metadata에는 반드시 다음을 포함합니다.
- image window, image spec, return horizon
- model name과 run seed
- split
- BTC image start/end date
- label end date
- label, future return, predicted class, `prob_up`, `prob_down`
- Grad-CAM target class
- checkpoint path
- prediction CSV path
- target layer names
- heatmap normalization method

## Implementation Contract

예정 코드 위치:
- `src/stage2_btc/interpretability/gradcam.py`
- `scripts/generate_stage2_gradcam.py`

구현 규칙:
- target score는 softmax 이전 class logit을 사용합니다.
- heatmap backpropagation에 softmax probability를 사용하지 않습니다.
- hook은 model variant에서 선택한 convolution module에만 등록합니다.
- 계산 후 hook을 제거합니다.
- checkpoint/config에 저장된 train-only normalization metadata를 사용합니다.
- rendered input image와 Grad-CAM heatmap을 모두 저장합니다.
- tensor shape와 흐름을 설명하는 한국어 주석/docstring을 남깁니다.

Full run 전 검증:
- local 또는 Kaggle smoke check에서 Up 1개, Down 1개를 먼저 생성합니다.
- figure가 blank가 아닌지 확인합니다.
- heatmap size가 input image size와 같은지 확인합니다.
- metadata row가 prediction CSV row와 일치하는지 확인합니다.
