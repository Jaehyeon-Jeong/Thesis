# FiLM Reference Review

## English

Reference implementation:
- Repository: `https://github.com/ethanjperez/film`
- Checked HEAD: `fe43ddf8a22b339dcca2efa07091ce9d498955cf`

Local summary points to carry into Stage 4:
- FiLM applies feature-wise affine modulation:
  `FiLM(F_c | gamma_c, beta_c) = gamma_c * F_c + beta_c`.
- For convolutional feature maps, gamma and beta are channel-wise and shared
  across spatial locations.
- The implementation notes use `gamma = 1 + delta_gamma` so the model starts
  near identity modulation.
- The important insertion choice for this project is after BatchNorm and before
  the nonlinearity.
- Gamma-only FiLM is a required ablation because gamma is central in the FiLM
  paper's analysis.

Stage 4 adaptation:
- The original FiLM project uses a language question encoder for visual
  reasoning.
- This thesis applies FiLM to BTC chart-image CNNs.
- Therefore, the core FiLM layer and gamma/beta generator idea are reused, but
  the condition source is separated into tracks:
  FiLM-only control, F&G numeric condition, News + non-LLM encoder, and
  News + LLM encoder.

## 한국어

Reference implementation:
- Repository: `https://github.com/ethanjperez/film`
- 확인한 HEAD: `fe43ddf8a22b339dcca2efa07091ce9d498955cf`

Stage 4에 가져갈 로컬 요약 핵심:
- FiLM은 feature-wise affine modulation입니다:
  `FiLM(F_c | gamma_c, beta_c) = gamma_c * F_c + beta_c`.
- Convolutional feature map에서는 gamma와 beta가 channel-wise이고 spatial 위치마다
  같은 값을 공유합니다.
- Implementation note에서는 `gamma = 1 + delta_gamma`를 사용해서 modulation이
  identity 근처에서 시작되도록 합니다.
- 이 프로젝트에서 중요한 삽입 위치는 BatchNorm 뒤, nonlinearity 전입니다.
- Gamma-only FiLM은 필수 ablation입니다. FiLM 논문 분석에서 gamma의 역할이
  중요하게 다뤄졌기 때문입니다.

Stage 4 adaptation:
- 원 FiLM project는 visual reasoning에서 language question encoder를 사용합니다.
- 이 논문은 BTC chart-image CNN에 FiLM을 적용합니다.
- 따라서 FiLM layer와 gamma/beta generator의 핵심 아이디어는 재사용하되,
  condition source는 FiLM-only control, F&G numeric condition,
  News + non-LLM encoder, News + LLM encoder로 분리합니다.
