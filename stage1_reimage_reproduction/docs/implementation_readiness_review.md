# Stage 1 Implementation Readiness Review

## English

Status:
- Implementation gate `1-I0` completed on 2026-04-30.
- No production code was implemented in this gate.

Decision:
- Ready to start `1-I1. Shared code/config scaffold implementation`.
- Readiness is for the public `monthly_20d` I20 full-spec reproduction path:
  `I20/R5`, `I20/R20`, and `I20/R60`.

## Checks Completed

| Check | Result |
| --- | --- |
| Root `PLAN.md` reviewed | Passed. Stage order and one-step workflow remain unchanged. |
| Stage 1 planning gates | Passed. `1-0` through `1-9` are complete. |
| Stage 1 planning docs | Passed. Data, labels, split, model, training, Kaggle, evaluation, Grad-CAM, and report plans exist. |
| Local data path | Passed. Local `monthly_20d` path is accessible. |
| Local data file counts | Passed. 27 `.dat` files and 27 `.feather` files found. |
| Local Python dependencies | Passed for local smoke-test needs: `torch`, `numpy`, `pandas`, `pyarrow`, `sklearn`, `matplotlib`, and `yaml` are importable. |
| Environment configs | Passed. `configs/env_local.yaml` and `configs/env_kaggle.yaml` exist. |
| Current code state | Clean for implementation. `src/`, `scripts/`, and `notebooks/` contain README placeholders only. |

## Scope Confirmed for First Implementation Pass

Implement only Stage 1 public-data reproduction:
- Load author/public rendered I20 full-spec `.dat` images.
- Pair them with `.feather` labels.
- Build `I20/R5`, `I20/R20`, and `I20/R60` datasets.
- Train/evaluate the GitHub-style `StockCNNI20` baseline.
- Produce predictions, metrics, stock ranking/portfolio outputs, and Grad-CAM
  figures according to the completed plans.

Do not implement in this Stage 1 pass:
- BTC data.
- Linear adapter.
- FiLM.
- News/LLM conditioning.
- Claims of full paper-wide reproduction.

## Important Caveats Before Coding

1. Root `PLAN.md` still lists OHLC image generator, MA line, and volume bars as
   broad Stage 1 implementation items. The immediate public-data implementation
   uses already-rendered I20 `.dat` images. Therefore, the first implementation
   pass is a public I20 reproduction, not a raw-CRSP image-generation pipeline.
   If raw OHLCV/CRSP data becomes available or the user asks for generator work,
   add a separate generator implementation gate before calling the paper-wide
   pipeline complete.
2. The checked GitHub I20 model applies `dilation=(2, 1)` to all three
   convolution layers, while the local Re-image summary emphasizes first-layer
   dilation. Stage 1 follows the GitHub model core and documents the mismatch.
3. The current workspace is not a Git repository. This does not block local
   implementation, but before Kaggle full runs the project should use either a
   code snapshot or a Git commit so the run manifest can record code provenance.
4. Kaggle full runs require uploading/attaching `monthly_20d` as a Kaggle
   Dataset, recommended name `reimage-monthly-20d`.
5. PDF page numbers for final citation-ready report text and model source
   comments still require visual PDF checks, because local automatic PDF text
   extraction is unavailable.
6. `I20/R5` paper classification accuracy was not extracted in the local summary.
   Do not invent this comparison value; check the PDF before the final report.
7. Stock H-L portfolio comparisons require matching decile, value-weighting, and
   annualization conventions before comparing to paper values.

## Implementation Guardrails

The next gates must preserve these rules:
- Use one shared codebase for local and Kaggle; path/runtime differences come
  from config.
- Local runs are smoke tests only unless explicitly marked otherwise.
- Kaggle `full_paper_style` is the primary reproduction mode.
- Model inputs are image tensors only; returns and metadata are labels or
  evaluation metadata, never input features.
- Keep tensor convention `(batch_size, 1, height=64, width=60)`.
- Use `CrossEntropyLoss` on logits; compute softmax only for prediction output.
- Preserve `Date`, `StockID`, `MarketCap`, target returns, labels, probabilities,
  predictions, and correctness in output files.
- Keep Grad-CAM target layers hookable: `layer1[0]`, `layer2[0]`, `layer3[0]`.
- Do not report smoke-test metrics as reproduction results.

## Next Gate

Proceed to:
- `1-I1. Shared code/config scaffold implementation`

Expected first implementation scope:
- Create importable `src/` package structure.
- Add config loading.
- Add output directory helpers.
- Add seed/reproducibility helpers.
- Add a minimal local import smoke check.

## 한국어

상태:
- 2026-04-30에 implementation gate `1-I0`를 완료했습니다.
- 이 gate에서는 production code를 구현하지 않았습니다.

판정:
- `1-I1. 공통 code/config scaffold 구현`으로 넘어갈 준비가 됐습니다.
- 이 readiness는 public `monthly_20d` I20 full-spec reproduction 경로,
  즉 `I20/R5`, `I20/R20`, `I20/R60`에 대한 것입니다.

## 완료한 확인

| 확인 항목 | 결과 |
| --- | --- |
| 루트 `PLAN.md` 확인 | 통과. 단계 순서와 한 단계씩 진행 원칙 유지. |
| Stage 1 planning gate | 통과. `1-0`부터 `1-9`까지 완료. |
| Stage 1 planning docs | 통과. data, label, split, model, training, Kaggle, evaluation, Grad-CAM, report plan 존재. |
| Local data path | 통과. local `monthly_20d` 경로 접근 가능. |
| Local data file count | 통과. `.dat` 27개와 `.feather` 27개 확인. |
| Local Python dependencies | 통과. local smoke-test에 필요한 `torch`, `numpy`, `pandas`, `pyarrow`, `sklearn`, `matplotlib`, `yaml` import 가능. |
| Environment config | 통과. `configs/env_local.yaml`, `configs/env_kaggle.yaml` 존재. |
| 현재 code 상태 | 구현 시작에 적합. `src/`, `scripts/`, `notebooks/`는 README placeholder만 있음. |

## 첫 구현 범위 확정

Stage 1 public-data reproduction만 구현합니다.
- 저자/public rendered I20 full-spec `.dat` image를 읽습니다.
- `.feather` label과 row-wise로 연결합니다.
- `I20/R5`, `I20/R20`, `I20/R60` dataset을 만듭니다.
- GitHub식 `StockCNNI20` baseline을 학습/평가합니다.
- 완료된 계획에 따라 prediction, metric, stock ranking/portfolio output,
  Grad-CAM figure를 만듭니다.

이번 1단계 구현에서 하지 않는 것:
- BTC data.
- Linear adapter.
- FiLM.
- News/LLM conditioning.
- full paper-wide reproduction이라고 주장하는 것.

## 코딩 전 중요 제한사항

1. 루트 `PLAN.md`에는 OHLC image generator, MA line, volume bars가 큰 1단계
   구현 항목으로 남아 있습니다. 하지만 즉시 구현할 public-data 경로는 이미
   렌더링된 I20 `.dat` image를 사용합니다. 따라서 첫 구현은 raw-CRSP
   image-generation pipeline이 아니라 public I20 reproduction입니다.
   raw OHLCV/CRSP data가 생기거나 사용자가 generator 작업을 요청하면,
   paper-wide pipeline 완료 전 별도 generator 구현 gate를 추가해야 합니다.
2. 확인한 GitHub I20 model은 세 convolution layer 모두에 `dilation=(2, 1)`을
   적용하지만, 로컬 Re-image 요약은 first-layer dilation을 강조합니다.
   1단계는 GitHub model core를 따르고 이 mismatch를 문서화합니다.
3. 현재 workspace는 Git repository가 아닙니다. local 구현은 막지 않지만,
   Kaggle full run 전에는 code snapshot 또는 Git commit을 사용해 run
   manifest에 code provenance를 기록해야 합니다.
4. Kaggle full run을 하려면 `monthly_20d`를 Kaggle Dataset으로 upload/attach
   해야 합니다. 권장 이름은 `reimage-monthly-20d`입니다.
5. 최종 보고서 citation과 model source comment에 들어갈 PDF page number는
   PDF를 눈으로 다시 확인해야 합니다. 현재 로컬 자동 PDF text extraction은
   사용할 수 없습니다.
6. `I20/R5` paper classification accuracy는 로컬 요약에서 추출되지 않았습니다.
   최종 보고 전 PDF에서 확인하고, 값을 임의로 만들지 않습니다.
7. Stock H-L portfolio 비교는 decile, value-weighting, annualization convention을
   맞춘 뒤 paper value와 비교합니다.

## 구현 Guardrail

다음 gate들은 아래 규칙을 유지해야 합니다.
- local과 Kaggle은 하나의 shared codebase를 사용하고, path/runtime 차이는 config로 처리합니다.
- local run은 명시적으로 다르게 표시하지 않는 한 smoke test입니다.
- Kaggle `full_paper_style`이 primary reproduction mode입니다.
- model input은 image tensor뿐입니다. return과 metadata는 label 또는 evaluation metadata이지 input feature가 아닙니다.
- tensor convention은 `(batch_size, 1, height=64, width=60)`입니다.
- logits에 `CrossEntropyLoss`를 쓰고, softmax는 prediction output에서만 계산합니다.
- output file에는 `Date`, `StockID`, `MarketCap`, target returns, labels,
  probabilities, predictions, correctness를 보존합니다.
- Grad-CAM target layer는 `layer1[0]`, `layer2[0]`, `layer3[0]`로 hook 가능해야 합니다.
- smoke-test metric을 reproduction result로 보고하지 않습니다.

## 다음 Gate

다음으로 진행할 항목:
- `1-I1. 공통 code/config scaffold 구현`

첫 구현 예상 범위:
- import 가능한 `src/` package structure 생성.
- config loading 추가.
- output directory helper 추가.
- seed/reproducibility helper 추가.
- 최소 local import smoke check 추가.
