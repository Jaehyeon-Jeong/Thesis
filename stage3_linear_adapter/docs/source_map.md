# Stage 3 Source Map

## English

This source map is initialized for Stage 3 and will be filled as each checklist
item confirms implementation details.

Known sources at scaffold time:

| Source | Role | Current use |
|:---|:---|:---|
| Root `PLAN.md` | Fixed research design | Defines Stage 3 as `BTC CNN baseline` vs `BTC CNN + Linear(bias=False)` |
| Stage 2 code/docs | Inherited BTC pipeline | Image generation, labels, split, metrics, and Grad-CAM must remain fixed |
| `lich99/Stock_CNN` | CNN core reference inherited through Stage 1/2 | CNN feature extractor must not be changed for Stage 3 |
| PyTorch `torch.nn.Linear` | Adapter implementation API | Linear adapter with `bias=False` |

Confirmed during planning:
- Stage 2 CNN feature dimensions are:
  - `I5`: `15,360`
  - `I20`: `46,080`
  - `I60`: `184,320`
- Naive `Linear(feature_dim, feature_dim)` is rejected as infeasible.
- First Linear comparison uses `adapter_dim=128`, `bias=False`.
- First comparison grid is `36` runs for seed `42`, matching the Stage 2
  single-seed grid.
- Full five-seed stability check is deferred.

Implementation docs:
- `docs/stage2_dependency_baseline_review.md`
- `docs/linear_adapter_design.md`
- `docs/training_evaluation_comparison_plan.md`
- `docs/gradcam_comparison_plan.md`
- `docs/kaggle_runner_output_plan.md`

## 한국어

이 source map은 Stage 3 시작용으로 초기화했습니다. 각 checklist 항목에서 구현 세부사항을
확인하면서 채워갑니다.

scaffold 시점의 근거:

| Source | 역할 | 현재 사용 |
|:---|:---|:---|
| Root `PLAN.md` | 고정 연구 설계 | Stage 3를 `BTC CNN baseline` vs `BTC CNN + Linear(bias=False)`로 정의 |
| Stage 2 code/docs | 상속받는 BTC pipeline | image generation, label, split, metric, Grad-CAM은 고정 |
| `lich99/Stock_CNN` | Stage 1/2를 통해 상속되는 CNN core reference | Stage 3에서 CNN feature extractor를 바꾸지 않음 |
| PyTorch `torch.nn.Linear` | Adapter 구현 API | `bias=False` Linear adapter |

계획 단계에서 확인한 것:
- Stage 2 CNN feature dimension:
  - `I5`: `15,360`
  - `I20`: `46,080`
  - `I60`: `184,320`
- 단순 `Linear(feature_dim, feature_dim)`는 계산상 불가능해서 제외합니다.
- 첫 Linear 비교는 `adapter_dim=128`, `bias=False`를 사용합니다.
- 첫 비교 grid는 Stage 2 single-seed grid와 같은 seed `42` 기준 `36`개 run입니다.
- Full five-seed 안정성 확인은 나중으로 미룹니다.

구현 문서:
- `docs/stage2_dependency_baseline_review.md`
- `docs/linear_adapter_design.md`
- `docs/training_evaluation_comparison_plan.md`
- `docs/gradcam_comparison_plan.md`
- `docs/kaggle_runner_output_plan.md`
