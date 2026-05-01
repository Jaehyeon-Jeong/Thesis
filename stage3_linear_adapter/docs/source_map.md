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

Items to confirm before implementation:
- Exact CNN feature tensor shape for `I5`, `I20`, `I60`.
- Whether the adapter maps `feature_dim -> feature_dim` or another documented
  adapter dimension.
- Whether the final classifier is reused after the adapter or rebuilt as part
  of a new model head.
- Which Stage 2 configurations are used for first Linear comparison.

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

구현 전 확인할 것:
- `I5`, `I20`, `I60`별 CNN feature tensor shape.
- adapter가 `feature_dim -> feature_dim`인지, 다른 adapter dimension을 둘 것인지.
- adapter 뒤 final classifier를 재사용할지, 새 model head로 구성할지.
- 첫 Linear 비교에 사용할 Stage 2 configuration 범위.
