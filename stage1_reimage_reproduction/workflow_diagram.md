# Stage 1 Workflow Diagram

## English

Canonical detailed map:
- [docs/stage1_execution_map.md](docs/stage1_execution_map.md)

```mermaid
flowchart TD
    A["monthly_20d .dat / .feather"] --> B["Data loading"]
    B --> C["Ret_5d / Ret_20d / Ret_60d > 0 -> Up=1"]
    C --> D["Time split: train / validation / test"]
    D --> E["Train-only pixel mean/std"]
    E --> F["StockCNNI20 training"]
    F --> G["Test prediction CSV"]
    G --> H["Metrics JSON"]
    G --> I["Grad-CAM figures"]
    H --> J["Output check"]
    I --> J
```

## 한국어

상세 기준 문서:
- [docs/stage1_execution_map.md](docs/stage1_execution_map.md)

```mermaid
flowchart TD
    A["monthly_20d .dat / .feather"] --> B["data loading"]
    B --> C["Ret_5d / Ret_20d / Ret_60d > 0 -> Up=1"]
    C --> D["time split: train / validation / test"]
    D --> E["train data로만 pixel mean/std 계산"]
    E --> F["StockCNNI20 학습"]
    F --> G["test prediction CSV 저장"]
    G --> H["metrics 저장"]
    G --> I["Grad-CAM 그림 저장"]
    H --> J["output check"]
    I --> J
```
