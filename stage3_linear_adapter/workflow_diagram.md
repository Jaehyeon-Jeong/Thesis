# Stage 3 Workflow Diagram

## English

```mermaid
flowchart TD
    A["Stage 2 BTC image pipeline<br/>fixed"] --> B["BTC image batch"]
    B --> C["Fixed Stock_CNN-style feature extractor"]
    C --> D["Flattened CNN feature vector"]
    D --> E["Linear adapter<br/>bias=False"]
    E --> F["Classification logits"]
    F --> G["Prediction CSV"]
    G --> H["Classification metrics"]
    G --> I["Trading metrics<br/>long/flat and long/short"]
    F --> J["Grad-CAM on CNN blocks"]
    H --> K["Baseline vs Linear report"]
    I --> K
    J --> K

    L["Stage 2 BTC CNN baseline outputs"] --> K
```

Stage 3 changes only the model head/adaptation path. The BTC image, label,
split, normalization, and metric definitions remain inherited from Stage 2.

## 한국어

```mermaid
flowchart TD
    A["Stage 2 BTC image pipeline<br/>고정"] --> B["BTC image batch"]
    B --> C["고정된 Stock_CNN식 feature extractor"]
    C --> D["flattened CNN feature vector"]
    D --> E["Linear adapter<br/>bias=False"]
    E --> F["classification logits"]
    F --> G["prediction CSV"]
    G --> H["classification metric"]
    G --> I["trading metric<br/>long/flat, long/short"]
    F --> J["CNN block 기준 Grad-CAM"]
    H --> K["Baseline vs Linear report"]
    I --> K
    J --> K

    L["Stage 2 BTC CNN baseline output"] --> K
```

Stage 3에서 바꾸는 것은 model head/adaptation path뿐입니다. BTC image, label,
split, normalization, metric 정의는 Stage 2에서 그대로 가져옵니다.
