# Stage 3 Workflow Diagram

## English

```mermaid
flowchart TD
    A["Stage 2 BTC image pipeline<br/>fixed"] --> B["BTC image batch"]
    B --> C["Fixed Stock_CNN-style feature extractor"]
    C --> D["Flattened CNN feature vector"]
    D --> E["Linear adapter/head<br/>adapter_dim=128<br/>bias=False"]
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
The first planned run is `I60/R20/ohlc_ma_vb`, seed `42`, followed by the
single-seed `36`-run grid.

## 한국어

```mermaid
flowchart TD
    A["Stage 2 BTC image pipeline<br/>고정"] --> B["BTC image batch"]
    B --> C["고정된 Stock_CNN식 feature extractor"]
    C --> D["flattened CNN feature vector"]
    D --> E["Linear adapter/head<br/>adapter_dim=128<br/>bias=False"]
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
split, normalization, metric 정의는 Stage 2에서 그대로 가져옵니다. 첫 실행은
`I60/R20/ohlc_ma_vb`, seed `42`이고, 이후 single-seed `36`-run grid로 갑니다.
