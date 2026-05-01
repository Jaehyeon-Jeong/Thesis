# Stage 2 Workflow Diagram

## English

```mermaid
flowchart TD
    A["BTC OHLCV Kaggle dataset"] --> B["BTC data audit"]
    B --> C["Generate 5/20/60-day BTC chart images"]
    C --> D["Label construction<br/>future return > 0 -> Up=1"]
    D --> E["Time split<br/>train / validation / test"]
    E --> F["Train-only normalization"]
    F --> G["Stage 1 Stock_CNN-style CNN core"]
    G --> H["BTC test prediction CSV"]
    H --> I["Classification metrics"]
    H --> J["Trading metrics<br/>long/flat and long/short"]
    H --> K["BTC Grad-CAM figures"]
    I --> L["Stage 2 report outputs"]
    J --> L
    K --> L
```

Stage 2 starts from BTC data preparation. Final comparisons against Stage 1 are
blocked until Stage 1 Kaggle full outputs exist.

## 한국어

```mermaid
flowchart TD
    A["BTC OHLCV Kaggle dataset"] --> B["BTC data audit"]
    B --> C["5/20/60일 BTC chart image 생성"]
    C --> D["label 생성<br/>future return > 0 -> Up=1"]
    D --> E["time split<br/>train / validation / test"]
    E --> F["train-only normalization"]
    F --> G["Stage 1 Stock_CNN식 CNN core"]
    G --> H["BTC test prediction CSV"]
    H --> I["classification metric"]
    H --> J["trading metric<br/>long/flat, long/short"]
    H --> K["BTC Grad-CAM 그림"]
    I --> L["Stage 2 report output"]
    J --> L
    K --> L
```

Stage 2는 BTC 데이터 준비부터 시작합니다. Stage 1 Kaggle full output이 나오기 전까지
Stage 1과의 최종 비교는 보류합니다.
