# Stage 4 Workflow Diagram

## English

```mermaid
flowchart TD
    A[Stage 2 BTC fixed pipeline] --> B[BTC chart image tensors]
    B --> C[Stock_CNN I5/I20/I60 blocks]
    C --> D[Insert FiLM after BatchNorm]

    S0[4A FiLM-only control] --> G[FiLM generator]
    S1[4B F&G numeric condition later] -. deferred .-> G
    S2[4C News + non-LLM encoder later] -. deferred .-> G
    S3[4D News + LLM encoder later] -. deferred .-> G

    G --> H[gamma and beta per block/channel]
    H --> D
    D --> I[logits and probabilities]
    I --> J[classification metrics]
    I --> K[trading metrics]
    I --> L[Grad-CAM figures]
    H --> M[gamma/beta parquet or csv export]
    J --> N[Stage 4 report]
    K --> N
    L --> N
    M --> N
```

## 한국어

```mermaid
flowchart TD
    A[Stage 2 BTC 고정 pipeline] --> B[BTC chart image tensor]
    B --> C[Stock_CNN I5/I20/I60 block]
    C --> D[BatchNorm 뒤 FiLM 삽입]

    S0[4A FiLM-only control] --> G[FiLM generator]
    S1[4B F&G numeric condition later] -. later .-> G
    S2[4C News + non-LLM encoder later] -. later .-> G
    S3[4D News + LLM encoder later] -. later .-> G

    G --> H[block/channel별 gamma와 beta]
    H --> D
    D --> I[logit과 probability]
    I --> J[classification metric]
    I --> K[trading metric]
    I --> L[Grad-CAM figure]
    H --> M[gamma/beta parquet 또는 csv export]
    J --> N[Stage 4 report]
    K --> N
    L --> N
    M --> N
```
