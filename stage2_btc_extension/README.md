# Stage 2: BTC Asset-Class Extension

## English

This folder is reserved for Stage 2 of the thesis pipeline.

Stage 2 objective:
- Keep the confirmed Re-image/Stock_CNN-style image CNN pipeline from Stage 1.
- Change the asset universe from public stock image shards to BTC OHLCV.
- Generate BTC chart images directly from raw OHLCV.
- Evaluate the BTC single-asset setting with both classification metrics and
  time-series trading metrics.
- Produce BTC Grad-CAM figures for every baseline run.

Current boundary:
- Stage 2 can start while Stage 1 Kaggle full runs are still running.
- Stage 2 final comparison and report must wait until Stage 1 full outputs are
  available.
- Stage 2 should use the paper batch size `128` by default because BTC has many
  fewer samples than the public stock shard.

Primary data:
- BTC OHLCV: `kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024`

Main documents:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 2 pipeline](docs/stage2_pipeline.md)
- [BTC image generation plan](docs/stage2_image_generation_plan.md)
- [Source map](docs/source_map.md)

## 한국어

이 폴더는 논문 파이프라인의 2단계 작업 공간입니다.

2단계 목표:
- 1단계에서 확정한 Re-image/Stock_CNN식 image CNN 파이프라인을 유지합니다.
- 자산군만 public stock image shard에서 BTC OHLCV로 바꿉니다.
- BTC raw OHLCV에서 chart image를 직접 생성합니다.
- BTC 단일 자산 setting에서는 classification metric과 time-series trading
  metric을 함께 봅니다.
- 모든 BTC baseline run에서 Grad-CAM 그림을 생성합니다.

현재 경계:
- Stage 1 Kaggle full run이 도는 동안 Stage 2 준비 작업은 시작할 수 있습니다.
- Stage 2의 최종 비교와 보고서는 Stage 1 full output이 나온 뒤 확정합니다.
- BTC dataset은 stock shard보다 훨씬 작으므로 Stage 2 기본 batch size는 논문값
  `128`을 유지합니다.

주요 데이터:
- BTC OHLCV: `kaggle.com/datasets/novandraanugrah/bitcoin-historical-datasets-2018-2024`

주요 문서:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 2 pipeline](docs/stage2_pipeline.md)
- [BTC image generation plan](docs/stage2_image_generation_plan.md)
- [Source map](docs/source_map.md)
