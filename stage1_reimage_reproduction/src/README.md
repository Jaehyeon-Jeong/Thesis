# Source Code

## English

Stage 1 source code is implemented one checklist gate at a time.

Current package:
- `stage1_reimage.config`: YAML config loading and validation.
- `stage1_reimage.paths`: path objects and output-directory helpers.
- `stage1_reimage.runtime`: runtime device selection.
- `stage1_reimage.seed`: reproducibility seed helper.
- `stage1_reimage.data.monthly20`: lazy/memmap loading for public
  `monthly_20d` image/label shards.
- `stage1_reimage.data.label_split`: horizon labels, deterministic split
  metadata, and train-only normalization metadata.
- `stage1_reimage.models.stock_cnn`: GitHub-style `StockCNNI20` baseline
  model for public I20 reproduction.
- `stage1_reimage.training.loop`: training loop, Xavier initialization, Adam,
  early stopping, checkpoint, history, and metadata helpers.

Not implemented yet:
- evaluation
- Grad-CAM

## 한국어

1단계 source code는 체크리스트 gate별로 하나씩 구현합니다.

현재 package:
- `stage1_reimage.config`: YAML config loading과 validation.
- `stage1_reimage.paths`: path 객체와 output-directory helper.
- `stage1_reimage.runtime`: runtime device 선택.
- `stage1_reimage.seed`: reproducibility seed helper.
- `stage1_reimage.data.monthly20`: public `monthly_20d` image/label shard
  lazy/memmap loading.
- `stage1_reimage.data.label_split`: horizon label, deterministic split
  metadata, train-only normalization metadata.
- `stage1_reimage.models.stock_cnn`: public I20 재현용 GitHub식
  `StockCNNI20` baseline model.
- `stage1_reimage.training.loop`: training loop, Xavier initialization, Adam,
  early stopping, checkpoint, history, metadata helper.

아직 구현하지 않은 것:
- evaluation
- Grad-CAM
