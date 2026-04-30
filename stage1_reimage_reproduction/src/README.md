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

Not implemented yet:
- labels/splits/normalization
- model
- training
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

아직 구현하지 않은 것:
- label/split/normalization
- model
- training
- evaluation
- Grad-CAM
