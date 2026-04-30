# 1-I4 Baseline CNN Model Implementation

## English

Status:
- Completed on 2026-04-30.

Purpose:
- Implement the Stage 1 I20 baseline CNN following the checked
  `lich99/Stock_CNN` model core.
- Keep the model hookable for later Grad-CAM.

Implemented files:
- `src/stage1_reimage/models/__init__.py`
- `src/stage1_reimage/models/stock_cnn.py`
- `scripts/check_model.py`
- Updated `configs/env_local.yaml`
- Updated `configs/env_kaggle.yaml`

Reference implementation:
- Repository: `https://github.com/lich99/Stock_CNN`
- Commit: `415e2acf2a5013afca67e383acd3edc61fced840`
- File: `models/baseline.py`

Paper reference:
- Re-image local summary maps CNN architecture and training details to
  pp. 12-21.
- Re-image Figure 7 is mapped to p. 18.

Implemented architecture:
- `layer1`: Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d.
- `layer2`: Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d.
- `layer3`: Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d.
- `fc1`: Dropout -> Linear.
- `forward` returns logits with shape `(batch_size, 2)`.
- Softmax is intentionally not applied inside `forward`.

Validation command:

```bash
python scripts/check_model.py --config configs/env_local.yaml --batch-size 2
```

Validation result:
- Passed locally.
- Parameter count: `708,866`.
- Shape checks:
  - input `(2, 1, 64, 60)`
  - after `layer1` `(2, 64, 13, 60)`
  - after `layer2` `(2, 128, 5, 60)`
  - after `layer3` `(2, 256, 3, 60)`
  - flatten `(2, 46080)`
  - logits `(2, 2)`
- Grad-CAM target layers resolve to:
  - `layer1_conv`
  - `layer2_conv`
  - `layer3_conv`

Known mismatch:
- The local paper summary emphasizes first-layer vertical dilation.
- The checked GitHub I20 implementation applies `dilation=(2, 1)` to all three
  convolution layers.
- Stage 1 follows the GitHub model core by user instruction and documents the
  mismatch.

Scope limits:
- No loss, optimizer, training loop, checkpoint, evaluation metrics, prediction
  CSV, or Grad-CAM algorithm was implemented in this gate.

## 한국어

상태:
- 2026-04-30 완료.

목적:
- 확인한 `lich99/Stock_CNN` model core를 따라 Stage 1 I20 baseline CNN을
  구현합니다.
- 이후 Grad-CAM에서 hook을 걸 수 있게 구조를 유지합니다.

구현한 파일:
- `src/stage1_reimage/models/__init__.py`
- `src/stage1_reimage/models/stock_cnn.py`
- `scripts/check_model.py`
- `configs/env_local.yaml` 업데이트
- `configs/env_kaggle.yaml` 업데이트

기준 구현:
- Repository: `https://github.com/lich99/Stock_CNN`
- Commit: `415e2acf2a5013afca67e383acd3edc61fced840`
- File: `models/baseline.py`

논문 근거:
- Re-image 로컬 요약은 CNN architecture/training detail을 pp. 12-21로
  매핑합니다.
- Re-image Figure 7은 p. 18로 매핑되어 있습니다.

구현한 구조:
- `layer1`: Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d.
- `layer2`: Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d.
- `layer3`: Conv2d -> BatchNorm2d -> LeakyReLU -> MaxPool2d.
- `fc1`: Dropout -> Linear.
- `forward`는 `(batch_size, 2)` logits를 반환합니다.
- softmax는 `forward` 안에서 의도적으로 적용하지 않습니다.

검증 명령:

```bash
python scripts/check_model.py --config configs/env_local.yaml --batch-size 2
```

검증 결과:
- 로컬에서 통과했습니다.
- Parameter count: `708,866`.
- Shape check:
  - input `(2, 1, 64, 60)`
  - after `layer1` `(2, 64, 13, 60)`
  - after `layer2` `(2, 128, 5, 60)`
  - after `layer3` `(2, 256, 3, 60)`
  - flatten `(2, 46080)`
  - logits `(2, 2)`
- Grad-CAM target layer:
  - `layer1_conv`
  - `layer2_conv`
  - `layer3_conv`

알려진 mismatch:
- 로컬 논문 요약은 first-layer vertical dilation을 강조합니다.
- 확인한 GitHub I20 구현은 세 convolution layer 모두에 `dilation=(2, 1)`을
  적용합니다.
- 1단계는 사용자 지시에 따라 GitHub model core를 따르고, 이 차이를 문서화합니다.

범위 제한:
- 이 gate에서는 loss, optimizer, training loop, checkpoint, evaluation metric,
  prediction CSV, Grad-CAM algorithm을 구현하지 않았습니다.
