# 3-I2 Linear Adapter Model Code

## English

Status: complete

Model file:
- `src/stage3_linear/models/linear_stock_cnn.py`

Insertion point:
```text
image -> Stage2 CNN blocks -> flatten -> Linear(flatten_dim, 128, bias=False)
      -> Linear(128, 2, bias=False) -> logits
```

Validated parameter counts:
- I5: Stage 2 baseline `155,138`, Stage 3 Linear `2,090,752`
- I20: Stage 2 baseline `708,866`, Stage 3 Linear `6,515,200`
- I60: Stage 2 baseline `2,952,962`, Stage 3 Linear `26,177,536`

## 한국어

상태: 완료

모델 파일:
- `src/stage3_linear/models/linear_stock_cnn.py`

삽입 위치:
```text
image -> Stage2 CNN blocks -> flatten -> Linear(flatten_dim, 128, bias=False)
      -> Linear(128, 2, bias=False) -> logits
```

검증된 parameter 수:
- I5: Stage 2 baseline `155,138`, Stage 3 Linear `2,090,752`
- I20: Stage 2 baseline `708,866`, Stage 3 Linear `6,515,200`
- I60: Stage 2 baseline `2,952,962`, Stage 3 Linear `26,177,536`

