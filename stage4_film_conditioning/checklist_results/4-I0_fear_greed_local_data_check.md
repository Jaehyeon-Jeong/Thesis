# 4-I0 Fear & Greed Local Data Check

## English

Status: local F&G data added and quick-checked.

Local folder:

```text
stage4_film_conditioning/FG_data
```

Files found:

| File | Size | Stage 4 use |
| --- | ---: | --- |
| `fear_greed_index.csv` | 92 KB | Use for primary structured numeric context |
| `historical_data.csv` | 45 MB | Do not use in first numeric context run; appears to be transaction/order data |

`fear_greed_index.csv` quick audit:

| Item | Value |
| --- | --- |
| Columns | `timestamp`, `value`, `classification`, `date` |
| Rows | `2,644` |
| Date range | `2018-02-01` to `2025-05-02` |
| Duplicate dates | `0` |
| Missing days inside range | `4` |
| Missing dates | `2018-04-14`, `2018-04-15`, `2018-04-16`, `2024-10-26` |
| Value range | `5` to `95` |
| Stage 2 test period coverage | `1,460/1,461` days |

Decision:
- This removes the local F&G availability blocker from `4-I0`.
- `4-I2` can now implement and test local F&G feature alignment.
- Raw CSV files must remain local and should not be tracked in GitHub.
- Kaggle runs should still attach the public F&G dataset so the final output is
  reproducible in the Kaggle environment.

## 한국어

상태: 로컬 F&G data 추가 및 빠른 확인 완료.

로컬 폴더:

```text
stage4_film_conditioning/FG_data
```

확인된 파일:

| 파일 | 크기 | Stage 4 사용 |
| --- | ---: | --- |
| `fear_greed_index.csv` | 92 KB | primary structured numeric context에 사용 |
| `historical_data.csv` | 45 MB | 첫 numeric context run에서는 사용하지 않음. 거래/order data로 보임 |

`fear_greed_index.csv` 빠른 audit:

| 항목 | 값 |
| --- | --- |
| Columns | `timestamp`, `value`, `classification`, `date` |
| Rows | `2,644` |
| Date range | `2018-02-01` to `2025-05-02` |
| Duplicate dates | `0` |
| 범위 안 missing day | `4` |
| Missing dates | `2018-04-14`, `2018-04-15`, `2018-04-16`, `2024-10-26` |
| Value range | `5` to `95` |
| Stage 2 test period coverage | `1,460/1,461` days |

결정:
- `4-I0`의 local F&G availability blocker가 해소됐습니다.
- `4-I2`에서 local F&G feature alignment 구현과 테스트를 진행할 수 있습니다.
- Raw CSV 파일은 local data로 유지하고 GitHub에 track하지 않습니다.
- Kaggle run에서는 여전히 public F&G dataset을 attach해서 Kaggle 환경 재현성을
  유지해야 합니다.
