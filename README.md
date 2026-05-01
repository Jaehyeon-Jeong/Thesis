# Thesis

## English

This repository organizes the thesis experiment pipeline for reproducing and
extending the image-based CNN framework from *Re-Imag(in)ing Price Trends*.

The current public GitHub contents are documentation, reproducibility metadata,
small audit outputs, diagrams, configs, and code scaffolds. Large data files,
paper PDFs, checkpoints, and large prediction outputs are intentionally not
tracked.

### Pipeline

| Stage | Purpose | Current status |
| --- | --- | --- |
| `stage0_data_check` | Audit data, papers, and reference implementations | Completed |
| `stage1_reimage_reproduction` | Reproduce the Re-image CNN pipeline on public I20 stock images | In progress: `I20/R60` seed-42 fast diagnostic archived; `I20/R20` archive is smoke-only; `I20/R5`, strict batch-128 run, and five-seed reproduction are later |
| `stage2_btc_extension` | Extend the confirmed pipeline to BTC OHLCV | Single-seed 36-run result package closed for now; five-seed stability check later |
| `stage3_linear_adapter` | Add a Linear comparison model | Scaffold/checklist created; next stage after status cleanup |
| Stage 4 | Add FiLM + News/LLM conditioning | Planned |

### Current Status

Stage 1:
- Current usable full test artifact: `I20/R60`, seed `42`, fast Kaggle
  diagnostic.
- `I20/R60` snapshot: accuracy `0.5312`, majority accuracy `0.5408`,
  ROC-AUC `0.5298`, test rows `1,376,215`.
- `I20/R20` is not ready as a full result in the local archive. The preserved
  metrics/Grad-CAM are validation-smoke outputs only.
- `I20/R5` is not archived locally yet.
- Later: strict paper-style batch size `128`, five independent runs/seeds, and
  final `10` up + `10` down Figure-13-style Grad-CAM.

Stage 2:
- Current result package: BTC single-seed grid, `36` experiments
  (`I5/I20/I60` x `R5/R20/R60` x four image specs), seed `42`.
- Best single-seed configuration: `I60/R20/ohlc_ma_vb`.
- Remaining Stage 2 work is the five-seed rerun for stability.

### Key documents

- [PLAN.md](PLAN.md)
- [Overall pipeline diagram](docs/overall_pipeline_diagram.md)
- [Execution environment diagram](docs/execution_environment_diagram.md)
- [Stage 0 checklist](stage0_data_check/checklist.md)
- [Stage 1 checklist](stage1_reimage_reproduction/checklist.md)
- [Stage 2 checklist](stage2_btc_extension/checklist.md)
- [Stage 3 checklist](stage3_linear_adapter/checklist.md)

### Data policy

Tracked:
- Markdown plans and result reports
- Mermaid diagrams
- source maps
- configs
- small CSV summaries
- small sample figures

Not tracked:
- paper PDFs
- `.dat` image shards
- `.feather` source labels
- checkpoints
- large prediction CSVs
- old scratch/test code

## 한국어

이 저장소는 *Re-Imag(in)ing Price Trends*의 이미지 기반 CNN 파이프라인을
재현하고 BTC/Linear/FiLM으로 확장하기 위한 논문 실험 프로젝트입니다.

현재 GitHub에는 문서, 재현성 메타데이터, 작은 audit 산출물, 다이어그램,
config, 코드 scaffold만 올립니다. 대용량 데이터, 논문 PDF, checkpoint,
대용량 prediction output은 의도적으로 추적하지 않습니다.

### 파이프라인

| 단계 | 목적 | 현재 상태 |
| --- | --- | --- |
| `stage0_data_check` | 데이터, 논문, reference implementation 확인 | 완료 |
| `stage1_reimage_reproduction` | public I20 stock image로 Re-image CNN pipeline 재현 | 진행 중: `I20/R60` seed-42 fast diagnostic 보존; `I20/R20` archive는 smoke-only; `I20/R5`, strict batch-128 run, five-seed reproduction은 later |
| `stage2_btc_extension` | 확인된 pipeline을 BTC OHLCV로 확장 | single-seed 36-run 결과 패키지는 현재 마무리; 5-seed 안정성 확인은 later |
| `stage3_linear_adapter` | Linear 비교 모델 추가 | scaffold/checklist 생성, status 정리 후 다음 단계 |
| Stage 4 | FiLM + News/LLM conditioning 추가 | 계획 |

### 현재 상태

Stage 1:
- 현재 full test artifact로 사용할 수 있는 것은 `I20/R60`, seed `42`, fast
  Kaggle diagnostic입니다.
- `I20/R60` snapshot: accuracy `0.5312`, majority accuracy `0.5408`,
  ROC-AUC `0.5298`, test rows `1,376,215`.
- `I20/R20`은 로컬 archive 기준 full 결과가 아닙니다. 현재 보존된
  metrics/Grad-CAM은 validation-smoke output입니다.
- `I20/R5`는 아직 로컬에 보존되어 있지 않습니다.
- Later: 논문식 strict batch size `128`, five independent runs/seeds, 최종
  `10` up + `10` down Figure-13-style Grad-CAM.

Stage 2:
- 현재 결과 패키지: BTC single-seed grid, `36`개 실험
  (`I5/I20/I60` x `R5/R20/R60` x image spec 4개), seed `42`.
- Single-seed best configuration: `I60/R20/ohlc_ma_vb`.
- 남은 Stage 2 작업은 안정성 확인용 five-seed rerun입니다.

### 주요 문서

- [PLAN.md](PLAN.md)
- [전체 파이프라인 다이어그램](docs/overall_pipeline_diagram.md)
- [실행환경 다이어그램](docs/execution_environment_diagram.md)
- [Stage 0 체크리스트](stage0_data_check/checklist.md)
- [Stage 1 체크리스트](stage1_reimage_reproduction/checklist.md)
- [Stage 2 체크리스트](stage2_btc_extension/checklist.md)
- [Stage 3 체크리스트](stage3_linear_adapter/checklist.md)

### 데이터 정책

GitHub에 올리는 것:
- Markdown 계획과 결과 보고
- Mermaid diagram
- source map
- config
- 작은 CSV summary
- 작은 sample figure

GitHub에 올리지 않는 것:
- 논문 PDF
- `.dat` image shard
- `.feather` source label
- checkpoint
- 대용량 prediction CSV
- 이전 scratch/test code
