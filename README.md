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
| `stage1_reimage_reproduction` | Reproduce the Re-image CNN pipeline on public I20 stock images | Planning + readiness completed; code implementation next |
| Stage 2 | Extend the confirmed pipeline to BTC OHLCV | Planned |
| Stage 3 | Add a Linear comparison model | Planned |
| Stage 4 | Add FiLM + News/LLM conditioning | Planned |

### Key documents

- [PLAN.md](PLAN.md)
- [Overall pipeline diagram](docs/overall_pipeline_diagram.md)
- [Execution environment diagram](docs/execution_environment_diagram.md)
- [Stage 0 checklist](stage0_data_check/checklist.md)
- [Stage 1 checklist](stage1_reimage_reproduction/checklist.md)

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
| `stage1_reimage_reproduction` | public I20 stock image로 Re-image CNN pipeline 재현 | 계획 + readiness 완료, 다음은 코드 구현 |
| Stage 2 | 확인된 pipeline을 BTC OHLCV로 확장 | 계획 |
| Stage 3 | Linear 비교 모델 추가 | 계획 |
| Stage 4 | FiLM + News/LLM conditioning 추가 | 계획 |

### 주요 문서

- [PLAN.md](PLAN.md)
- [전체 파이프라인 다이어그램](docs/overall_pipeline_diagram.md)
- [실행환경 다이어그램](docs/execution_environment_diagram.md)
- [Stage 0 체크리스트](stage0_data_check/checklist.md)
- [Stage 1 체크리스트](stage1_reimage_reproduction/checklist.md)

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
