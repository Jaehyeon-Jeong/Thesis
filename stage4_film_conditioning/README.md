# Stage 4: FiLM Conditioning

## English

This folder is reserved for Stage 4 of the thesis pipeline.

Stage 4 objective:
- Keep the Stage 2 BTC image-generation, label, split, normalization,
  evaluation, trading metric, and Grad-CAM pipeline fixed.
- Add FiLM modulation inside the Stock_CNN-style convolution blocks.
- Compare BTC baseline, Stage 3 Linear, and FiLM variants.
- Follow the core implementation idea of `ethanjperez/film` as closely as the
  BTC chart-classification setting allows.

Current scope:
- Today is **FiLM only**.
- News data and LLM conditioning are not implemented today.
- The first Stage 4 work is the folder/checklist/source-map scaffold and the
  FiLM insertion design.

Condition-source tracks:
- `4A FiLM-only control`: no external data. This is a mechanism/smoke-test
  track. If used, gamma/beta are static learned parameters or generated from an
  explicitly documented no-leakage internal condition.
- `4B F&G index + FiLM`: daily Fear & Greed style numeric sentiment condition.
  This requires a separate data-source audit before implementation.
- `4C News + non-LLM encoder + FiLM`: BTC news is encoded by a non-LLM text
  encoder and then used to generate FiLM gamma/beta.
- `4D News + LLM encoder + FiLM`: the same BTC news idea, but the news is
  encoded by an LLM before generating FiLM gamma/beta. This is deferred.

Key modeling rule:
- Do not change the Stage 2 BTC data pipeline.
- Do not change the Stock_CNN block count for I5/I20/I60.
- Insert FiLM inside each CNN block after BatchNorm and before activation:
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.
- Implement both Gamma-only FiLM and Full FiLM when the first FiLM model is
  ready for comparison.
- Save gamma/beta values by sample/date/layer whenever FiLM is used.

Current status:
- Stage 4 folder and checklist scaffold are created.
- Stage 4 is not yet implemented.
- Stage 4 starts with FiLM-only planning. F&G, News, and LLM tracks are listed
  but deferred until their data/source audits are done.

Main documents:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 4 pipeline](docs/stage4_pipeline.md)
- [Source map](docs/source_map.md)
- [FiLM reference review](docs/film_reference_review.md)
- [Condition track plan](docs/condition_track_plan.md)
- [FiLM insertion design](docs/film_insertion_design.md)

## 한국어

이 폴더는 논문 파이프라인의 4단계 작업 공간입니다.

4단계 목표:
- Stage 2의 BTC image generation, label, split, normalization, evaluation,
  trading metric, Grad-CAM 파이프라인은 고정합니다.
- Stock_CNN-style convolution block 내부에 FiLM modulation을 추가합니다.
- BTC baseline, Stage 3 Linear, FiLM variant를 비교합니다.
- BTC chart classification setting이 허용하는 범위에서
  `ethanjperez/film`의 핵심 구현 아이디어를 최대한 따릅니다.

현재 범위:
- 오늘은 **FiLM only**입니다.
- News data와 LLM conditioning은 오늘 구현하지 않습니다.
- 첫 Stage 4 작업은 folder/checklist/source-map scaffold와 FiLM 삽입 설계입니다.

Condition-source track:
- `4A FiLM-only control`: 외부 데이터를 쓰지 않습니다. mechanism/smoke-test
  용도입니다. 사용한다면 gamma/beta는 static learned parameter이거나, leakage가
  없다고 문서화한 내부 condition에서 생성합니다.
- `4B F&G index + FiLM`: daily Fear & Greed 계열 numeric sentiment condition입니다.
  구현 전에 별도 data-source audit이 필요합니다.
- `4C News + non-LLM encoder + FiLM`: BTC news를 LLM이 아닌 text encoder로
  condition vector로 바꾼 뒤 FiLM gamma/beta를 생성하는 track입니다.
- `4D News + LLM encoder + FiLM`: 같은 BTC news를 LLM encoder로 condition
  vector로 바꾼 뒤 FiLM gamma/beta를 생성하는 track입니다. 나중으로 미룹니다.

핵심 모델링 규칙:
- Stage 2 BTC data pipeline은 바꾸지 않습니다.
- I5/I20/I60의 Stock_CNN block 수는 바꾸지 않습니다.
- FiLM은 각 CNN block 내부의 BatchNorm 뒤, activation 전에 삽입합니다:
  `Conv2d -> BatchNorm2d -> FiLM -> LeakyReLU -> MaxPool2d`.
- 첫 FiLM 모델 비교가 준비되면 Gamma-only FiLM과 Full FiLM을 모두 구현합니다.
- FiLM을 사용하면 sample/date/layer 기준 gamma/beta 값을 저장합니다.

현재 상태:
- Stage 4 folder와 checklist scaffold를 만들었습니다.
- Stage 4 구현은 아직 시작하지 않았습니다.
- Stage 4는 FiLM-only planning부터 시작합니다. F&G, News, LLM track은 목록에
  올려두되, data/source audit 전까지 구현하지 않습니다.

주요 문서:
- [Checklist](checklist.md)
- [Workflow diagram](workflow_diagram.md)
- [Stage 4 pipeline](docs/stage4_pipeline.md)
- [Source map](docs/source_map.md)
- [FiLM reference review](docs/film_reference_review.md)
- [Condition track plan](docs/condition_track_plan.md)
- [FiLM insertion design](docs/film_insertion_design.md)
