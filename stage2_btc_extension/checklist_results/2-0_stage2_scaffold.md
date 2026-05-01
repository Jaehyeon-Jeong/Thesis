# 2-0 Stage 2 Scaffold

## English

Result:
- Created the Stage 2 workspace for the BTC asset-class extension.
- Added a Stage 2 checklist, workflow diagram, pipeline document, and source map.
- Fixed the Stage 2 starting boundary: preparation can proceed while Stage 1
  full Kaggle runs are still executing, but final Stage 2 comparison waits for
  Stage 1 full outputs.
- Fixed the Stage 2 default batch policy: use paper batch size `128` because
  BTC data is smaller than the public stock shard.

Outputs:
- [README.md](../README.md)
- [checklist.md](../checklist.md)
- [workflow_diagram.md](../workflow_diagram.md)
- [stage2_pipeline.md](../docs/stage2_pipeline.md)
- [source_map.md](../docs/source_map.md)

## 한국어

결과:
- BTC 자산군 확장을 위한 Stage 2 작업공간을 만들었습니다.
- Stage 2 checklist, workflow diagram, pipeline 문서, source map을 추가했습니다.
- Stage 2 시작 경계를 고정했습니다: Stage 1 full Kaggle run이 도는 동안 준비 작업은
  진행할 수 있지만, Stage 2 최종 비교는 Stage 1 full output 이후 확정합니다.
- Stage 2 기본 batch 정책을 고정했습니다: BTC 데이터는 public stock shard보다 작으므로
  논문 batch size `128`을 사용합니다.

산출물:
- [README.md](../README.md)
- [checklist.md](../checklist.md)
- [workflow_diagram.md](../workflow_diagram.md)
- [stage2_pipeline.md](../docs/stage2_pipeline.md)
- [source_map.md](../docs/source_map.md)
