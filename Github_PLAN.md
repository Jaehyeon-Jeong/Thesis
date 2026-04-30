# GitHub `Thesis` Publishing Plan

## English

## Permanent Operating Rule

This file is the fixed GitHub publishing rule for this thesis workspace.

For every future work unit:
1. Check this file before starting the work.
2. Apply the agreed GitHub structure and upload policy.
3. Update the clean publish repo at `/Users/jaehyeonjeong/Desktop/논문/Thesis`.
4. Save checklist result reports under each stage's `checklist_results/`.
5. Update the stage `checklist.md` with links to result reports.
6. Create or update the stage `workflow_diagram.md` when a new stage starts.
7. Exclude paper PDFs, raw data, checkpoints, large predictions, and scratch
   test files from GitHub.
8. Unless the user explicitly says not to push, commit and push the publish repo
   after each completed work unit.

Default commit/push behavior:
- Commit only clean publish artifacts in `/Users/jaehyeonjeong/Desktop/논문/Thesis`.
- Do not turn the whole source workspace `/Users/jaehyeonjeong/Desktop/논문` into
  a Git repo.
- Before pushing, verify that blocked file types are not tracked.
- After pushing, verify local `HEAD` and remote `HEAD` match.

## Repository Structure

```text
Thesis/
  README.md
  PLAN.md
  Github_PLAN.md
  .gitignore
  docs/
  stage0_data_check/
  stage1_reimage_reproduction/
  stage2_btc_extension/
  stage3_linear_adapter/
  stage4_film_news_llm/
```

## Upload Policy

Track:
- code
- configs
- checklists
- checklist result reports
- source maps
- workflow diagrams
- small summary tables
- small sample figures

Do not track:
- paper PDFs
- raw `.dat` image shards
- source `.feather` labels
- checkpoints
- large predictions
- old scratch/test code
- `.DS_Store`

## 한국어

## 영구 운영 규칙

이 파일은 이 논문 작업공간의 GitHub publish 고정 규칙입니다.

앞으로 모든 작업 단위마다:
1. 작업 시작 전에 이 파일을 확인합니다.
2. 합의한 GitHub 구조와 업로드 정책을 적용합니다.
3. clean publish repo인 `/Users/jaehyeonjeong/Desktop/논문/Thesis`를 업데이트합니다.
4. 각 stage의 `checklist_results/`에 체크리스트 결과 보고를 저장합니다.
5. stage `checklist.md`에는 결과 보고 링크를 연결합니다.
6. 새 stage를 시작할 때는 `workflow_diagram.md`를 만들거나 업데이트합니다.
7. 논문 PDF, raw data, checkpoint, 대용량 prediction, scratch/test file은 GitHub에서 제외합니다.
8. 사용자가 명시적으로 push를 막지 않으면, 작업 단위가 끝날 때마다 publish repo를 commit/push합니다.

기본 commit/push 동작:
- `/Users/jaehyeonjeong/Desktop/논문/Thesis` 안의 clean publish artifact만 commit합니다.
- 원본 작업공간 `/Users/jaehyeonjeong/Desktop/논문` 전체를 Git repo로 만들지 않습니다.
- push 전 금지 파일 형식이 tracking되지 않는지 확인합니다.
- push 후 local `HEAD`와 remote `HEAD`가 일치하는지 확인합니다.

## Repository 구조

```text
Thesis/
  README.md
  PLAN.md
  Github_PLAN.md
  .gitignore
  docs/
  stage0_data_check/
  stage1_reimage_reproduction/
  stage2_btc_extension/
  stage3_linear_adapter/
  stage4_film_news_llm/
```

## 업로드 정책

추적:
- code
- config
- checklist
- checklist result report
- source map
- workflow diagram
- 작은 summary table
- 작은 sample figure

추적하지 않음:
- 논문 PDF
- raw `.dat` image shard
- source `.feather` label
- checkpoint
- 대용량 prediction
- 이전 scratch/test code
- `.DS_Store`
