# Stage 0: Source and Data Check

## English

This folder is only for Stage 0. No model implementation or training code
belongs here.

Stage 0 goal:
- Confirm what data is actually available locally.
- Confirm what the author-provided `monthly_20d` dataset contains.
- Confirm what can and cannot be reproduced in Stage 1.
- Confirm source references before writing the Re-image reproduction code.

Fixed local sources:
- Re-image summary: `../자료조사/Re-image 요약.md`
- Re-image PDF: `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- Stock public image data: `../테스트/Test/img_data/monthly_20d`
- Re-image GitHub reference: `https://github.com/lich99/Stock_CNN`

Stage 0 outputs:
- `docs/source_reference_check.md`
- `docs/monthly20_data_check.md`
- `data_inventory/monthly20_files.csv`
- `data_inventory/monthly20_labels.csv`
- `outputs/figures/sample_images/`
- `logs/`

Important rule:
- Stage 0 only checks and documents data/source facts.
- Stage 1 implementation starts only after Stage 0 findings are reviewed.

## 한국어

이 폴더는 0단계 전용입니다. 모델 구현 코드나 학습 코드는 이 폴더에 넣지 않습니다.

0단계 목표:
- 로컬에 실제로 어떤 데이터가 있는지 확인합니다.
- 저자 공개 `monthly_20d` 데이터가 무엇을 포함하는지 확인합니다.
- 1단계에서 무엇을 재현할 수 있고, 무엇은 추가 데이터가 필요한지 정리합니다.
- Re-image 재현 코드를 작성하기 전에 논문과 참고 GitHub 근거를 확인합니다.

고정 로컬 자료:
- Re-image 요약: `../자료조사/Re-image 요약.md`
- Re-image PDF: `../자료조사/Xiu-Re-Imagining-Price-Trends.pdf`
- 저자 공개 주식 이미지 데이터: `../테스트/Test/img_data/monthly_20d`
- Re-image GitHub 참고 구현: `https://github.com/lich99/Stock_CNN`

0단계 산출물:
- `docs/source_reference_check.md`
- `docs/monthly20_data_check.md`
- `data_inventory/monthly20_files.csv`
- `data_inventory/monthly20_labels.csv`
- `outputs/figures/sample_images/`
- `logs/`

중요 규칙:
- 0단계는 자료와 데이터 사실만 확인하고 문서화합니다.
- 1단계 구현은 0단계 확인 결과를 검토한 뒤에만 시작합니다.
