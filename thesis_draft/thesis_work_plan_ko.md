# 논문 초안 작업 계획

## 기준

- 작업 위치: `/Users/jaehyeonjeong/Desktop/논문/thesis_draft`
- 학교 기준: MIPT
- 논문 유형: Bachelor thesis
- formatting rule source: 사용자가 제공한 MIPT PDF 2개만 사용
- HSE/다른 대학/일반 ВКР 자료는 formatting에 사용하지 않음
- 본문 초안: 영어
- 계획/체크리스트/진행 기록: 한국어
- 목표 분량: appendix 제외 30-40 pages
- originality 기준: 최소 70%

## 현재 완료

- [x] MIPT PDF 2개 위치 확인
- [x] PDF가 스캔본임을 확인
- [x] macOS Vision OCR로 규정 텍스트 추출
- [x] formatting requirement 1차 정리
- [x] PDF에서 확인 불가능한 항목을 별도로 분리
- [x] Bachelor thesis 기준으로 분량/originality 기준 고정
- [x] 별도 체크리스트 파일 생성: `checklist.md`
- [x] 초기 교수님 자료와 현재 실험의 연결점 정리: `source_extraction_notes.md`
- [x] thesis evidence map 초안 작성: `evidence_map.md`
- [x] 영어 thesis skeleton 초안 작성: `thesis_draft.md`
- [x] source/license register 초안 작성: `source_license_register.md`
- [x] Stage2/N15/N16 image variant 설명 반영
  - `ohlc`, `ohlc_ma`, `ohlc_vb`, `ohlc_ma_vb` baseline 비교
  - `ohlc_vb + funding_plus_cftc_oi`를 same-image positive case로 정리

## 다음 작업 순서

1. 영어 thesis draft v0 작성: skeleton을 실제 paragraph로 확장
2. Stage1-Stage5 결과를 chapter별로 연결
3. Stage4 `4-N14B2-B6` 조건부 해석을 보강 분석으로 실행/반영
4. supervisor에게 보낼 1-page summary 초안 작성
5. `VKR_format_requirements.md` 기준으로 DOCX 변환 준비

## 초안 구조

1. Introduction
2. Related Work
3. Data and Preprocessing
4. Methodology
5. Experimental Design
6. Results
7. Interpretability and Conditional Analysis
8. Limitations
9. Conclusion

## 남은 분석

### Stage5

- [x] `5-11`: FinBERT+F&G correction/regression 조건별 분석
- [x] `5-12`: Grad-CAM + gamma/beta + news/F&G context export
- [x] `5-13`: Stage5 최종 보고 및 thesis title decision
- 결론: Stage5는 1차 thesis draft 기준으로 종료. 최종 Stage5 claim은
  strong LLM/news outperformance가 아니라 small positive / conditional
  calibration.
- 추천 제목:
  `Context-Conditioned FiLM for Bitcoin Direction Prediction from Price Charts`

### Stage4

- [ ] `4-N14B2-B6`: N16 derivatives/leverage context 조건부 해석
- 목적: 전체 accuracy가 아니라 특정 regime에서 correction이 발생하는지 확인

## 논문 주장 작성 원칙

- 모든 성능 주장은 CSV/report/figure에 연결
- 모든 데이터/모델/API는 source/license register에 연결
- 전체 성능 향상이 약한 경우, 조건부 향상과 해석 가능성을 분리해서 서술
- 실패한 실험도 contribution으로 정리: strong visual baseline 위에서 context-FiLM이 왜 보수적으로 작동했는지 설명
