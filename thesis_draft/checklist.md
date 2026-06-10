# Thesis Draft Checklist

## Current Position

현재 thesis draft 작업은 아래 순서로 진행합니다.

1. Formatting / rulebook 확인은 1차 완료.
2. Source extraction / thesis skeleton / evidence map은 1차 완료.
3. **영어 thesis draft v1 재구성 완료**. 내부 실험명 `Stage1/Stage2`
   중심 구조를 제거하고, 일반 ML/금융 논문 흐름인
   `Introduction -> Literature Review -> Data -> Methodology -> Experimental
   Design -> Empirical Results -> Interpretability -> Limitations -> Conclusion`
   구조로 재작성했습니다. MIPT 지침은 title/abstract/contents/references
   같은 문서 순서와 formatting에만 적용합니다.

연결된 상세 체크리스트:

- Stage5 상세 체크리스트: `../stage5_llm_news_embedding/checklist.md`
- GitHub publish 체크리스트: `../Thesis/stage5_llm_news_embedding/checklist.md`

바로 이어서 할 항목:

- [x] `5-11`: FinBERT+F&G correction/regression 조건별 분석
- [x] `5-12`: Grad-CAM + gamma/beta + news/F&G context export
- [x] `5-13`: Stage5 final report and thesis title decision
- [x] `5-14`: FinBERT+F&G gamma/beta relaxation ablation
- [x] `4-N14B2-B6`: N16 derivatives/leverage context 조건부 해석
- [x] thesis draft v0: chapter별 본문 확장
- [x] LaTeX/PDF preview 재작성
  - `thesis_draft.tex`
  - `thesis_draft.pdf`
  - 현재 preview는 32쪽으로, bachelor 권장 30-40쪽 범위에 진입
  - Abstract는 단독 페이지, Contents는 별도 페이지, Abbreviations는 표 형식
- [x] thesis draft v1: academic thesis 구조로 재작성
  - Abstract 추가
  - 내부 Stage 명칭은 Appendix의 experiment log mapping으로 이동
  - 본문은 research question / data / method / result / discussion 중심으로 정리
- [x] thesis draft v2: 내부 Stage 명칭을 본문/부록에서도 제거
  - repository artifact는 `Reproducibility Artifacts`로만 정리
  - PDF/LaTeX를 primary source로 고정
- [x] Stage2/N15 image-variant evidence 연결
  - `ohlc`, `ohlc_ma`, `ohlc_vb`, `ohlc_ma_vb` baseline 비교
  - `ohlc_vb` derivatives/leverage same-image positive case 설명
- [x] missing evidence / optional experiment 정리
  - `missing_evidence_and_experiments.md`

## 0. Formatting / Rulebook

- [x] MIPT PDF 2개만 공식 formatting source로 고정
- [x] HSE/다른 대학 formatting source 제외
- [x] 스캔 PDF OCR 확인
- [x] `VKR_format_requirements.md` 작성
- [x] Bachelor thesis 기준 반영
  - [x] appendix 제외 30-40 pages
  - [x] originality 최소 70%
- [ ] VKR regulation PDF page 12 수동 확인
- [ ] 최종 title page는 MIPT personal cabinet / кафедра 양식 확인
- [ ] 영어 thesis에서 `Figure/Table` caption 표기 허용 여부 supervisor 또는 кафедра에 확인

## 1. Thesis Source Preparation

- [x] `자료조사/Research Idea.docx`에서 유지할 부분 추출
  - research gap
  - motivation
  - FiLM + multimodal market context 출발점
- [x] `자료조사/Thesis plan.docx`에서 유지할 부분 추출
- [x] 교수님 방향성 문서에서 thesis에 들어갈 핵심 설계 이유 추출
- [x] 현재 실험 결과와 달라진 초기 계획 항목 정리
  - surge/crash/neutral -> R20 Up/Down
  - broad social media -> BTC headline news, FinBERT, F&G
  - full LLM reasoning -> news-derived numeric context / embedding / sentiment vectors
  - main model -> frozen Stage2 CNN + bounded last-block FiLM
- [x] 추출 결과 정리: `source_extraction_notes.md`

## 2. Draft Skeleton

- [x] 영어 thesis skeleton 작성
- [x] annotation draft 작성, 1500 characters 이하
- [x] chapter별 claim/evidence placeholder 연결
- [x] MIPT 기본 formatting을 반영한 LaTeX preview 생성
- [x] figure/table placeholder 생성
  - 최종 draft에는 7개 figure, 9개 numbered table, 2개 appendix protocol이 포함됨

Target chapter structure:

1. Introduction
2. Related Work
3. Data and Preprocessing
4. Methodology
5. Experimental Design
6. Results
7. Interpretability and Conditional Analysis
8. Limitations
9. Conclusion

## 3. Citation / License Register

- [x] 논문 citation table 초안 작성
- [x] 데이터 source/license table 초안 작성
- [x] 모델/API source table 초안 작성
  - OpenAI embedding API
  - FinBERT / Hugging Face model
  - F&G source
  - Kaggle datasets
  - CFTC/OFR/public financial data
- [x] raw news text 재배포 금지 표시
- [x] API key 저장 금지 표시
- [x] large checkpoint/raw data GitHub 업로드 금지 표시

## 4. Remaining Analysis Before Draft Claims

### Stage5

- [x] `5-11`: FinBERT+F&G correction/regression 조건별 분석
  - Stage2 wrong -> Stage5 correct
  - Stage2 correct -> Stage5 wrong
  - extreme F&G / neutral F&G
  - high positive / high negative FinBERT sentiment
  - 결과: Stage5 accuracy `0.580569`, Stage2 accuracy `0.579320`,
    corrections `95`, regressions `86`, net corrections `+9`.
  - 조건부 positive bucket: Stage2 uncertain 45-55, F&G greed,
    high/news-positive FinBERT regimes.
- [x] `5-12`: Grad-CAM + gamma/beta + news/F&G context export
  - 결과: targeted Grad-CAM/report artifact `30`개, selected modulation rows
    `40`개.
  - 해석: FinBERT+F&G FiLM은 큰 feature rewrite가 아니라 gamma mean
    약 `1.0003`, beta mean 약 `0.00008` 수준의 conservative calibration.
  - probability 변화는 주로 `prob_up`을 낮추는 방향이라 false-Up correction과
    true-Up regression을 동시에 설명함.
- [x] `5-13`: Stage5 final report and thesis title decision
  - 결과: Stage5는 1차 thesis draft 기준으로 종료.
  - 최종 Stage5 후보: frozen Stage2 `I60/R20/ohlc_ma_vb` + FinBERT+F&G
    bounded last-block FiLM.
  - 논문 claim: 큰 LLM/news 성능 향상이 아니라 small positive /
    conditional calibration.
  - 추천 제목:
    `Context-Conditioned FiLM for Bitcoin Direction Prediction from Price Charts`.
- [x] `5-14`: FinBERT+F&G gamma/beta relaxation ablation
  - 목적: 최종 bounded FiLM 설정 `gamma=0.02`, `beta=0.02`가 임의 선택이
    아니라, 더 강한 gamma 또는 beta 조절이 안정적 개선을 만들지 못했는지
    확인하는 마지막 mechanism ablation.
  - 고정 조건: frozen Stage2 `I60/R20/ohlc_ma_vb` CNN/classifier,
    FinBERT+F&G context, seeds `42,43,44,45,46`.
  - 비교: 5-9E conservative bounded FiLM vs `gamma_relaxed_g0p10_b0p02`
    vs `beta_relaxed_g0p02_b0p10`.
  - 결과: relaxed variants는 ROC-AUC/Brier는 개선하지만 accuracy를
    `0.580569`에서 `0.569466`으로 낮춤.
  - Thesis 반영 위치: 4.3의 bounded FiLM 제한식 근거, 6.6의 robustness
    discussion.

### Stage4

- [x] `4-N14B2-B6`: N16 derivatives/leverage context 조건부 해석
  - high funding / low funding
  - high CFTC OI / low CFTC OI
  - high leverage-regime proxy
  - correction/regression comparison

### Part 6 Interpretation / Robustness

- [x] Part 6 claim boundary 정리
  - 큰 성능 향상 / SOTA claim 금지
  - small conditional calibration claim으로 고정
  - Grad-CAM/gamma-beta는 causal proof가 아니라 diagnostic evidence로 사용
- [x] FinBERT+F&G 조건부 해석 근거 연결
  - Stage2 uncertain bucket, greed bucket, high-news-count bucket에서 positive
    conditional evidence 존재
  - correction `95`, regression `86`, net `+9`
- [x] Generic text representation negative control 반영
  - TF-IDF/SVD와 dense embedding 결과를 Table 8에 포함
  - FinBERT를 선택한 이유를 sentiment-aligned feature 관점으로 설명
- [x] N16 derivatives/leverage bucket table 보강
  - N14-B2-B6 로컬 후처리 분석 완료
  - strongest positive bucket: Stage2 uncertain + high funding,
    delta accuracy `+0.039604`, corrections `24`, regressions `12`,
    net `+12`
  - high funding 20d bucket: delta accuracy `+0.008304`, net `+12`
  - 해석: global outperformance가 아니라 weak bullish / uncertain visual
    prediction에 대한 conservative bearish calibration evidence
- [x] final figure selection
  - FinBERT+F&G Grad-CAM panel 1개
  - visual baseline, context summary, correction/regression figure 포함
- [ ] optional gamma/beta channel-level audit
  - context-feature vs gamma/beta correlation
  - correction vs regression modulation 비교

## 5. First Draft Completion Criteria

- [x] thesis draft v0 시작: skeleton을 본문 paragraph로 확장
- [x] 모든 chapter skeleton 완성
- [x] Stage2 four-image baseline과 N15/N16 image-variant 결과를 Results에 연결
- [x] Stage1-Stage5 실험 목적과 결과를 각 chapter에 연결
- [ ] 모든 성능 claim에 CSV/report 경로 연결
- [ ] 모든 figure/table에 source artifact 연결
- [x] 실패한 실험도 limitation/contribution으로 정리
- [x] formatting rule과 충돌하는 부분 없음
  - 2026-06-10 최종 LaTeX/PDF 검수 기준: A4, MIPT margins, Times New Roman,
    14pt main text, 1.5 spacing, title page no visible page number,
    page numbering from page 2, required document order, table/figure numbering 확인
- [ ] supervisor에게 보낼 1-page summary 초안 작성
