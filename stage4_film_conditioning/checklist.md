# Stage 4 Checklist

## English

Proceed one item at a time. Stage 4 must keep the Stage 2 BTC pipeline fixed
and add FiLM conditioning in clearly separated condition-source tracks.

Planning phase:
- [x] 4-0. Stage 4 folder, checklist, and workflow scaffold
  - Result: [4-0 Stage 4 scaffold](checklist_results/4-0_stage4_scaffold.md)
- [ ] 4-1. FiLM paper, local summary, and `ethanjperez/film` reference review
- [ ] 4-2. Stage 2/Stage 3 dependency and baseline-output review
- [ ] 4-3. FiLM insertion-point design for Stock_CNN blocks
- [ ] 4-4. Condition-source track design
  - 4A: FiLM-only control
  - 4B: F&G index + FiLM
  - 4C: News dataset + FiLM with non-LLM encoder
  - 4D: News dataset -> LLM + FiLM
- [ ] 4-5. Gamma-only FiLM vs Full FiLM comparison plan
- [ ] 4-6. Grad-CAM and gamma/beta export plan
- [ ] 4-7. Kaggle runner and output backup plan

Implementation phase:
- [ ] 4-I0. Implementation readiness review
- [ ] 4-I1. Shared Stage 4 config/code scaffold
- [ ] 4-I2. FiLM layer module
- [ ] 4-I3. FiLM generator module for the current condition track
- [ ] 4-I4. StockCNN + FiLM model variants for I5/I20/I60
- [ ] 4-I5. BTC FiLM runner using fixed Stage 2 data pipeline
- [ ] 4-I6. Prediction, classification metric, and trading metric export
- [ ] 4-I7. Grad-CAM and FiLM gamma/beta export
- [ ] 4-I8. Local or small Kaggle smoke test
- [ ] 4-I9. Kaggle single-config FiLM run
- [ ] 4-I10. Kaggle grid runner for selected FiLM variants
- [ ] 4-I11. Stage 4 result report

Deferred condition tracks:
- [ ] 4-FG1. F&G index source audit
- [ ] 4-FG2. F&G date alignment and leakage check
- [ ] 4-N1. BTC news dataset audit
- [ ] 4-N2. News-to-trading-date alignment and leakage check
- [ ] 4-N3. Non-LLM news encoder implementation
- [ ] 4-L1. LLM encoder source/model decision
- [ ] 4-L2. LLM embedding/cache/reproducibility plan
- [ ] 4-L3. LLM-conditioned FiLM implementation

Important:
- Today, do not implement News/LLM conditioning.
- FiLM-only control is useful for verifying insertion, logging, Grad-CAM, and
  gamma/beta export. It is not evidence that external information helps.
- Any external condition source must pass date alignment and no-future-leakage
  checks before model training.
- Grad-CAM remains required. FiLM gamma/beta must be saved with the same
  sample/date/layer keys used by Grad-CAM.

## 한국어

한 항목씩 진행합니다. Stage 4는 Stage 2 BTC pipeline을 고정하고, condition source를
명확히 나눈 상태에서 FiLM conditioning을 추가해야 합니다.

계획 단계:
- [x] 4-0. Stage 4 폴더, checklist, workflow scaffold
  - 결과: [4-0 Stage 4 scaffold](checklist_results/4-0_stage4_scaffold.md)
- [ ] 4-1. FiLM 논문, local summary, `ethanjperez/film` reference 확인
- [ ] 4-2. Stage 2/Stage 3 dependency와 baseline output 확인
- [ ] 4-3. Stock_CNN block 기준 FiLM 삽입 위치 설계
- [ ] 4-4. Condition-source track 설계
  - 4A: FiLM-only control
  - 4B: F&G index + FiLM
  - 4C: News dataset + non-LLM encoder + FiLM
  - 4D: News dataset -> LLM + FiLM
- [ ] 4-5. Gamma-only FiLM vs Full FiLM 비교 계획
- [ ] 4-6. Grad-CAM과 gamma/beta export 계획
- [ ] 4-7. Kaggle runner와 output backup 계획

구현 단계:
- [ ] 4-I0. 구현 readiness review
- [ ] 4-I1. Stage 4 공통 config/code scaffold
- [ ] 4-I2. FiLM layer module
- [ ] 4-I3. 현재 condition track용 FiLM generator module
- [ ] 4-I4. I5/I20/I60용 StockCNN + FiLM model variant
- [ ] 4-I5. 고정된 Stage 2 data pipeline을 쓰는 BTC FiLM runner
- [ ] 4-I6. prediction, classification metric, trading metric export
- [ ] 4-I7. Grad-CAM과 FiLM gamma/beta export
- [ ] 4-I8. local 또는 작은 Kaggle smoke test
- [ ] 4-I9. Kaggle single-config FiLM run
- [ ] 4-I10. 선택된 FiLM variant용 Kaggle grid runner
- [ ] 4-I11. Stage 4 결과 보고

나중으로 미루는 condition track:
- [ ] 4-FG1. F&G index source audit
- [ ] 4-FG2. F&G date alignment와 leakage check
- [ ] 4-N1. BTC news dataset audit
- [ ] 4-N2. News-to-trading-date alignment와 leakage check
- [ ] 4-N3. Non-LLM news encoder 구현
- [ ] 4-L1. LLM encoder source/model 결정
- [ ] 4-L2. LLM embedding/cache/reproducibility 계획
- [ ] 4-L3. LLM-conditioned FiLM 구현

중요:
- 오늘은 News/LLM conditioning을 구현하지 않습니다.
- FiLM-only control은 삽입 위치, logging, Grad-CAM, gamma/beta export를 확인하는
  용도입니다. 외부 정보가 도움이 된다는 증거로 해석하지 않습니다.
- 외부 condition source는 model training 전에 date alignment와 no-future-leakage
  check를 통과해야 합니다.
- Stage 4에서도 Grad-CAM은 필수입니다. FiLM gamma/beta는 Grad-CAM과 같은
  sample/date/layer key로 저장합니다.
