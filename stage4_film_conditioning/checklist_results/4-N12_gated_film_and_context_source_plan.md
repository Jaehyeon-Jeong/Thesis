# 4-N12 Gated FiLM And Context-Source Plan

Status: N12-A completed, N12-B completed, N12-C completed, N12-D completed for existing context sources.

## Why N12

N10-B showed that the N10 news-FiLM model mostly applies a small Down/risk-off
correction near the Stage 2 decision boundary:

```text
Stage2 wrong -> N10 correct: 27
Stage2 correct -> N10 wrong: 24
Net correction: +3 / 7205
```

This means the next experiment should not randomly increase FiLM scale. The
model should use the interpretation: context correction should depend on the
frozen Stage 2 baseline confidence.

## N12-A. Uncertainty-Gated News FiLM

Purpose:

```text
Let context-FiLM intervene more when the Stage 2 chart model is uncertain.
```

Candidate gate:

```text
uncertainty = 4 * prob_up_stage2 * (1 - prob_up_stage2)
```

Candidate FiLM:

```text
gamma = 1 + uncertainty * scale * tanh(raw_gamma)
beta  =     uncertainty * scale * tanh(raw_beta)
```

Initial setup:

```text
image: I60/R20/ohlc_ma_vb
backbone: Stage 2 CNN frozen
classifier: Stage 2 classifier frozen
context: news SVD32 + news-count features
scale candidates: 0.02, 0.05
seeds: 42, 43, 44, 45, 46
```

## N12-B. Confidence-Gated News FiLM

Purpose:

```text
Test whether context can strengthen high-confidence Stage 2 visual evidence.
```

Candidate gate:

```text
confidence = abs(2 * prob_up_stage2 - 1)
```

Candidate FiLM:

```text
gamma = 1 + confidence * scale * tanh(raw_gamma)
beta  =     confidence * scale * tanh(raw_beta)
```

Risk:

```text
If Stage 2 is confidently wrong, confidence gating may strengthen the wrong
class. Evaluate correction/regression counts, not accuracy alone.
```

## N12-C. Technical-Only Frozen FiLM Ablation

Purpose:

```text
Separate image-derived technical context from image-external context.
```

Candidate features:

```text
bb_percent_b_60
bb_bandwidth_60
mfi_60
rv_60
```

Expected interpretation:

```text
The gain may be small because ohlc_ma_vb already contains MA/volume/shape
information, but this ablation makes the Stage 4 context-source comparison
cleaner.
```

Runner:

```text
notebooks/kaggle_stage4_n12c_technical_only_pretrained_frozen_bounded_film_one_cell.md
```

Default grid:

```text
method: film_full_bounded_last_block
scales: 0.02, 0.05
seeds: 42, 43, 44, 45, 46
```

Result:

```text
best accuracy setting: scale 0.02
accuracy mean: 0.579736
ROC-AUC mean: 0.584778
interpretation: effectively tied with Stage 2; technical context is mostly
redundant with ohlc_ma_vb visual information
```

## N12-D. Context-Source Comparison

Purpose:

```text
Stop adding one-off variants and decide which context source is defensible
under the same frozen Stage 2 FiLM protocol.
```

Compare under the same frozen Stage 2 bounded-FiLM protocol:

```text
F&G-only
news-only
technical-only
news + F&G
```

Metrics:

- accuracy,
- ROC-AUC,
- Brier score,
- F1,
- predicted-Up rate,
- correction count,
- regression count,
- net correction.

Interpretation:

```text
technical-only mostly tests redundant chart-derived context;
F&G/news test image-external crypto sentiment/context;
news + F&G tests whether verbose headlines and compact regime score complement
each other.
```

N12-D produced the context-source table before moving to macro/RORO.

Result:

```text
best compact accuracy candidate: F&G-only bounded FiLM, scale 0.02
accuracy mean: 0.580291
Stage 2 baseline accuracy mean: 0.579320

best ROC-AUC/calibration signal: news SVD8, scale 0.05
ROC-AUC mean: 0.586619
tradeoff: lower accuracy/F1 and lower predicted-Up rate

technical-only context: effectively tied with Stage 2
interpretation: BB/MFI/RV-like information is mostly redundant with ohlc_ma_vb
```

Review:
[4-N12-D context-source comparison](4-N12-D_context_source_comparison.md)

Tables:

- `reports/tables/stage4_n12d_context_source_comparison.csv`
- `reports/tables/stage4_n12d_context_source_comparison_compact.csv`
- `reports/tables/stage4_n12d_context_source_recommendation.csv`

## Result Upload Policy

For every N12 substep:

- Commit and push the updated checklist before running Kaggle if code changes.
- After Kaggle, save the downloaded local result bundle path in the matching
  checklist result file.
- Commit only compact tables, JSON summaries, result notes, and runnable
  notebooks.
- Do not commit large output bundles, checkpoints, or repeated full prediction
  CSVs unless they are intentionally small and needed for a table.
- If a result table already exists for the same experiment, update the existing
  note instead of creating a duplicate result document.

## N13. Macro/RORO Context Extension

Run after N12-D because the completed context-source comparison shows that
chart-derived technical context is mostly redundant with `ohlc_ma_vb`, while
image-external context remains the more defensible direction.

Purpose:

```text
Test whether official financial stress and macro risk-on/risk-off regime can
condition the frozen BTC chart model better than BTC technical indicators that
are already visible in the chart.
```

Important terminology:

```text
OFR FSI is not a direct RORO index.
It is an official financial-stress index and will be used as a risk-off proxy:
high OFR FSI = higher financial stress = risk-off context.

The RORO proxy is separate.
It is KC Fed-inspired, but built only from public FRED/Cboe data.
Do not claim full replication of the KC Fed RORO index.
```

### N13-0. Source Audit And Terminology Lock

Tasks:

```text
verify official links
verify date coverage over 2018-2024
verify direct CSV load path
document missing-date/as-of policy
document feature interpretation and leakage rule
```

Primary sources:

```text
OFR FSI:
https://www.financialresearch.gov/financial-stress-index/data/fsi.csv

Cboe VIX:
https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv

FRED/FRED CSV:
VIXCLS, SP500, NASDAQCOM, DTWEXBGS, DGS10, BAMLH0A0HYM2
```

### N13-1. OFR FSI Feature Builder

Feature interpretation:

```text
ofr_fsi high -> financial stress high -> risk-off proxy
do not hard-code BTC down
let FiLM learn whether and how this context modulates BTC chart features
```

Candidate features:

```text
ofr_fsi_value
ofr_fsi_mean_20
ofr_fsi_mean_60
ofr_fsi_delta_20
ofr_fsi_delta_60
ofr_fsi_std_60
ofr_credit
ofr_equity_valuation
ofr_funding
ofr_safe_assets
ofr_volatility
```

### N13-2. FSI-Only Frozen Bounded FiLM

Protocol:

```text
image: I60/R20/ohlc_ma_vb
seeds: 42,43,44,45,46
Stage 2 CNN: loaded and frozen
Stage 2 classifier: frozen
context source: OFR FSI features only
method: bounded last-block FiLM first
scale: 0.02 first
```

Required metrics:

```text
accuracy
ROC-AUC
Brier score
F1
predicted-Up rate
correction count
regression count
net correction
seed-level collapse check
```

### N13-3. KC Fed-Inspired Public-Data RORO Proxy Builder

Candidate raw components:

```text
VIX level/change
S&P500 20-day return
NASDAQ 20-day return
Broad Dollar Index return/change
US 10Y yield change
High-yield OAS change
gold return, optional
BTC-equity rolling correlation, optional diagnostic
```

Direction rule:

```text
positive component value should mean risk-off pressure
VIX up -> positive
HY OAS up -> positive
equity return down -> positive
dollar strength up -> positive, depending on final audit
10Y move -> keep as raw component first; avoid overclaiming direction
```

Synthetic score:

```text
train-fit PCA first component on risk-off-aligned public features
then fix the sign so higher roro_proxy means more risk-off
store both roro_proxy and raw components for interpretation
```

### N13-4. RORO-Proxy-Only Frozen Bounded FiLM

```text
same protocol as N13-2
context source: public-data RORO proxy features only
goal: compare synthetic risk-regime proxy against official OFR FSI
```

### N13-5. Macro Context-Source Comparison

Compare:

```text
Stage 2 frozen visual baseline
F&G-only bounded FiLM
news-only bounded/gated FiLM
technical-only bounded FiLM
FSI-only bounded FiLM
RORO-proxy-only bounded FiLM
optional: FSI + F&G, RORO + F&G
```

Selection rule:

```text
Choose a final Stage 4 macro candidate only if it improves accuracy or
ROC/Brier without predicted-class collapse.
If metric gains are tiny, keep it as interpretability/context evidence rather
than a performance claim.
```

### N13-6. Macro Interpretability Export

Target samples:

```text
Stage 2 wrong -> N13 correct
Stage 2 correct -> N13 wrong
high-FSI/risk-off dates
low-FSI/risk-on or calm dates
```

Export:

```text
targeted Grad-CAM
FSI/RORO raw and normalized values
gamma/beta summaries
prob_up change from Stage 2 to N13
correction/regression labels
```

## N13-B. Sentiment/Event Feature Extension

Run only if the TF-IDF/SVD headline representation remains too weak or too hard
to interpret after N12-D/N13 planning.

Candidate features:

```text
headline sentiment score
positive / negative / neutral counts
crypto regulation / exchange / ETF / macro event tags
```

Leakage rule:

```text
Only headlines available by strict t-1 may be used.
Record encoder/model/version/date/cache hash.
```

## N14. Final Stage 4 Interpretability Report

Purpose:

```text
Turn the selected Stage 4 model into thesis-ready evidence.
```

Required sections:

- Stage 2 baseline vs selected context-FiLM metrics;
- correction/regression table;
- predicted-Up distribution;
- targeted Stage 2 vs Stage 4 Grad-CAM;
- gamma/beta/modulation-gate summaries;
- representative `Stage2 wrong -> Stage4 correct` and
  `Stage2 correct -> Stage4 wrong` samples.
