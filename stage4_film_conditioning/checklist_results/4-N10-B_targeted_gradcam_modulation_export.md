# 4-N10-B Targeted Grad-CAM + Gamma/Beta Modulation Export

Status: executed on Kaggle and reviewed locally.

Purpose:
- Use the N10-A selected sample list instead of default high-confidence samples.
- Compare the same `sample_index` under:
  - Stage 2 visual baseline.
  - N10 news-context bounded last-block FiLM.
- Focus on the two thesis-relevant transitions:
  - `Stage2 wrong -> N10 correct`.
  - `Stage2 correct -> N10 wrong`.

Design:
- Target class is the true label for both Stage 2 and N10 exports.
- Stage 2 export writes targeted Grad-CAM only.
- N10 export writes targeted Grad-CAM plus:
  - normalized news context values,
  - context embedding summary,
  - FiLM gamma/delta-gamma/beta summary,
  - full modulation JSON with top/bottom channel values.

Prepared artifacts:
- `notebooks/kaggle_stage4_n10_b_targeted_gradcam_modulation_one_cell.md`
- `scripts/generate_stage4_gradcam_context.py` targeted mode.
- Stage 2 `scripts/generate_stage2_gradcam.py` targeted mode.

Expected Kaggle output:
- `/kaggle/working/stage4_n10b_targeted_gradcam_modulation_bundle.zip`

Reviewed local bundle:

```text
/Users/jaehyeonjeong/Desktop/논문/N10_b_results
```

Result summary:

| Item | Value |
|---|---:|
| Compared samples | `7205` |
| Seeds | `42, 43, 44, 45, 46` |
| `Stage2 wrong -> N10 correct` | `27` |
| `Stage2 correct -> N10 wrong` | `24` |
| Net correction | `+3 / 7205` |
| Targeted Grad-CAM samples exported | `44` |

Interpretation:

```text
N10-B confirms that the N10 news-FiLM model mostly acts as a very small
Down-side/risk-off correction on top of the frozen Stage 2 chart baseline.
It fixes some Stage 2 Up false positives, but it also creates similar Up-label
regressions. Therefore this is useful as an interpretability artifact, not as
a strong performance-gain claim.
```

Observed probability behavior:

```text
Stage2 wrong -> N10 correct:
  prob_up_stage2 mean ~= 0.5047
  prob_up_n10 mean    ~= 0.4963

Stage2 correct -> N10 wrong:
  prob_up_stage2 mean ~= 0.5026
  prob_up_n10 mean    ~= 0.4964
```

Observed modulation magnitude:

```text
block4_gamma_mean ~= 1.00009
block4_delta_gamma range ~= -0.0012 to +0.0013
block4_beta range        ~= -0.0012 to +0.0013
```

This matches the N8/N9 design intent: bounded last-block FiLM should preserve
the Stage 2 CNN and apply only a small context-conditioned correction.

How to read the output:
- For correction samples, check whether N10 lowers/raises class evidence in a
  way that matches the true label and whether Grad-CAM shifts toward meaningful
  chart regions.
- For regression samples, check whether the same modulation suppresses useful
  Stage 2 evidence.
- If correction/regression behavior is concentrated around uncertain Stage 2
  probabilities, the next justified follow-up is uncertainty-gated FiLM.

Current next-step read:

```text
The useful cases are near the decision boundary, not high-confidence cases.
If continuing architecture work, use uncertainty-gated FiLM so context
correction is stronger only when the frozen Stage 2 baseline is uncertain.
```
