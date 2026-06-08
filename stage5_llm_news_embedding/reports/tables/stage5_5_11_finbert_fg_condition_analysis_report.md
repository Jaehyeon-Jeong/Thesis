# 5-11 FinBERT+F&G Conditional Correction Analysis

## Purpose

This analysis checks where the Stage5 FinBERT+F&G bounded FiLM model
corrects or regresses relative to the frozen Stage2 `I60/R20/ohlc_ma_vb`
visual baseline. It is designed for the thesis interpretability chapter,
not as a new training run.

## Overall Transition Summary

- Matched decisions: `7205`
- Stage2 accuracy: `0.579320`
- Stage5 accuracy: `0.580569`
- Delta accuracy: `0.001249`
- Corrections: `95`
- Regressions: `86`
- Net corrections: `9`
- Changed prediction rate: `0.025121`
- Mean probability-up delta: `-0.016158`

Interpretation: the model changes only a small subset of decisions.
The net correction is positive, but the effect is modest and should be
presented as conditional correction evidence rather than a large accuracy
gain.

## Transition-Level Context Means

| transition   |    n |   fg_value_mean |   fg_mean_60_mean |   finbert_news_count_20d_mean |   finbert_sentiment_mean_20d_mean |   finbert_news_fg_proxy_20d_mean |
|:-------------|-----:|----------------:|------------------:|------------------------------:|----------------------------------:|---------------------------------:|
| both_correct | 4088 |       50.073141 |         50.159834 |                    232.870841 |                          0.062236 |                        74.227319 |
| both_wrong   | 2936 |       47.723774 |         48.347939 |                    232.246594 |                          0.049059 |                        71.492761 |
| correction   |   95 |       56.357895 |         54.034561 |                    256.821053 |                          0.098143 |                        82.947674 |
| regression   |   86 |       50.406977 |         48.660078 |                    240.581395 |                          0.079707 |                        75.724258 |

## Positive Buckets

| bucket_family                                         | bucket     |    n |   delta_acc |   corrections |   regressions |   net_corrections |   net_correction_rate |   mean_prob_up_delta |
|:------------------------------------------------------|:-----------|-----:|------------:|--------------:|--------------:|------------------:|----------------------:|---------------------:|
| stage2_uncertain_45_55                                | yes        |  801 |    0.012484 |            94 |            84 |                10 |              0.012484 |            -0.022330 |
| fg_regime                                             | greed      | 2120 |    0.010849 |            46 |            23 |                23 |              0.010849 |            -0.017625 |
| finbert_negative_ratio_20d_q20_80                     | low_20pct  | 1445 |    0.008997 |            35 |            22 |                13 |              0.008997 |            -0.020605 |
| fg_mean_60_q20_80                                     | high_20pct | 1455 |    0.008247 |            26 |            14 |                12 |              0.008247 |            -0.017840 |
| finbert_news_count_7d_q20_80                          | high_20pct | 1465 |    0.007509 |            33 |            22 |                11 |              0.007509 |            -0.022129 |
| finbert_news_fg_proxy_60d_q20_80                      | high_20pct | 1445 |    0.006228 |            28 |            19 |                 9 |              0.006228 |            -0.020954 |
| finbert_sentiment_mean_60d_q20_80                     | high_20pct | 1445 |    0.006228 |            28 |            19 |                 9 |              0.006228 |            -0.020954 |
| stage2_uncertain_40_60                                | yes        | 1573 |    0.005722 |            95 |            86 |                 9 |              0.005722 |            -0.021740 |
| finbert_confidence_weighted_sentiment_mean_20d_q20_80 | high_20pct | 1445 |    0.004844 |            32 |            25 |                 7 |              0.004844 |            -0.020525 |
| finbert_confidence_weighted_sentiment_mean_60d_q20_80 | high_20pct | 1445 |    0.004844 |            27 |            20 |                 7 |              0.004844 |            -0.020877 |
| finbert_news_fg_proxy_20d_q20_80                      | high_20pct | 1445 |    0.004844 |            32 |            25 |                 7 |              0.004844 |            -0.020530 |
| finbert_sentiment_mean_20d_q20_80                     | high_20pct | 1445 |    0.004844 |            32 |            25 |                 7 |              0.004844 |            -0.020530 |

## Negative Buckets

| bucket_family                                         | bucket       |    n |   delta_acc |   corrections |   regressions |   net_corrections |   net_correction_rate |   mean_prob_up_delta |
|:------------------------------------------------------|:-------------|-----:|------------:|--------------:|--------------:|------------------:|----------------------:|---------------------:|
| finbert_news_count_60d_q20_80                         | low_20pct    | 1450 |   -0.008276 |            12 |            24 |               -12 |             -0.008276 |            -0.013112 |
| fg_regime                                             | neutral      | 1185 |   -0.006751 |             7 |            15 |                -8 |             -0.006751 |            -0.014549 |
| finbert_news_count_20d_q20_80                         | low_20pct    | 1460 |   -0.006164 |            12 |            21 |                -9 |             -0.006164 |            -0.011743 |
| finbert_news_count_7d_q20_80                          | low_20pct    | 1530 |   -0.005229 |            10 |            18 |                -8 |             -0.005229 |            -0.011893 |
| finbert_confidence_weighted_sentiment_mean_20d_q20_80 | low_20pct    | 1445 |   -0.004844 |            10 |            17 |                -7 |             -0.004844 |            -0.013629 |
| finbert_news_fg_proxy_20d_q20_80                      | low_20pct    | 1445 |   -0.004844 |            10 |            17 |                -7 |             -0.004844 |            -0.013573 |
| finbert_sentiment_mean_20d_q20_80                     | low_20pct    | 1445 |   -0.004844 |            10 |            17 |                -7 |             -0.004844 |            -0.013573 |
| finbert_sentiment_mean_7d_q20_80                      | low_20pct    | 1445 |   -0.004152 |            10 |            16 |                -6 |             -0.004152 |            -0.014562 |
| fg_regime                                             | extreme_fear | 1450 |   -0.004138 |            14 |            20 |                -6 |             -0.004138 |            -0.015148 |
| fg_value_q20_80                                       | low_20pct    | 1450 |   -0.004138 |            14 |            20 |                -6 |             -0.004138 |            -0.015148 |
| fg_delta_60_q20_80                                    | high_20pct   | 1505 |   -0.003322 |            19 |            24 |                -5 |             -0.003322 |            -0.017439 |
| finbert_news_fg_proxy_60d_q20_80                      | low_20pct    | 1445 |   -0.002076 |            12 |            15 |                -3 |             -0.002076 |            -0.013371 |

## Selected Samples For 5-12

Selected sample rows for targeted Grad-CAM/modulation export: `60`.
Use the selected-samples CSV as the input list for Stage5 `5-12`.

## Written Outputs

- `merged_decisions`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_11_finbert_fg_condition_analysis_merged_decisions.csv`
- `bucket_summary`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_11_finbert_fg_condition_analysis_bucket_summary.csv`
- `transition_context_summary`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_11_finbert_fg_condition_analysis_transition_context_summary.csv`
- `selected_samples_for_5_12`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_11_finbert_fg_condition_analysis_selected_samples_for_5_12.csv`
- `report`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_11_finbert_fg_condition_analysis_report.md`
- `manifest`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_11_finbert_fg_condition_analysis_manifest.json`

## Thesis Use

Use this result to support a cautious claim: FinBERT+F&G context
adds a small, conservative correction layer to the visual baseline.
It should not be described as a large performance improvement.
