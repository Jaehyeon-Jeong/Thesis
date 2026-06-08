# 5-12 Stage5 FinBERT+F&G Targeted Grad-CAM and Modulation Export

## Purpose

Export Stage2 and Stage5 Grad-CAM figures for the selected 5-11
correction/regression samples, together with Stage5 gamma/beta
modulation summaries and aggregated FinBERT/F&G context values.

## Scope

- Stage2 baseline: `stage2_i60_ohlc_ma_vb_r20`
- Stage5 candidate: `stage4_film_full_bounded_last_block_i60_ohlc_ma_vb_r20_c60_stage5_finbert_fg_sentiment_v1_pretrained_frozen_s0p02`
- Context: `stage5_finbert_fg_context_i60_ohlc_ma_vb_r20_stage5_finbert_fg_sentiment_v1`
- Seeds: `[42, 43, 44, 45, 46]`
- Target class source: true `label`
- Selected limit per transition panel and seed: `4`
- Selected input rows before per-seed/panel limit: `60`

## Exported Artifacts

- `modulation_summary`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_modulation_summary.csv`
- `modulation_by_panel`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_modulation_by_panel.csv`
- `stage5_samples`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_stage5_samples.csv`
- `stage2_samples`: `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/tables/stage5_5_12_finbert_fg_targeted_gradcam_stage2_samples.csv`
- copied Grad-CAM/report artifacts: `30` files in `/Users/jaehyeonjeong/Desktop/논문/stage5_llm_news_embedding/reports/figures/gradcam`

## Interpretation Notes

- This export is post-hoc. It does not train or tune the model.
- Stage2 and Stage5 use the same selected samples and true-label Grad-CAM target.
- Stage5 modulation uses the 5-9E trained checkpoint with bounded FiLM scale `0.02`.
- Raw news text is not redistributed here. The exported context is the aggregated
  FinBERT/F&G numeric context already used by the model.

## Commands

- Total commands run: `10`
