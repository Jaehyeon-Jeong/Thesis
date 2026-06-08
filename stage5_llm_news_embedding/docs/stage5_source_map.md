# Stage 5 Source Map

This file records the methodological sources for Stage 5. Exact page references
should be filled in when the thesis draft cites them.

## LLM Embedding Representation

- Chen, Kelly, and Xiu, *Expected Returns and Large Language Models*.
- Use in Stage 5:
  - Treat pretrained language models as text representation extractors.
  - Convert financial news text into numerical vectors.
  - Use the vectors as downstream prediction features.
- Stage 5 difference:
  - The news representation is not used as a standalone return predictor.
  - It is injected into a chart CNN through FiLM conditioning.

## OpenAI Embedding API

- OpenAI vector embeddings guide:
  `https://platform.openai.com/docs/guides/embeddings`
- OpenAI model page for `text-embedding-3-small`:
  `https://platform.openai.com/docs/models/text-embedding-3-small`
- Use in Stage 5:
  - First executable backend is `text-embedding-3-small`.
  - API output vector is cached per deterministic news input text.
  - API key is provided through `OPENAI_API_KEY`.
  - Stage 5 does not use a chat model for the main embedding feature.

## Prompt-Based News Classification

- Lopez-Lira and Tang, *Can ChatGPT Forecast Stock Price Movements?*
- Use in Stage 5:
  - Prompt an LLM to classify financial news as positive, negative, or unknown.
  - Cache the reason and confidence for interpretability.
- Stage 5 difference:
  - Prompt labels are auxiliary explanation features.
  - The main feature is the embedding representation.

## FiLM Conditioning

- Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*.
- Use in Stage 5:
  - Context vector produces `gamma` and `beta`.
  - FiLM modulates CNN feature maps.
- Stage 5 adaptation:
  - Context is financial news embedding rather than language-question embedding.
  - Visual backbone is the Stage 2 BTC chart CNN.

## BTC Chart Baseline

- Stage 2 local results.
- Main baseline:
  - `I60/R20/ohlc_ma_vb`
  - accuracy mean `0.5793`
  - ROC-AUC mean `0.5849`
