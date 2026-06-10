# Source and License Register

This is a working register for thesis writing. Final license/terms must be verified before submission and before any GitHub release.

## Papers

| Source | Role in thesis | Local artifact | License/status |
|---|---|---|---|
| Re-image price trends paper | Stage1 baseline and chart-image methodology | `자료조사/Xiu-Re-Imagining-Price-Trends.pdf` | Citation needed; redistribution of PDF not needed |
| `lich99/Stock_CNN` GitHub repository | Initial public PyTorch implementation reference for the Re-image chart-CNN pipeline | `https://github.com/lich99/Stock_CNN` | Cite repository; verify license before redistributing copied source code |
| FiLM paper | Feature-wise modulation method | `자료조사/FiLM.pdf` | Citation needed; code inspiration should be cited separately if used |
| `ethanjperez/film` GitHub repository | Public FiLM implementation reference for feature-wise affine modulation | `https://github.com/ethanjperez/film` | Cite repository; do not paste source code into thesis appendix unless license terms are checked |
| Grad-CAM paper | Visual explanation method | `자료조사/Grad-CAM.pdf` | Citation needed |
| LLM/expected returns paper | Motivation for LLM-derived text representations | `/Users/jaehyeonjeong/Desktop/ExpectedReturns_LLMs.pdf` | Citation needed |
| ChatGPT/financial headline reaction paper | Motivation for prompt-based financial news analysis | To verify | Citation needed |

## Data Sources

| Source | Used for | Redistribution policy in thesis/repo |
|---|---|---|
| BTC OHLCV dataset / Binance-derived data | Stage2-Stage5 BTC chart images and labels | Do not upload raw data to GitHub; cite dataset/source |
| Fear & Greed index | Market sentiment/regime context | Cite source; store only derived aligned features where allowed |
| `edaschau/bitcoin_news` / BTC headline data | News headline embedding and FinBERT features | Do not redistribute raw text in public thesis repo; derived features only |
| OFR FSI | Macro stress context | Cite OFR source; derived features only |
| CFTC/CME Bitcoin futures data | Derivatives/leverage context | Cite CFTC source; derived features only |
| Public market data such as VIX/SP500/DXY | RORO proxy experiments | Cite data source; derived features only |

## Models and APIs

| Model/API | Used for | Notes |
|---|---|---|
| Stage2 CNN implementation | Visual baseline | Local code in Stage2, adapted from the Re-image/`Stock_CNN` structure and extended for BTC experiments | Cite Re-image paper and `lich99/Stock_CNN`; do not include copied third-party source snippets in thesis appendix unless license is confirmed |
| FiLM implementation reference | Modulation design | Cite FiLM paper and `ethanjperez/film`; local code adapts the FiLM idea to BTC chart features and numerical context |
| OpenAI `text-embedding-3-small` | Headline embedding table | API key must not be stored; cache derived vectors with model/version metadata |
| ProsusAI/FinBERT | Headline-level financial sentiment | Cite Hugging Face model card and underlying FinBERT paper/model source |
| Kaggle | Execution/data environment | Record dataset versions and notebook outputs; do not commit large outputs |

## GitHub / Public Repo Exclusions

- Raw news text.
- API keys and `.env` files.
- Large checkpoints: `.pt`, `.pth`.
- Large prediction CSV files.
- Raw market data files when license/terms are unclear.
- Third-party PDFs.
