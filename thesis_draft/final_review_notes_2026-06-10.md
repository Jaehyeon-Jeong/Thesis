# Final Thesis Review Notes — 2026-06-10

## Scope

Reviewed the current LaTeX/PDF thesis draft from the title page to the appendices.
The review covered MIPT formatting requirements recorded in
`VKR_format_requirements.md`, thesis logic, table/figure numbering, visible layout,
major typo patterns, and unnecessary/internal wording.

## Files Checked

- `Thesis_Jaehyeon_Jeong.tex`
- `Thesis_Jaehyeon_Jeong.pdf`
- `thesis_draft.log`
- `VKR_format_requirements.md`
- `checklist.md`

## Confirmed Formatting Status

- PDF page count: 46 pages.
- A4 page size confirmed by PDF media box.
- Main LaTeX settings use Times New Roman, 14pt base class, 1.5 spacing,
  left/right/top/bottom margins of 30/15/20/20 mm.
- Title page counts as page 1 but has no visible page number.
- Visible page numbering starts from page 2 at bottom center.
- Required document order is present:
  title page, abstract, contents, abbreviations, main text, references, appendices.
- Abstract is on its own page.
- Contents are separated from the abstract.
- List of abbreviations is formatted as a table.
- Table numbering starts from Table 1 after excluding the front-matter abbreviation table.
- Figure numbering starts from Figure 1.
- References are in a compact GOST-like numbered format and ordered by first use.
- Appendices start on separate pages.
- Build log has no LaTeX errors, undefined references, citation warnings, or overfull boxes.

## Main Fixes Applied During Review

- Corrected title-page field from `Major of study :` to `Field of study:`.
- Shortened the abstract so body plus keywords remain below 1500 characters.
- Changed abstract and contents spacing to 1.5.
- Fixed table numbering so the first main-text table is `Table 1`.
- Updated manual table references after the table-number correction.
- Removed internal/weak wording:
  - replaced `Stage 4 experiments` with `early context experiments`;
  - replaced informal `useless` wording with academic phrasing;
  - replaced `state-of-the-art forecasting system` with `dominant forecasting system`.
- Removed the near-empty orphan page after Section 4.6 by fitting the final line on the previous page.
- Updated `Makefile` to run XeLaTeX three times so the table of contents stays synchronized.

## Logic Review

The thesis now follows a defensible sequence:

1. Transfer chart-image prediction from the stock Re-image idea to BTC.
2. Screen BTC chart-only settings before selecting the main context target.
3. Establish `I60/R20/ohlc_ma_vb` as the strongest robust chart-only baseline.
4. Explain why chart-derived context is mostly redundant with a strong visual image.
5. Separate external context into sentiment, news, derivatives/leverage, and macro stress groups.
6. Justify frozen bounded FiLM as a baseline-preserving protocol.
7. Report that global gains are small but selected external context provides calibration/correction evidence.
8. Use correction/regression counts, conditional buckets, Grad-CAM, and gamma/beta summaries for interpretability.
9. Conclude conservatively without claiming large forecasting dominance.

## Remaining External Confirmation Items

These are not issues in the draft itself, but they require supervisor/department confirmation:

- Final title-page wording/template from MIPT personal cabinet or department.
- Whether English `Figure/Table` caption wording is accepted or should be Russian-style.
- Manual confirmation of the OCR-missing page 12 of the MIPT VKR regulation PDF.
- Whether a separate AI-use declaration is required by the department.

## Residual Notes

- The thesis is currently a LaTeX/PDF draft, not the final DOCX/personal-cabinet title-page version.
- Large raw datasets, checkpoints, API keys, and raw news text should not be placed in the public repository.
- The current claim strength is intentionally conservative: context-FiLM provides small calibration and selected conditional corrections, not a large universal performance improvement.
## Red-Flag Follow-up — 2026-06-10

- Verified that the gamma-relaxed and beta-relaxed Stage5 prediction exports have different file/checkpoint hashes but identical numeric prediction columns for every seed. They are therefore not treated as independent empirical rows in the thesis. The two rows were removed from the main news/sentiment result table, and the text was reframed around bounded FiLM and stronger-modulation diagnostics.
- Added the final BTC split dates to the empirical design: train 2018-04-29 to 2020-12-10, validation 2018-04-30 to 2020-12-11, test 2021-01-01 to 2024-12-11; label end dates for the 20-day test target run to 2024-12-31.
- Corrected the split description: the final test boundary is chronological and post-2021, but train/validation are randomly separated inside the pre-2021 development pool. Appendix Protocol B1 now uses `chronological post-2021 test; random train/validation inside pre-2021 pool` instead of a simplified `chronological` label. The limitations section now notes that overlapping 60-day images and 20-day labels can make validation optimistic for checkpoint selection, so thesis conclusions are framed around the held-out post-2021 test set.
- Added sign-test caveats for correction/regression counts: FinBERT+F\&G p≈0.552; derivatives all-test p≈0.195; uncertain-chart/high-funding p≈0.065 two-sided. Added the overlapping 20-day label caveat.
- Replaced `Part 6` with `Section 6` and named the OpenAI `text-embedding-3-small` model in the main text.
- Fixed `Makefile` to build `Thesis_Jaehyeon_Jeong.tex` / `Thesis_Jaehyeon_Jeong.pdf`.
