$pdf_mode = 5;  # XeLaTeX
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 2;

# Keep output next to thesis_draft.tex so Preview/Skim auto-refreshes the same PDF.
$aux_dir = '.';
$out_dir = '.';

# macOS PDF previewer.
$pdf_previewer = 'open %S';

# Extra cleanup extensions.
push @generated_exts, 'synctex.gz', 'fls', 'fdb_latexmk', 'nav', 'snm';
