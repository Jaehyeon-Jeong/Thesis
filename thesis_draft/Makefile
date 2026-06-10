TEX = Thesis_Jaehyeon_Jeong.tex
PDF = Thesis_Jaehyeon_Jeong.pdf

.PHONY: build watch open clean distclean

build:
	rm -f Thesis_Jaehyeon_Jeong.aux Thesis_Jaehyeon_Jeong.out Thesis_Jaehyeon_Jeong.toc Thesis_Jaehyeon_Jeong.fls Thesis_Jaehyeon_Jeong.fdb_latexmk Thesis_Jaehyeon_Jeong.synctex.gz Thesis_Jaehyeon_Jeong.xdv Thesis_Jaehyeon_Jeong.log
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)
	xelatex -interaction=nonstopmode -halt-on-error $(TEX)

watch:
	open $(PDF) || true
	latexmk -pvc -pdfxe -interaction=nonstopmode -synctex=1 $(TEX)

open:
	open $(PDF)

clean:
	latexmk -c $(TEX)

distclean:
	latexmk -C $(TEX)
