TARGET=em_etal

all:
	Rscript -e "library(knitr); knit('${TARGET}.Rnw')"
	latexmk -bibtex -pdf ${TARGET}.tex


.PHONY: clean
clean:
	latexmk -f -c ${TARGET}.tex
	rm -f ${TARGET}.tex

.PHONY: deepclean
deepclean:
	latexmk -f -c ${TARGET}.tex
	rm -rf cache figure ${TARGET}.tex ${TARGET}.pdf
