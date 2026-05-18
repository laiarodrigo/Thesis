LATEXMK ?= latexmk
MAIN := main.tex
OUTDIR := build

.PHONY: pdf watch clean

pdf:
	$(LATEXMK) -pdf -file-line-error -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(MAIN)

watch:
	$(LATEXMK) -pvc -pdf -file-line-error -interaction=nonstopmode -halt-on-error -outdir=$(OUTDIR) $(MAIN)

clean:
	$(LATEXMK) -C -outdir=$(OUTDIR)
	rm -rf $(OUTDIR)
