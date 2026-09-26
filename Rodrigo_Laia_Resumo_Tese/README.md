# Dissertation summary paper

This directory contains the complete LaTeX source project for the shorter
paper derived from the dissertation.

## Main files

- `sample-acmtog.tex` contains the article.
- `sample-base.bib` contains the bibliography.
- `acmart.cls`, the bibliography styles, and the data-model file are local
  copies required by the ACM template.

## Build

Install a TeX distribution with `latexmk`, then run:

```bash
make pdf
```

The generated PDF is written to `build/sample-acmtog.pdf`. Use `make watch`
for automatic rebuilding or `make clean` to remove generated files.

The root-level `sample-acmtog.pdf` is the deliberately committed reference
build. Other LaTeX intermediates are ignored.
