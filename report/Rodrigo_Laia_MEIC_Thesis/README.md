# Local thesis workflow

This directory contains the Overleaf project exported into the repository.

## Edit here

- `main.tex` wires the full document together.
- `Chapters/Thesis-Chapter_*.tex` contains the main thesis body.
- `Chapters/EN-Abstract.tex`, `Chapters/PT-Resumo.tex`, and related files hold
  the abstract and keywords.
- `Front_Cover.tex` contains title, author, supervisors, and committee data.
- `Preamble_commands.tex` contains package configuration and language settings.
- `Bibliography.bib` contains references.

## Build locally

From this directory:

```bash
make pdf
```

For automatic rebuilds while writing:

```bash
make watch
```

To remove generated files:

```bash
make clean
```

The build output is written to `build/` so the source tree stays clean.
