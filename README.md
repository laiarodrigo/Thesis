# Portuguese Variant Identification and Translation

Code, configurations, results, and reports for a thesis on identifying and
translating European and Brazilian Portuguese.

## Repository structure

- `configs/` — training and data-building configurations
- `data/` — small versioned inputs and artifact manifests
- `docs/` — reproducibility and artifact-release notes
- `report/` — the dissertation and summary paper LaTeX projects
- `results/` — compact result tables, statistics, and model manifests
- `scripts/` — data preparation, training, evaluation, and analysis entry points
- `src/` — reusable preprocessing and alignment modules

Large datasets, model weights, databases, and generated evaluation artifacts are
not tracked in Git. See `docs/artifact_release.md` for the artifact inventory and
release process.

## Environment

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Install the PyTorch build appropriate for the target CUDA environment when the
default package is not compatible.

## Reports

The complete LaTeX projects are in:

- `report/Rodrigo_Laia_MEIC_Thesis/` — dissertation
- `report/Rodrigo_Laia_Resumo_Tese/` — summary paper

For example, build the dissertation with:

```bash
cd report/Rodrigo_Laia_MEIC_Thesis
make pdf
```

See `report/README.md` for the report entry points and reference PDFs.
