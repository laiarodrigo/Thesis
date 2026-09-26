# Final-Protocol Ablation Guardrails

These scripts and configs are for final-protocol ablations only. They should
not be mixed with recovered legacy runs.

## Directory Boundary

- Final-protocol configs: `configs/encoder_decoder/t5gemma2_4b/final/`
- Final-protocol data: `data/encoder_decoder/t5gemma2/final_protocol/`
- Final-protocol eval data: `data/encoder_decoder/t5gemma2/final_eval/`
- Final-protocol outputs: `outputs/encoder_decoder/final/`

Recovered legacy runs use different directories and different task formats:

- Recovered configs: `configs/encoder_decoder/t5gemma2_4b/comparison_staged/`
- Recovered data: `data/encoder_decoder/t5gemma2/compare_staged_v2/`
- Recovered outputs: `outputs/encoder_decoder/compare_staged/`

Do not compare a `final/` config with a `compare_staged/` data path as if it
were a one-variable ablation.

## Equal-Row Terms

`noequal` in recovered legacy configs is not the same experimental condition as
`noequal_cls` in the final-protocol ablations.

- Final `noequal_cls`: removes only equal-pair classification rows from the
  final unified training view. Translation rows, including equal translation
  rows, are preserved.
- Final regular unified data: includes equal-pair classification rows duplicated
  with `<pt-br>` and `<pt-pt>` labels. It does not use an `equal` class.
- Final classification evaluation: should use `classification_noequal_test.jsonl`
  so test metrics cover only `pt-br` and `pt-pt`.
- Recovered `noequal`: belongs to the old rendered data format and may also
  involve legacy control strings, `BR`/`PT` targets, different equal-row policy,
  different classification loss, and different batching.

## Current One-Variable Ablations

- `*_noequal_cls_final.yaml`: current final formulation, but equal-pair
  classification training rows removed.
- `*_unrestricted_cls_loss_final.yaml`: current final data and tokens, but
  without `model.restricted_classification_tokens`, so classification uses the
  ordinary full-vocabulary first-token loss.

The decoding ablation is evaluation-only and should be run on the same adapter
with greedy and beam-4 settings.
