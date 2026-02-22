# T5Gemma2 4B Boilerplate

This folder holds launch helpers for the larger `google/t5gemma-2-4b-4b` track.

## Translation (LoRA)

Dry-run:

```bash
bash scripts/encoder_decoder/t5gemma2_4b/run_step4a_translation_boilerplate.sh
```

Execute:

```bash
bash scripts/encoder_decoder/t5gemma2_4b/run_step4a_translation_boilerplate.sh --execute
```

Config path:

`configs/encoder_decoder/t5gemma2_4b/translation_lora_boilerplate.yaml`

## Classification Head (Encoder + Head)

Dry-run:

```bash
bash scripts/encoder_decoder/t5gemma2_4b/run_step4b_classification_head_boilerplate.sh
```

Execute:

```bash
bash scripts/encoder_decoder/t5gemma2_4b/run_step4b_classification_head_boilerplate.sh --execute
```

Smoke-run first:

```bash
bash scripts/encoder_decoder/t5gemma2_4b/run_step4b_classification_head_boilerplate.sh --execute --smoke-run
```

Config path:

`configs/encoder_decoder/t5gemma2_4b/classification_head_boilerplate.yaml`
