# Thesis candidate models for TED2020 pilot 01

## Corrected recommendation

Use the final dissertation's 4B **decoder-controlled PT/BR systems**, not the
older `<br-pt>` comparison models. The two candidates below have the same base
model, LoRA rank, Stage A initialization, control placement, and decoding
protocol. Their principal difference is the Stage B data condition.

### Model A — conservative GPT-Wikipedia condition

- **Run/checkpoint name:** `B4-D-G-PTBR`
- **Base model:** `google/t5gemma-2-4b-4b`
- **LoRA adapter path:** `outputs/encoder_decoder/control_strings_ptbr/t5gemma2_4b_stageB_gpt_wiki_decoder_ptbr`
- **Training config:** `configs/encoder_decoder/t5gemma2_4b/control_strings_ptbr/stageB_gpt_wiki_decoder_ptbr.yaml`
- **Stage A adapter:** `outputs/encoder_decoder/control_strings_ptbr/t5gemma2_4b_stageA_opensubs_decoder_ptbr`
- **LoRA:** rank 48, alpha 96, dropout 0.05
- **Control:** decoder-controlled, base-vocabulary strings `BR` and `PT`
- **Stage:** Stage B GPT-Wikipedia after Stage A OpenSubtitles
- **Reason:** this is the final dissertation's strongest supervised conservative
  BP-to-EP condition. In the final aggregate ledger, beam-4 rewriting reaches
  BLEU 87.6803 / TER 0.0895 on Golden and 41.0657 / 0.4633 on FRMT. In the
  dissertation's directional BP-to-EP comparison, its prediction matches or
  beats copying on 74.0% of Golden sentences by sentence BLEU and 74.4% by TER.

### Model B — broader GPT-Wikipedia + FRMT condition

- **Run/checkpoint name:** `B4-D-GF-PTBR`
- **Base model:** `google/t5gemma-2-4b-4b`
- **LoRA adapter path:** `outputs/encoder_decoder/control_strings_ptbr/t5gemma2_4b_stageB_gpt_wiki_frmt_decoder_ptbr`
- **Training config:** `configs/encoder_decoder/t5gemma2_4b/control_strings_ptbr/stageB_gpt_wiki_frmt_decoder_ptbr.yaml`
- **Stage A adapter:** `outputs/encoder_decoder/control_strings_ptbr/t5gemma2_4b_stageA_opensubs_decoder_ptbr`
- **LoRA:** rank 48, alpha 96, dropout 0.05
- **Control:** decoder-controlled, base-vocabulary strings `BR` and `PT`
- **Stage:** Stage B GPT-Wikipedia + FRMT after Stage A OpenSubtitles
- **Reason:** this matched system is substantially broader. In the final
  aggregate ledger, beam-4 rewriting reaches BLEU 68.1052 / TER 0.2368 on
  Golden and 44.5668 / 0.4322 on FRMT. In the directional BP-to-EP comparison,
  it matches or beats copying on only 31.0% of Golden sentences by sentence
  BLEU and 31.2% by TER. It therefore provides the intended contrast without
  also changing rank, control placement, or Stage A lineage.

The no-equal-translation GF variant is not recommended. Its FRMT beam-4 result
is only marginally higher, while it introduces an additional training-policy
difference and is appreciably worse on Golden. The standard pair is the cleaner
pilot comparison.

## Exact BP-to-EP representation

The dissertation defines `PT` and `BR` as **source-variety labels**. For both
selected decoder-controlled models, one BP-to-EP example is represented as:

```text
encoder input:  {source_bp}
decoder output: BR {European Portuguese rewrite}
```

There is no encoder prefix and no prose prompt. In particular, the input is
not `<br-pt> {source_bp}`. The generated leading `BR` is retained in
`raw_model_output`, checked as the expected source-variety label, and removed
only when producing `cleaned_candidate`.

Each TED sentence is passed independently. The ten-sentence passage is never
given to either generator.

## Loading and thesis decoding behavior

Reuse `scripts/encoder_decoder/eval/evaluate_encdec.py`:

- tokenizer lookup: explicit tokenizer path, adapter directory, then base model;
- `AutoModelForSeq2SeqLM.from_pretrained(..., trust_remote_code=True)`;
- `PeftModel.from_pretrained(base_model, adapter_dir)`;
- bfloat16 on CUDA and float32 on CPU, followed by evaluation mode;
- maximum source length 512 tokens;
- deterministic generation (`do_sample=False`);
- beam size 4, length penalty 0.8, and early stopping;
- adaptive output cap `max(8, min(384, int(source_tokens * 1.15 + 6)))`;
- `max_new_tokens=256` as the non-adaptive ceiling setting;
- no-repeat n-gram size 4, repetition penalty 1.2, and forced EOS;
- decoder label removal with `strip_decoder_label_prefix`, without overwriting
  the raw decoded output.

These are the beam-4 settings used in the final PT/BR evaluation wrapper and
the settings behind the beam-4 results used for model selection. Greedy results
also exist in the report, but they are not the selected pilot default.

## Artifact audit — generation blocker

The configs and final dissertation/report ledger exist locally, but the two
adapter directories and their final evaluation directories do not exist in
this checkout as of 2026-10-06. The report records both runs as complete and
synced to Ceph. The exact adapters must be restored at the paths above (or their
restored paths supplied explicitly) before generation. Do not substitute an
older `comparison_staged` adapter.

## Future commands (do not run until the adapters are restored)

```bash
python cmu_evaluator_project/scripts/run_thesis_candidate.py \
  --source cmu_evaluator_project/data/prepared/pilot_01/source/source.json \
  --system-id B4-D-G-PTBR \
  --adapter-dir outputs/encoder_decoder/control_strings_ptbr/t5gemma2_4b_stageB_gpt_wiki_decoder_ptbr \
  --training-config configs/encoder_decoder/t5gemma2_4b/control_strings_ptbr/stageB_gpt_wiki_decoder_ptbr.yaml \
  --output cmu_evaluator_project/data/prepared/pilot_01/candidates/thesis_model_a.json
```

For Model B, use `B4-D-GF-PTBR` and its adapter/config paths above. The wrapper
processes one sentence at a time and preserves both raw and cleaned output.
