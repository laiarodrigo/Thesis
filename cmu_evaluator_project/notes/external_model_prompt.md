# Fixed Gemma 4 prompt for TED2020 pilot 01

Model: `google/gemma-4-E4B-it`

The runner sends one user message per source sentence. It does not provide the
surrounding TED passage. The message is the following fixed instruction,
followed by `Source sentence:` and the exact frozen BP sentence.

```text
You are given one sentence in Brazilian Portuguese.

Rewrite it in standard European Portuguese while preserving the original
meaning, wording, tone, and structure as much as possible.

Make only changes that are necessary for an acceptable European Portuguese
version.

Do not paraphrase, summarize, expand, explain, or stylistically improve the
sentence.

If the original sentence is already acceptable in European Portuguese,
return it unchanged.

Return only the rewritten sentence and no explanation.

Source sentence:
{source_bp}
```

The placeholder is replaced with exactly one frozen source sentence. The
official Gemma 4 processor renders this user message with its chat template,
`add_generation_prompt=True`, and `enable_thinking=False`. The exact message
and rendered prompt are retained in every output row.
