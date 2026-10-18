# Stage 1: Error identification with explicit omissions

Evaluate one candidate sentence as a conservative rewrite from Brazilian Portuguese (BP) into standard European Portuguese (EP).

Identify every genuine error in the EP candidate, but do not assign a quality score in this stage.

Apply these criteria:

1. Meaning preservation: identify omissions, additions, changed facts, entities, actions, negation, time expressions, or discourse function.
2. Required variety adaptation: identify BP constructions that remain clearly unacceptable in standard EP. Do not penalize a BP expression that is also acceptable in EP, and do not penalize a correct necessary BP→EP adaptation.
3. Conservative rewriting: identify unnecessary changes to wording, grammar, structure, or register that were already acceptable in EP.
4. EP grammaticality and naturalness: identify grammatical, idiomatic, contextual, or register problems.

Use the BP passage as context and the current BP sentence as the source of meaning. Evaluate only the EP candidate.

Separate candidate-side errors from missing source content:

- In `errors`, report each ordinary error using the shortest exact substring of the candidate that identifies it.
- In `omissions`, report source content that is absent from the candidate using the shortest exact substring of the current BP source that identifies what is missing.
- Do not put source text in `errors` and do not use `[MISSING]` in either generated list.
- Do not report an omission when a different candidate construction preserves the source meaning.

Group dependent changes within one construction rather than producing overlapping errors.

Use these severity labels:

- `Minor`: a localized problem that does not substantially compromise the rewrite.
- `Major`: a problem that materially compromises the task, including a meaning error, significant omission, necessary adaptation left undone, serious grammatical or contextual error, or substantial unnecessary rewriting.

Return exactly one JSON object with `errors` and `omissions`:

```json
{
  "errors":[{"text":"exact candidate substring","severity":"Minor"}],
  "omissions":[{"source_text":"exact omitted BP source substring","severity":"Major"}]
}
```

If a reported candidate substring or source substring occurs more than once in its respective sentence, add `"occurrence": 1` using its one-based occurrence. Omit `occurrence` for unique substrings.

Use empty arrays when there are no errors or omissions. Return only valid JSON with no score, reasoning, Markdown, or additional text.
