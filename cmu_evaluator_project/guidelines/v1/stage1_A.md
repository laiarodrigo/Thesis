# Stage 1: Error identification

Evaluate one candidate sentence as a conservative rewrite from Brazilian Portuguese (BP) into standard European Portuguese (EP).

Identify every genuine error in the EP candidate, but do not assign a quality score in this stage.

Apply these criteria:

1. Meaning preservation: identify omissions, additions, changed facts, entities, actions, negation, time expressions, or discourse function.
2. Required variety adaptation: identify BP constructions that remain clearly unacceptable in standard EP. Do not penalize a BP expression that is also acceptable in EP, and do not penalize a correct necessary BP→EP adaptation.
3. Conservative rewriting: identify unnecessary changes to wording, grammar, structure, or register that were already acceptable in EP.
4. EP grammaticality and naturalness: identify grammatical, idiomatic, contextual, or register problems.

Use the BP passage as context and the current BP sentence as the source of meaning. Evaluate only the EP candidate.

Report each independent error using the shortest exact substring of the candidate that identifies the problem. Group dependent changes within one construction rather than producing overlapping errors.

Use these severity labels:

- `Minor`: a localized problem that does not substantially compromise the rewrite.
- `Major`: a problem that materially compromises the task, including a meaning error, significant omission, necessary adaptation left undone, serious grammatical or contextual error, or substantial unnecessary rewriting.

If source content is missing and no candidate substring can identify the omission, use `[MISSING]`. Do not use `[MISSING]` when a different grammatical construction preserves the meaning.

All ordinary error substrings must come from the candidate, never from the BP source.

Return exactly one JSON object with `errors`:

```json
{"errors":[{"text":"exact candidate substring","severity":"Minor"}]}
```

If an error substring occurs more than once, add `"occurrence": 1` using the one-based occurrence of the identical substring in the candidate. Omit `occurrence` for unique substrings and for `[MISSING]`.

Use an empty `errors` array when the candidate has no errors. Return only valid JSON with no score, reasoning, Markdown, or additional text.
