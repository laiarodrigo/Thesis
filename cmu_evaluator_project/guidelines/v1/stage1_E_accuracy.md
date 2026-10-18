# Stage 1A: Accuracy and variety error identification

Evaluate one candidate sentence as a conservative rewrite from Brazilian Portuguese (BP) into standard European Portuguese (EP).

In this independent pass, identify only errors involving meaning preservation or required BP→EP adaptation. Do not assign a quality score.

Check for:

- omitted or added information;
- changed facts, entities, actions, negation, time expressions, or discourse function;
- BP constructions that remain clearly unacceptable in standard EP;
- incorrect BP→EP substitutions.

Do not penalize a correct necessary BP→EP adaptation or an unchanged expression that is acceptable in both varieties. Leave purely grammatical, naturalness, register, and unnecessary-rewriting issues to the other pass.

Use the BP passage as context and the current BP sentence as the source of meaning. Evaluate only the EP candidate.

Report each independent error using the shortest exact substring of the candidate that identifies the problem. Group dependent changes within one construction rather than producing overlapping errors.

Use `Minor` for a localized problem that does not substantially compromise the rewrite and `Major` for a problem that materially compromises meaning or the required variety adaptation.

If source content is missing and no candidate substring can identify the omission, use `[MISSING]`. Do not use `[MISSING]` when another construction preserves the meaning.

Return exactly one JSON object:

```json
{"errors":[{"text":"exact candidate substring","severity":"Major"}]}
```

If an error substring occurs more than once, add a one-based `occurrence`. Omit it for unique substrings and `[MISSING]`. Use an empty `errors` array when this pass finds no errors. Return only valid JSON with no score, reasoning, Markdown, or additional text.
