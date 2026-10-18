# Stage 1B: Fluency and conservatism error identification

Evaluate one candidate sentence as a conservative rewrite from Brazilian Portuguese (BP) into standard European Portuguese (EP).

In this independent pass, identify only errors involving EP grammaticality, naturalness, register, or unnecessary rewriting. Do not assign a quality score.

Check for:

- grammatical, syntactic, agreement, punctuation, or idiomaticity problems;
- unnatural wording or inappropriate register in passage context;
- unnecessary changes to source wording, grammar, sentence structure, or register that was already acceptable in EP.

Use the BP source to decide whether a candidate change was necessary. Do not penalize a correct required BP→EP adaptation. Leave meaning-preservation errors and retained BP-only constructions to the other pass unless they directly create a fluency or conservatism error covered here.

Report each independent error using the shortest exact substring of the candidate that identifies the problem. Group dependent changes within one construction rather than producing overlapping errors.

Use `Minor` for a localized problem that does not substantially compromise the rewrite and `Major` for a serious grammatical, contextual, register, or substantial unnecessary-rewriting problem.

All error substrings must come from the candidate. This pass must not use `[MISSING]` because omissions belong to the accuracy/variety pass.

Return exactly one JSON object:

```json
{"errors":[{"text":"exact candidate substring","severity":"Minor"}]}
```

If an error substring occurs more than once, add a one-based `occurrence`. Omit it for unique substrings. Use an empty `errors` array when this pass finds no errors. Return only valid JSON with no score, reasoning, Markdown, or additional text.
