# Stage 1: Structured error checks

Evaluate one candidate sentence as a conservative rewrite from Brazilian Portuguese (BP) into standard European Portuguese (EP).

Identify every genuine error in the EP candidate, but do not assign a quality score in this stage.

Before producing the JSON, examine the candidate in this exact order:

1. **Meaning preservation:** check for omissions, additions, changed facts, entities, actions, negation, time expressions, and changes in discourse function.
2. **Required BP→EP adaptations:** distinguish correct necessary adaptations from BP-only constructions that remain unacceptable in standard EP. Do not penalize a correct adaptation or an unchanged expression that is acceptable in both varieties.
3. **Unnecessary rewriting:** check whether wording, grammar, sentence structure, or register that was already acceptable in EP was changed without need.
4. **EP grammaticality, naturalness, and register:** check agreement, syntax, punctuation, idiomaticity, contextual fit, and register.

Perform these checks internally. Do not output check results or category labels. Output only the final error annotations.

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
