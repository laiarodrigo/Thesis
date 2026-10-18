# Stage 2: Holistic quality score

Assign a holistic 0–100 quality score to one candidate rewrite from Brazilian Portuguese (BP) into standard European Portuguese (EP).

You receive the original BP passage context, current BP source sentence, EP candidate, and the strictly validated Stage 1 candidate-side error annotations. Treat those annotations as fixed evidence: do not add, remove, merge, reinterpret, or reclassify errors.

Score the whole candidate according to meaning preservation, required BP→EP adaptation, conservative rewriting, and EP grammaticality, naturalness, context, and register.

Use these broad bands:

- 85–100: Very good
- 65–80: Good
- 45–60: Acceptable
- 25–40: Borderline
- 0–20: Not acceptable

Use intermediate scores when appropriate. Do not use a mechanical points-per-error formula. Major errors should not normally receive very-good scores.

Assign 100 if and only if the validated `errors` array is empty. If the array contains any error, the score must be below 100.

Return exactly one JSON object:

```json
{"score":75}
```

Return only valid JSON with no errors list, reasoning, Markdown, or additional text.
