# Task

Evaluate one candidate sentence as a conservative rewrite from Brazilian Portuguese (BP) into standard European Portuguese (EP).

The goal is to preserve the source meaning and wording as closely as possible while making the changes necessary for grammatical, natural European Portuguese.

A change is not automatically correct because it sounds more typical of EP. An unchanged expression is not automatically incorrect because it comes from BP.

## Evaluation criteria

Assess the candidate according to these four criteria:

1. **Meaning preservation:** Does the candidate preserve the source meaning, information, and discourse function? Identify omissions, additions, and meaning changes.

2. **Required variety adaptation:** Does the candidate retain any BP constructions that are clearly unacceptable in standard EP? Identify required adaptations that were not made. Do not penalize BP expressions that are also acceptable in EP.

3. **Conservative rewriting:** Has the candidate unnecessarily changed words, grammar, sentence structure, or register that were already acceptable in EP? A grammatically valid EP expression can still be an error if it represents an unjustified change. Do not penalize necessary adaptations merely because they differ from the source.

4. **EP grammaticality and naturalness:** Is the candidate grammatical, natural, and appropriate in its passage context? Accept different valid EP constructions when they preserve the source and satisfy the conservative rewriting objective.

Use the BP passage as context and the current BP sentence as the source of meaning. Evaluate only the EP candidate.

Portuguese allows omitted subjects, and BP `você` has no single mandatory EP replacement. Accept natural, meaning-preserving constructions that fit the context and register.

## Error annotation

Report each independent error using the shortest exact substring of the EP candidate that identifies the problem.

Use these severity labels:

- `Minor`: A localized problem that does not substantially compromise the rewrite, including small unnecessary substitutions, minor grammatical or punctuation issues, or limited stylistic changes.
- `Major`: A problem that materially compromises the task, including meaning errors, significant omissions, necessary variety adaptations left undone, serious grammatical or contextual errors, or substantial unnecessary rewriting.

Group dependent changes within the same grammatical construction rather than marking multiple overlapping errors.

If content is missing and no existing candidate text can identify the omission, use `[MISSING]`.

Do not use `[MISSING]` when the meaning is preserved through a different grammatical construction.

All error substrings must come from the candidate, never from the BP source.

## Quality score

Assign a whole-candidate quality score from 0 to 100 using these broad bands:

- 85–100: Very good
- 65–80: Good
- 45–60: Acceptable
- 25–40: Borderline
- 0–20: Not acceptable

Use intermediate scores when appropriate.

Assign 100 if and only if no errors are identified.

Scores below 100 must reflect the overall impact and severity of the errors. Major errors should not normally receive very-good scores.

Do not use a mechanical points-per-error formula.

## Output

Return exactly one JSON object with `errors` and `score`.

Each error contains:
- `text`: the exact erroneous candidate substring, or `[MISSING]`.
- `severity`: `Minor` or `Major`.

Do not generate character offsets. They are calculated separately.

If an error substring occurs multiple times in the candidate, identify which occurrence is erroneous using the optional `occurrence` field, a one-based index counting identical substring occurrences from the beginning of the candidate.

For unique substrings, omit `occurrence`.

For missing content, use `[MISSING]` without `occurrence`.

Use an empty `errors` array and score 100 when no errors are identified.

Return only valid JSON. Do not provide reasoning, explanations, Markdown, or additional text.
