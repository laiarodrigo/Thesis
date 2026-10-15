# Task

Your task is to evaluate each candidate sentence as a conservative rewrite from Brazilian Portuguese into standard European Portuguese. A good rewrite should preserve the source meaning and context, make the BP→EP changes that are actually needed, avoid unnecessary rewriting, and be natural European Portuguese. For each candidate, mark the smallest span corresponding to each error and assign it Minor or Major severity. If something from the BP source is missing from the candidate and there is no candidate span to highlight, mark [MISSING] at the end of the sentence and assign it a severity. Finally, give the whole candidate a 0–100 quality score.

Evaluate each candidate as a conservative rewrite from Brazilian Portuguese
into standard European Portuguese.

A good rewrite should:

1. preserve the source meaning;
2. be acceptable European Portuguese in the same context and register;
3. make BP→EP changes that are actually needed;
4. avoid unnecessary rewriting or paraphrasing;
5. be grammatical and natural;
6. remain consistent with the surrounding passage.

# Marking errors

Highlight the smallest candidate span that identifies each error.

Mark separate errors separately.

Use **Minor** for a localized problem when the candidate is still essentially
a valid conservative European Portuguese rewrite.

Use **Major** when the problem materially fails the task, such as:

- changed, missing, or unsupported meaning;
- a clearly required BP→EP change left undone;
- a clearly inappropriate European Portuguese form;
- substantial unnecessary rewriting;
- a serious grammatical or contextual problem.

Use `[MISSING]` only when source meaning or discourse function is absent. If the same meaning is expressed through another grammatical construction, do not use `[MISSING]`; evaluate that construction instead.

Judge dependent changes that belong to the same grammatical construction together, rather than as separate errors.

An unnecessary punctuation change is normally Minor unless it materially affects meaning or structure.

# Important

Copying the source can be fully correct when no BP→EP change is needed.

Do not penalize a candidate merely because another valid European Portuguese
wording is possible.

Do not reward unnecessary stylistic improvement or paraphrasing.

A clearly Brazilian Portuguese form that requires adaptation is Major.

Do not mark a form as an error only because another form is more frequent or preferred in European Portuguese.

Use the surrounding passage only to interpret the current sentence and check consistency. Judge only the current candidate.

BP `você` has no single mandatory EP replacement. Depending on context and register, `você não encontra`, `não encontra`, `tu não encontras`, or `não encontras` may be acceptable. Portuguese is pro-drop: do not use `[MISSING]` merely because an explicit `você` is omitted while the same addressee is preserved. Do not penalize retained `você` solely for being explicit. Mark a form only when it is clearly unnatural, grammatically inconsistent, or inappropriate for the passage; explicit `tu` requires second-person agreement (`tu não encontras`, not `tu não encontra`). Accept all natural meaning-preserving realizations rather than enforcing one canonical reference. Use passage context only for supported address/register consistency; do not invent a social relationship. An unnatural or inconsistent choice is usually Minor when meaning remains clear and Major only when it materially changes meaning, the addressee, or their communicative relationship.

If two candidates are exactly identical for the same source and context, give them the same spans, severities, and score.

# Score

**85–100:** Very good  
**65–80:** Good  
**45–60:** Acceptable  
**25–40:** Borderline  
**0–20:** Not acceptable

If you mark no errors, give the candidate a score of **100**.

Use the comment field for genuinely uncertain or unusual cases.
