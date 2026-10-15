# Conservative BP→EP Rewriting Annotation Guidelines

Your task is to evaluate each candidate sentence as a conservative rewrite from Brazilian Portuguese into standard European Portuguese. A good rewrite should preserve the source meaning and context, make the BP→EP changes that are actually needed, avoid unnecessary rewriting, and be natural European Portuguese. For each candidate, mark the smallest span corresponding to each error and assign it Minor or Major severity. If something from the BP source is missing from the candidate and there is no candidate span to highlight, mark [MISSING] at the end of the sentence and assign it a severity. Finally, give the whole candidate a 0–100 quality score.

## Task

You will evaluate candidate rewrites from Brazilian Portuguese (BP) into
standard European Portuguese (EP).

The goal is conservative rewriting. The candidate should make the changes
needed for an acceptable European Portuguese version while preserving the
source sentence as closely as possible.

A good candidate should:

1. preserve the meaning of the source;
2. be acceptable standard European Portuguese in the same context and
   register;
3. make BP→EP changes that are actually required;
4. avoid unnecessary rewriting or paraphrasing;
5. be grammatical and natural;
6. remain consistent with the surrounding passage.

## Marking errors

Mark errors only in the candidate rewrite, not in the Brazilian Portuguese
source.

Highlight the smallest span that clearly identifies each problem.

If there are several independent problems, mark them separately.

Use `[MISSING]` only when source meaning or discourse function is absent from the candidate. If source words disappear because the same meaning is expressed by another grammatical construction, do not use `[MISSING]`; evaluate the new construction itself. Dependent grammatical changes that belong to the same construction should be judged together, not as separate errors.

Each marked error must be assigned one of two severity levels: Minor or Major.

## Minor errors

Use **Minor** for a localized problem when the candidate remains essentially
a valid conservative European Portuguese rewrite.

Examples include:

- a small grammatical or fluency problem;
- a spelling or punctuation problem;
- a small unnecessary wording change with limited effect on the sentence.

Severity depends on the impact of the problem, not on the number of words in
the highlighted span.

An unnecessary punctuation change is normally Minor when it is localized and does not materially affect meaning or structure. It is Major only when its impact materially affects the interpretation, structure, or success of the rewrite.

## Major errors

Use **Major** when a problem materially causes the candidate to fail the
conservative BP→EP rewriting task.

Examples include:

- source meaning that is changed, omitted, or unsupported content that is
  added;
- a clearly required BP→EP adaptation that has been left undone;
- substantial unnecessary rewriting or paraphrasing;
- a form of address or register that conflicts with the established context;
- a serious grammatical or contextual problem.

A one-word error can be Major if its impact is substantial.

Not every unnecessary change is Major. A small unnecessary substitution can
be Minor when its impact is limited.

A clearly Brazilian Portuguese form that requires adaptation for the intended standard European Portuguese output is a Major error.

Do not mark a form as an error merely because another form is more frequent or preferred in European Portuguese. A change is required only when the source form is clearly inappropriate for standard European Portuguese in context.

## Conservative rewriting

Copying the source can be fully correct when the Brazilian Portuguese sentence
is already acceptable in standard European Portuguese in context.

There is not necessarily one unique correct European Portuguese rewrite. Do
not penalize a candidate simply because another valid wording is possible.

## Using passage context

Use the surrounding passage only to interpret the current sentence and to assess consistency. Judge only the current candidate; do not require it to express information that appears only in neighbouring sentences.

Relevant contextual information includes:

- referents;
- tense;
- terminology;
- discourse relations;
- forms of address;
- register;
- consistency across sentences.

Do not invent information that is not supported by the source or its context.

## Second-person `você` and subject omission

Do not apply a mandatory string replacement to BP `você`. European Portuguese
allows different second-person realizations according to context, register,
politeness, and the speaker/addressee relationship. For example, BP `você não
encontra ...` may naturally correspond to:

- `não encontra ...`, with an omitted subject and third-person/formal or
  respectful treatment;
- `tu não encontras ...`; or
- `não encontras ...`, with an omitted informal singular `tu`.

Portuguese is pro-drop. Omitting the explicit source pronoun while preserving
the same addressee is not missing content, so do not use `[MISSING]` merely
because `você` disappeared. In particular, `você não encontra` → `não
encontra` can be valid, and `você não encontra` → `não encontras` can be valid
when an informal `tu` interpretation is contextually appropriate. The mixed
form `você não encontras` is not the intended example. If `tu` is explicit,
it requires second-person agreement: `tu não encontras`, not `tu não
encontra`.

Do not mark an output as erroneous solely because explicit `você` remains when
the complete sentence is otherwise natural and contextually acceptable in EP.
Mark the chosen realization only when it is clearly inappropriate or unnatural
for the passage context/register, or when it causes a grammatical or
consistency problem. Passage context may establish address/register
consistency, but do not force neighbouring content or infer an unexpressed
social relationship.

Accept all natural, meaning-preserving EP realizations. This task evaluates
acceptability, not agreement with one canonical reference. A clearly unnatural
or inconsistent realization is usually Minor when the meaning remains clear;
use Major only when it materially changes meaning, the addressee, or the
communicative relationship.

## Overall score

After marking errors, assign the candidate an overall score from 0 to 100.

If two systems produce exactly the same candidate for the same source and context, assign the same error spans, severities, and overall score.

Use the following broad scale:

**85–100 — Very good**

The candidate is a successful or nearly successful conservative BP→EP
rewrite. Meaning is preserved, required adaptation is handled well, the
output is acceptable European Portuguese, and unnecessary rewriting is absent
or negligible.

**65–80 — Good**

The candidate is largely successful but contains one notable problem or
several smaller problems. It remains mostly suitable as a conservative
European Portuguese rewrite.

**45–60 — Acceptable**

The main meaning remains recoverable, but there are substantial problems with
European Portuguese adaptation, meaning preservation, conservatism, fluency,
or contextual consistency.

**25–40 — Borderline**

The candidate has frequent or serious problems and would require significant
revision before it could be accepted as a conservative European Portuguese
rewrite.

**0–20 — Not acceptable**

The candidate substantially fails the task because of serious meaning
distortion, missing or unsupported content, inappropriate target-variety
language, excessive rewriting, severe grammatical problems, or serious
contextual inconsistency.

If you mark no errors, assign a score of **100**.

Do not use a mechanical points-per-error formula. The overall score should
reflect the quality of the candidate as a whole.

## Uncertainty and comments

Use the comment field when you encounter a genuinely uncertain or unusual
case.

In particular, add a short comment if you are unsure whether a Brazilian
Portuguese form actually requires adaptation into European Portuguese.

Do not invent a rule when the guidelines do not resolve the case.

## Methodological basis

This protocol adapts ESA/cESA and MQM-style impact-based severity. Minor and
Major are determined by their effect on the intended conservative rewriting
task rather than by linguistic error category.
