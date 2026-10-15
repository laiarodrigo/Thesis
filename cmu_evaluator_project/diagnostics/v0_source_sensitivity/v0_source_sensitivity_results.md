# V0 source-sensitivity diagnostic

SLURM job: `13862`

This diagnostic uses 25 synthetic evaluations: five sentence groups crossed with five conditions. It contains no human gold annotations and is separate from the official V0 DEV artifacts.

## Run integrity

- Expected input IDs: 25
- Prediction records: 25
- Missing, duplicate, or unexpected IDs: 0
- Structurally valid judgments: 23
- Invalid judgments: 2
- Invalidity reason: `errors[0]_text_not_in_candidate` (2 records)
- Raw JSON-only responses: 1
- Responses accepted after permitted Markdown-wrapper normalization: 24
- Instructions SHA-256: `c9d22177d847f66fbe28c15bfad47bab810b342be7d47998852ae8ebd073a798` (25/25)
- Prompt/schema SHA-256: `71aa52e88c9e1c37ed3fd7efff9c960c7bf4224dce2e5a60a8ae6f5e418adeb2` (25/25)
- Model: `google/gemma-4-E4B-it`
- Requested revision: `ee0ef6023621cff504d758262d4e04895a5af4a2`
- Deterministic generation, sampling disabled, thinking disabled: 25/25

Scores marked **unvalidated** come from the raw JSON of an invalid prediction. They are shown for transparency but are not treated as structurally valid judgments.

## Results

### 1. Refrigerator

| Condition | BP source | EP candidate | Score | Difference from correct | Predicted errors | Valid |
|---|---|---|---:|---:|---|---|
| Correct | O menino abriu a geladeira. | O menino abriu o frigorífico. | 95 | 0 | `frigorífico` — Minor | Yes |
| Meaning change | O menino abriu a geladeira. | O menino fechou o frigorífico. | 30 | −65 | `fechou` — Major; `frigorífico` — Major | Yes |
| Major omission | O menino abriu a geladeira. | O menino abriu. | **30 (unvalidated)** | **−65 (unvalidated)** | Raw: `a geladeira` — Major; invalid because this text is absent from the candidate | **No** |
| Unrelated | O menino abriu a geladeira. | A menina leu um livro. | 20 | −75 | `A menina` — Major | Yes |
| Source conflict | O menino fechou a janela. | O menino abriu o frigorífico. | 10 | −85 | `abriu` — Major; `frigorífico` — Major | Yes |

### 2. Speaking to Pedro

| Condition | BP source | EP candidate | Score | Difference from correct | Predicted errors | Valid |
|---|---|---|---:|---:|---|---|
| Correct | Ela está falando com o Pedro. | Ela está a falar com o Pedro. | 95 | 0 | `está a falar` — Minor | Yes |
| Meaning change | Ela está falando com o Pedro. | Ela está a falar com a Maria. | 90 | −5 | `a falar` — Minor | Yes |
| Major omission | Ela está falando com o Pedro. | Ela está a falar. | **30 (unvalidated)** | **−65 (unvalidated)** | Raw: `com o Pedro` — Major; invalid because this text is absent from the candidate | **No** |
| Unrelated | Ela está falando com o Pedro. | Ela comprou uma casa nova. | 0 | −95 | `Ela comprou uma casa nova.` — Major | Yes |
| Source conflict | Ela está falando com a Maria. | Ela está a falar com o Pedro. | 60 | −35 | `o Pedro` — Major | Yes |

### 3. Mobile phone

| Condition | BP source | EP candidate | Score | Difference from correct | Predicted errors | Valid |
|---|---|---|---:|---:|---|---|
| Correct | Comprei um celular ontem. | Comprei um telemóvel ontem. | 95 | 0 | `telemóvel` — Minor | Yes |
| Meaning change | Comprei um celular ontem. | Comprei um computador ontem. | 50 | −45 | `computador` — Major | Yes |
| Major omission | Comprei um celular ontem. | Comprei um telemóvel. | 85 | −10 | `telemóvel` — Minor | Yes |
| Unrelated | Comprei um celular ontem. | A praia estava vazia. | 0 | −95 | `A praia estava vazia.` — Major | Yes |
| Source conflict | Vendi um celular hoje. | Comprei um telemóvel ontem. | 10 | −85 | `Comprei` — Major; `telemóvel` — Major; `ontem` — Major | Yes |

### 4. Bus

| Condition | BP source | EP candidate | Score | Difference from correct | Predicted errors | Valid |
|---|---|---|---:|---:|---|---|
| Correct | Ele pegou o ônibus cedo. | Ele apanhou o autocarro cedo. | 90 | 0 | `apanhou` — Minor; `autocarro` — Minor | Yes |
| Meaning change | Ele pegou o ônibus cedo. | Ele perdeu o autocarro cedo. | 65 | −25 | `perdeu` — Major; `autocarro` — Minor | Yes |
| Major omission | Ele pegou o ônibus cedo. | Ele apanhou o autocarro. | 85 | −5 | `apanhou` — Minor; `autocarro` — Minor | Yes |
| Unrelated | Ele pegou o ônibus cedo. | O jantar estava delicioso. | 0 | −90 | `O jantar estava delicioso.` — Major | Yes |
| Source conflict | Ele pegou o trem tarde. | Ele apanhou o autocarro cedo. | 10 | −80 | `apanhou` — Major; `autocarro` — Major; `cedo` — Major | Yes |

### 5. Train

| Condition | BP source | EP candidate | Score | Difference from correct | Predicted errors | Valid |
|---|---|---|---:|---:|---|---|
| Correct | Eu estou esperando o trem. | Eu estou à espera do comboio. | 90 | 0 | `à espera` — Minor | Yes |
| Meaning change | Eu estou esperando o trem. | Eu estou à espera do autocarro. | 85 | −5 | `à espera` — Minor | Yes |
| Major omission | Eu estou esperando o trem. | Eu estou à espera. | 30 | −60 | `à espera` — Major | Yes |
| Unrelated | Eu estou esperando o trem. | Eu fechei a porta da cozinha. | 0 | −90 | `fechei a porta da cozinha` — Major | Yes |
| Source conflict | Eu estou esperando o ônibus. | Eu estou à espera do comboio. | 85 | −5 | `à espera` — Minor | Yes |

## Diagnostic interpretation

These are behavioral checks, not formal gold metrics.

- **Correct candidates:** all five received high scores (90–95; mean 93), but none received 100. Gemma incorrectly marked necessary BP→EP adaptations such as `frigorífico`, `telemóvel`, `autocarro`, and `à espera` as Minor errors.
- **Meaning changes:** all five scored below their paired correct candidate. The differences were −65, −5, −45, −25, and −5 (mean −29). The Pedro→Maria and comboio→autocarro changes were barely penalized and were not identified as semantic errors, so sensitivity was inconsistent.
- **Major omissions:** all five raw scores were below the paired correct score, but two predictions were structurally invalid because Gemma copied the omitted BP-source text into the error span instead of using `[MISSING]`. Of the three valid judgments, two omissions retained scores of 85. Omission detection was therefore weak and inconsistent despite evidence that the model noticed the omitted content in the invalid cases.
- **Unrelated candidates:** all five were valid, all received a Major error, and scores were 0–20. Their mean difference from the correct condition was −89. This behavior was consistent.
- **Source conflicts:** the identical fixed candidates scored lower under every conflicting source. Differences were −85, −35, −85, −80, and −5 (mean −58). Four conflicts were substantially penalized, while the final ônibus→comboio conflict was almost missed.

Overall, the source-conflict and unrelated conditions show that V0 does consult the BP source. The diagnostic also exposes two distinct weaknesses: it often penalizes correct required BP→EP adaptations, and it can miss or underweight localized semantic differences and omissions.
