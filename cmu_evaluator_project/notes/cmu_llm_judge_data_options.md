# CMU LLM-as-a-Judge: Data Construction Plan

Before constructing the dataset, I think we first need to define three things:
the task, the linguistic differences the data should cover, and which existing
resources can provide those examples. These decisions should guide the selection
of the actual examples.

## Stage 1 — Task Definition

Because the CMU visit is short, the project will focus only on **Brazilian
Portuguese → European Portuguese (BP→EP)**. The task is **conservative BP→EP
adaptation**.

A good adaptation should:

- preserve the original meaning;
- produce Portuguese acceptable in European Portuguese;
- make the BP→EP changes that are actually necessary;
- avoid unnecessary stylistic rewriting or paraphrasing;
- leave parts of the source unchanged when they do not need adaptation,
  including the whole sentence when appropriate;
- allow more than one valid EP realization when appropriate.

### Context

Some BP→EP decisions may depend on context: the appropriate treatment of a word
can depend on its meaning or use in the sentence or surrounding discourse.
It may be difficult to know in advance which sentences need additional context,
and only some of the resources we are considering may preserve usable
surrounding text.

**Proposed approach:** for samples from a resource that provides
reliable sentence-level surrounding context, retain the **previous sentence and
the next sentence**:

```text
previous sentence
[SOURCE SENTENCE TO ADAPT]
next sentence
```

Only the middle/source sentence is adapted and later annotated. The neighboring
sentences are provided only to help interpret it. This keeps the adaptation unit
short, which matters because we will later compare or annotate multiple
candidates for the same source.

## Stage 2 — Linguistic Coverage / Data Requirements

The next step is to identify the main linguistic differences between BP and EP
that the dataset should represent. The table below summarizes commonly discussed
and well-documented differences in the literature.

| Difference | Brazilian Portuguese | European Portuguese | Notes | Sources |
| --- | --- | --- | --- | --- |
| Orthography | *econômico*, *prêmio* | *económico*, *prémio* | Some spelling differences remain under the Orthographic Agreement. | [Alves (2025)][A25] |
| Orthography — older spellings | *ato*, *batismo* | Older spellings include *acto*, *baptismo*. | These older contrasts have been reduced by the Orthographic Agreement. | [Costa-jussà et al. (2018)][CJ18] |
| Progressive constructions | Usually *estar* + gerund: *Ele está correndo*. | Usually *estar a* + infinitive: *Ele está a correr*. | This describes a preference for expressing an ongoing action, not every use of the gerund. | [Costa-jussà et al. (2018)][CJ18]; [Alves (2025)][A25] |
| Clitic placement (short object pronouns) | Often before the verb: *Ele me viu*. | Often after the verb in affirmative clauses: *Ele viu-me*. | Placement also depends on sentence structure and register. | [Costa-jussà et al. (2018)][CJ18] |
| Subject-pronoun realization / pronoun drop | More frequent explicit subjects: *Eu fui ao supermercado*. | More frequent omitted subjects: *Fui ao supermercado*. | Both varieties permit omission; discourse affects the choice. | [Barreiro and Mota (2018)][BM18]; [Alves (2025)][A25] |
| Forms of address / pronoun choice | *Você* is common across formal and informal situations; *tu* also occurs regionally. | *Tu* is common in familiar address; formal address often uses third-person forms, sometimes without an explicit pronoun. | Choice depends strongly on region, register, and speaker relationship. | [Barreiro and Mota (2018)][BM18]; [Costa-jussà et al. (2018)][CJ18] |
| Determiners with possessives | The article is often omitted: *meu carro*. | The article is typically present: *o meu carro*. | The pattern depends on the noun-phrase construction. | [Barreiro and Mota (2018)][BM18]; [Alves (2025)][A25] |
| Lexical differences | *trem*, *terno*, *ônibus* | *comboio*, *fato*, *autocarro* | These pairs refer to a train, a suit, and a bus. Meaning and usage matter; a word is not necessarily exclusive to one variety. | [Barreiro and Mota (2018)][BM18]; [Alves (2025)][A25] |
| Multiword / phraseological differences | *todo o mundo* | *toda a gente* | Both express “everybody”; phraseological preferences do not always imply exclusive usage. | [Barreiro and Mota (2018)][BM18] |
| Null objects (leaving an understood object unspoken) | Object omission occurs in a wider range of constructions. | Also permits omitted objects, but with more restrictions in the constructions discussed by Cyrino. | Both allow examples such as *Joana viu [—] na TV ontem*. The referent and sentence structure matter; [—] marks omitted material. | [Cyrino (2001)][CY01], §1 |
| Verb-phrase ellipsis (leaving understood verbal material unspoken) | Allows omission after a gerund in examples such as *... e a Maria também está lendo [—]*. | Comparable omission after *... está a ler [—]* is restricted in Cyrino’s examples. | Both varieties allow ellipsis, but the verb construction affects what can be omitted. | [Cyrino (2001)][CY01], §2 |
| Shared Portuguese | Many expressions are also acceptable in EP. | Many expressions are also acceptable in BP. | Not every sentence contains a variety-specific difference. | [Barreiro and Mota (2018)][BM18] |

These are the main well-documented differences and shared uses currently
considered for coverage in the dataset.

## Stage 3 — Resource Inventory and Mapping

The goal is to obtain **BP source sentences with valid conservative EP targets**.
The resources below can provide existing pairs or BP sources for which we need
to create and human-validate a target.

| Resource | Context | EP target | Possible role | Main limitation |
| --- | --- | --- | --- | --- |
| GPT-Wikipedia | No reliable surrounding context. | Available. | Conservative, minimal-change BP→EP examples. | Synthetic. |
| [Golden Collection][GC] | Currently unverified. | Available, but needs checking. | Natural conservative BP→EP examples, including sources that can remain unchanged. | Reference quality and context need checking. |
| [FRMT][FRMT] | Potentially available; needs checking. | Available, but needs checking for conservative adaptation. | Regional phenomena and possible examples of broader rewriting for the judge to penalize. | Often contains more reformulation than our task wants; each example needs checking, since not all are unsuitable. |
| [DSL-TL][DSL] | No or limited context. | Needs to be created and human-validated. | Clearly variety-specific or shared/common sentences. | Not a rewriting dataset; selected BP sources need targets. |
| [PtBrVarId][PTD] | Varies. | Needs to be created and human-validated. | Natural BP examples across domains. | Targets need creation; context quality varies. |
| [OpenSubtitles](https://aclanthology.org/L16-1147/) | Neighboring subtitle turns may be available. | Corresponding EP material available, but needs checking. | Conversational pronouns, forms of address, clitics, and register. | Noisy segmentation/alignment and translation choices. |
| [TED data](https://huggingface.co/datasets/IWSLT/ted_talks_iwslt) | Sentence sequences may be available. | Available, but needs checking for conservative adaptation. | Connected spoken examples. | Translations may contain broader rewriting. |
| [e-PACT / Gold-CLUE-Paraphrases](https://multi3generation.inesc-id.pt/?page_id=1701) | Needs verification. | Corresponding EP material available. | Multiword and syntactic differences. | Literary or translator style may influence the pairs. |
| [CHAVE][CHID] | Yes, document context is available. | Needs to be created and human-validated. | Natural BP sources with reliably recoverable previous and next sentences. | No existing BP→EP pair; older newspaper data limits period and domain coverage. |

## Decisions to Confirm

1. **Context:** Where reliable context is available, we propose keeping:

   ```text
   previous sentence
   [BP SOURCE SENTENCE]
   next sentence
   ```

   Only the middle sentence is adapted and later annotated. Does this seem
   reasonable, particularly since CHAVE currently appears to be the clearest
   source with reliably recoverable document context?

2. **Dataset size and candidates per source:** Each BP→EP source–target pair will
   have several candidate adaptations for humans to annotate, providing
   supervision to train the LLM judge. Given the limited visit time and human
   annotation effort, approximately how many **source sentences** and how many
   **candidate outputs per source sentence** would be realistic and useful?

## What Comes Next

**Following the meeting decisions, the next steps are:**

1. **Draft the annotation guidelines and phenomenon decision sheet**
   Define what annotators must inspect, distinguish required adaptations from
   preferences or context-dependent alternatives, and anchor minor/major
   severity and the 0–100 score. The working draft is
   [`CMU_BP_EP_annotation_guidelines_v0.1.md`](../guidelines/CMU_BP_EP_annotation_guidelines_v0.1.md).

2. **Select the 40-sentence self-pilot**
   Choose a deliberately varied sample from two or three domains and multiple
   documents. Represent the linguistic differences in Stage 2, shared/no-change
   cases, and difficult boundary cases. Where reliable context exists, keep the
   previous and next sentences.

3. **Create four pilot candidates per source**
   Use copy/null, the strongest available conservative model, a weaker model,
   and a contrast candidate that contributes a missing behaviour such as
   over-rewriting or meaning drift. Blind candidate identity and randomize the
   display order.

4. **Self-annotate and revise**
   Annotate the 160 candidate outputs using error spans with minor/major
   severity and a 0–100 score. Log unclear decisions, revise the guidelines and
   data-selection rules, and re-annotate a subset to test stability.

5. **Select and annotate the main dataset**
   Only after the pilot, select the full source set using the revised rules.
   Establish validated conservative EP targets for researcher-side quality
   control where needed, but do not show a single reference as the only valid
   answer to annotators. Human annotations become the gold data for the judge.

6. **Split the data**
   Separate the annotated examples into development and held-out test data,
   keeping every candidate from the same source—and preferably every source
   from the same document—in one split. Add a training split only if evaluator
   fine-tuning is later justified by the amount of gold data.

7. **Develop and evaluate the LLM judge**
   Begin without evaluator fine-tuning: use development data for instructions,
   prompting, few-shot examples, and model selection. Run the frozen setup on
   held-out examples and compare predicted spans, severities, and scores with
   human gold. Fine-tune an evaluator only as a later branch if the dataset is
   sufficiently large.

[BM18]: https://aclanthology.org/W18-3912/
[CJ18]: https://aclanthology.org/W18-3931/
[CY01]: https://ojs.uel.br/revistas/uel/index.php/signum/article/view/4366
[A25]: https://aclanthology.org/2025.vardial-1.2/
[DSL]: https://aclanthology.org/2024.lrec-main.882/
[GC]: https://arxiv.org/html/2408.07457v1
[FRMT]: https://aclanthology.org/2023.tacl-1.39/
[PTD]: https://huggingface.co/datasets/liaad/PtBrVId
[CHID]: https://linguateca.pt/acesso/PJR.html
