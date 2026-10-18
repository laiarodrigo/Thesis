# Pearmut 1.1.5 cESA findings

These findings come from the locally installed package at `thesis/lib/python3.12/site-packages/pearmut` (version 1.1.5), its client bundle, CLI validation code, and the local `humeval-tutorial/our_own_humeval/campaigns/0c_cesa.json` example. The installation was not modified.

## Document representation and display

- A document is a list of segment dictionaries. For `task-based` assignment, the nesting is `data -> task/user -> documents -> segments`.
- The pilot's single task therefore contains one document, and that document contains ten ordered `{src, tgt}` segments.
- Pearmut sends the whole document as one payload and renders every segment on one annotation page, in order.
- Each segment is a row: its `src` is shown, followed by all `tgt` entries as horizontally aligned candidate columns.
- Consequently, previous and next **source** segments are visible by scrolling, and previous and next **candidate outputs** for every system are also visible. Context is not limited to only one neighbor.
- All four candidates are visible side by side for every row. Horizontal scrolling is synchronized across rows so column identity stays aligned throughout the document.
- No `ref` field is needed; omitting it means no reference is shown.

## Identity and order

- `show_model_names: false` hides the model-name heading, although the internal dictionary keys remain in the payload and annotation log.
- On campaign import, `shuffle` defaults to `true`. Pearmut shuffles the order of the `tgt` dictionary once per document and applies the same order to every segment in that document.
- Version 1.1.5 creates `random.Random()` without accepting or recording a campaign seed. Its automatic ordering is therefore not reproducible from the original template.
- For this pilot, the final builder should use a recorded seed to map the four roles to anonymous keys and order them, save that private mapping, and set Pearmut `shuffle: false`. This preserves the chosen blind order exactly.

## cESA annotations

- cESA appends the configured special token (default `[missing]`; this pilot uses `[MISSING]`) to every candidate. Selecting it creates an ordinary character-index span at the synthetic token's appended index.
- A span is stored as `{start_i, end_i, category, severity}`. In cESA, `category` remains `null`; `severity` is the lowercase string `minor` or `major`. Error-type categories are required only by the `MQM` protocol, not cESA.
- The overall score is stored per segment and per internal model key as `score`. The cESA UI uses a 0–100 range in steps of 5.
- The optional settings-box comment is stored once for the submitted document/item as top-level `comment`, not separately per sentence or candidate.
- A submitted task-based log row contains the whole source/candidate `item`, `annotation` (one entry per segment and model), `actions`, optional `comment`, `user_id`, and `item_i`.

## Timing

- The client records epoch-second action entries, beginning with `load` and ending with `submit`; score and span actions include segment `index` and internal `model` key.
- These raw `actions` are saved with the submitted document in `data/outputs/<campaign_id>.jsonl`.
- Progress stores per-user/per-campaign `time_start`, `time_end`, and cumulative `time`.
- Cumulative `time` sums adjacent action gaps but caps every gap at 61 seconds to suppress long pauses. It is an active-time approximation, not exact elapsed time.
- With all ten segments in one document, the primary unit is one page/document submission. Some action events identify a segment, but Pearmut does not store a reliable independent elapsed duration for each sentence or candidate.
