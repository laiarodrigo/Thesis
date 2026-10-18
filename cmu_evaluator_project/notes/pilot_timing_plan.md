# Pilot annotation timing plan

## What Pearmut 1.1.5 records

For each submitted document, Pearmut writes an `actions` array to `data/outputs/<campaign_id>.jsonl`. Every action has a Unix timestamp in seconds and an action name. `score`, `create_span`, and `delete_span` actions also carry the segment `index` and candidate/model key. The sequence begins with `load` and ends with `submit`.

The progress file also stores, per user and campaign:

- `time_start`: earliest submitted action timestamp;
- `time_end`: latest submitted action timestamp; and
- `time`: cumulative approximate active seconds.

Pearmut computes `time` as the sum of gaps between consecutive actions, with each gap capped at 61 seconds. Timing is therefore available at action level in the raw log and as an aggregate per user/campaign. It is not a reliable per-candidate duration. This pilot uses ten Pearmut documents/pages. Each page shows the full ten-sentence BP passage as read-only source context and submits the four candidates for one current sentence. Each successful **Next** is therefore a resumable four-judgment checkpoint.

## Values to report

After the self-pilot, report both:

1. **Pearmut active-time estimate:** the user's final `time` value from progress; and
2. **raw elapsed time:** the sum of `submit.time - load.time` over the ten completed JSONL document entries.

Also retain the raw `actions` array. Segment-indexed events can support exploratory diagnostics, but they must not be presented as complete per-segment reading time.

## Manual fallback and cross-check

Even when Pearmut logging is available, run a simple external timer as a cross-check:

1. start immediately before opening the pilot annotation page;
2. annotate all 10 source items / 40 candidate outputs;
3. stop immediately after the final submission;
4. record total elapsed active time; and
5. separately record the duration and reason for any major interruption.

If the pilot is paused between pages, stop the manual timer only after the
current page has submitted successfully. Restart it immediately before opening
or resuming the writable annotation link, and report the sum of all timed
annotation sessions. Do not leave a manual timer running overnight. Do not
subtract ordinary thinking time. If an interruption occurs within a session,
report both wall-clock elapsed time and interruption-adjusted active time. No
timing values are fabricated in this planning document.
