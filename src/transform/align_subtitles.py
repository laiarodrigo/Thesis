import pysrt, re
from datetime import timedelta
from fuzzywuzzy import fuzz
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd
from typing import List, Dict, Sequence, Tuple, Callable


TIME_WINDOW = timedelta(seconds=60)  # Maximum allowable time difference (start with a large value, aim at around 0.5s)
WEIGHT_TIME = 0.3
WEIGHT_TEXT = 0.7
SIMILARITY_THRESHOLD = 30  # Minimum fuzzy ratio for alignment, considering similarity metric frm fuzzywuzzy

# def load_subtitles(filepath):
#     subs = pysrt.open(filepath)
#     return [
#         {
#             "start": pd.to_timedelta(sub.start.ordinal, unit="ms"),
#             "end":   pd.to_timedelta(sub.end.ordinal,   unit="ms"),
#             "text":  sub.text.strip(),
#         } for sub in subs
#     ]

def time_diff_score(t1, t2, delta):
    diff = abs((t1 - t2).total_seconds())
    if diff > delta.total_seconds(): return 0.0
    return 1.0 - (diff / delta.total_seconds())

def align_subtitles_greedy(subs_a, subs_b, time_window=TIME_WINDOW):
    aligned_pairs = []
    used_b_indices = set()    
    for i, a in enumerate(subs_a):
        best_match = None
        best_score = 0
        for j, b in enumerate(subs_b):
            if j in used_b_indices: continue  # Avoid multiple matches
            if abs((a['start'] - b['start'])) <= time_window:
                time_score = time_diff_score(a['start'], b['start'], time_window)
                text_score = fuzz.ratio(a['text'], b['text']) / 100.0
                total_score = WEIGHT_TIME * time_score + WEIGHT_TEXT * text_score              
                if total_score > best_score:
                    best_match = (i, j, total_score)
                    best_score = total_score
        if best_match and best_score >= SIMILARITY_THRESHOLD / 100.0:
            i, j, score = best_match
            aligned_pairs.append((subs_a[i], subs_b[j], score))
            used_b_indices.add(j)
        else: aligned_pairs.append((subs_a[i], None, 0))    
    return aligned_pairs

# def align_subtitles_optimal_hungarian(subs_a, subs_b, time_window=TIME_WINDOW):
#     num_a = len(subs_a)
#     num_b = len(subs_b)
#     delta = time_window
#     cost_matrix = np.full((num_a, num_b), -1e6)  # Negative cost because linear_sum_assignment minimizes
#     for i, a in enumerate(subs_a):
#         for j, b in enumerate(subs_b):
#             if abs((a['start'] - b['start'])) <= delta:
#                 time_score = time_diff_score(a['start'], b['start'], delta)
#                 text_score = fuzz.ratio(a['text'], b['text']) / 100.0
#                 total_score = WEIGHT_TIME * time_score + WEIGHT_TEXT * text_score
#                 cost_matrix[i, j] = -total_score
#     row_ind, col_ind = linear_sum_assignment(cost_matrix)
#     aligned_pairs = []
#     for i, j in zip(row_ind, col_ind):
#         if cost_matrix[i, j] == -1e6: aligned_pairs.append((subs_a[i], None, 0))
#         else:
#             score = -cost_matrix[i, j]
#             if score >= SIMILARITY_THRESHOLD / 100.0: aligned_pairs.append((subs_a[i], subs_b[j], score))
#             else: aligned_pairs.append((subs_a[i], None, 0))
#     unmatched_a = set(range(num_a)) - set(row_ind)
#     for i in unmatched_a: aligned_pairs.append((subs_a[i], None, 0))
#     return aligned_pairs

def align_subtitles_optimal_hungarian(subs_a, subs_b):
    num_a = len(subs_a)
    num_b = len(subs_b)
    delta = TIME_WINDOW
    #print(f"Aligning {num_a} subtitles from A with {num_b} subtitles from B")
    cost_matrix = np.full((num_a, num_b), 1e6)  # Negative cost because linear_sum_assignment minimizes
    for i, a in enumerate(subs_a):
        for j, b in enumerate(subs_b):
            if abs((a['start'] - b['start'])) <= delta:
                time_score = time_diff_score(a['start'], b['start'], delta)
                text_score = fuzz.ratio(a['text'], b['text']) / 100.0
                total_score = WEIGHT_TIME * time_score + WEIGHT_TEXT * text_score
                cost_matrix[i, j] = -total_score
        # print(f"Processed subtitle A[{i}]: {a['text']} (Cost: {cost_matrix[i]})")
    
    #print('np.all',np.all(cost_matrix == 1e6))  # Check if all costs are valid
    # Print the (i, j) indices where cost_matrix is not -1e6
    indices = np.argwhere(cost_matrix != 1e6)

    cost_matrix_2 = cost_matrix.copy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix_2)
    aligned_pairs = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] == 1e6: 
            aligned_pairs.append((subs_a[i], None, 0))
        else:
            score = -cost_matrix[i, j]            
            if score >= SIMILARITY_THRESHOLD / 100.0: 
                #print('matching pair (i, j)', i, j)
                aligned_pairs.append((subs_a[i], subs_b[j], score))
            else: 
                #print('no pair found')
                aligned_pairs.append((subs_a[i], None, 0))
    #print('segunda volta')
    #print('np.all',np.all(cost_matrix == 1e6))  # Check if all costs are valid
    # Print the (i, j) indices where cost_matrix is not 1e6
    # indices = np.argwhere(cost_matrix != 1e6)
    # for i, j in indices:
        #print(f"cost_matrix[{i}, {j}] = {cost_matrix[i, j]}")
    
    unmatched_a = set(range(num_a)) - set(row_ind)
    for i in unmatched_a: aligned_pairs.append((subs_a[i], None, 0))
    return aligned_pairs


def find_offset_between_subtitles(aligned_pairs, min_score=0.6):
    time_differences = []
    scores = []
    matched = 0
    
    for a, b, score in aligned_pairs:
        if b is not None and score > min_score:
            matched += 1
            time_differences.append((a['start'] - b['start']).total_seconds())
            scores.append(score)

    if len(time_differences) == 0:
        return 0.0, 0.0
    
    total = len(aligned_pairs)
    coverage = matched / total if total > 0 else 0.0
    avg_offset = np.mean(time_differences) if time_differences else 0.0
    max_offset = np.max(np.abs(time_differences)) if time_differences else 0.0
    mean_score = np.mean(scores) if scores else 0.0

    return avg_offset, max_offset, coverage, mean_score

def find_offset_between_subtitles_percentile(
    aligned_pairs: Sequence[Tuple[Dict[str, object], Dict[str, object], float]],
    top_percent: float = 10.0,          # keep the top-10 % most similar
    min_pairs: int = 3                  # need at least this many to compute stats
) -> Tuple[float, float, float, float]:

    diffs, scores = [], []
    for a, b, s in aligned_pairs:
        if b is not None:
            diffs.append((a["start"] - b["start"]).total_seconds())
            scores.append(float(s))

    total = len(scores)
    if total == 0:
        return 0.0, 0.0, 0.0, 0.0

    # percentile threshold
    q = 100.0 - top_percent            # e.g. 90-th percentile for top 10 %
    score_cutoff = np.percentile(scores, q)

    # keep only pairs at / above the cutoff
    kept_diffs  = [d for d, s in zip(diffs, scores) if s >= score_cutoff]
    kept_scores = [s for s       in scores            if s >= score_cutoff]
    kept_diffs = np.array(kept_diffs)

    if len(kept_diffs) < min_pairs:    
        print(f"Warning: not enough pairs ({len(kept_diffs)})")
        return 0.0, 0.0, len(kept_diffs) / total, (np.mean(kept_scores) if kept_scores else 0.0)

    avg_offset = float(np.median(kept_diffs))        # median, not mean
    max_offset = float(np.max(np.abs(kept_diffs)))
    coverage   = len(kept_diffs) / total
    mean_score = float(np.mean(kept_scores))

    return avg_offset, max_offset, coverage, mean_score

def shift_srt(subs, offset_seconds):
    delta = pd.to_timedelta(offset_seconds, unit='s')
    shifted = []
    for record in subs:
        shifted.append(
            {
                "start": record["start"] - delta,   # subtract the signed Δ
                "end":   record["end"]   - delta,
                "text":  record["text"],
            }
        )
    return shifted

def eliminate_new_lines(subtitles):
    for sub in subtitles:
        if '\n' in sub['text']:
            sub['text'] = sub['text'].replace('\n', ' ')

def strip_tags_str(text: str) -> str:
    # remove <…> and {…} tags and trim whitespace
    return re.sub(r'<[^>]+>|\{[^}]+\}', '', text).strip()

# 2) in‐place block‐level cleaner
def clean_sub_blocks(blocks: List[Dict[str, object]]) -> None:
    """
    Given a list of dicts each with a "text" field,
    strip tags from every block in place.
    """
    for b in blocks:
        b["text"] = strip_tags_str(b["text"])

def merge_subtitle_fragments(
    blocks: Sequence[Dict[str, object]],
    gap_threshold: timedelta | pd.Timedelta = pd.Timedelta(milliseconds=120),
    join_punct: str                   = " ",
    terminal_stop: str                = ".?!",
) -> List[Dict[str, object]]:
    gap_threshold = pd.Timedelta(gap_threshold)
    merged: List[Dict[str, object]] = []
    i = 0
    n = len(blocks)

    while i < n:
        curr = blocks[i].copy()
        i += 1

        while i < n:
            nxt  = blocks[i]
            gap  = nxt["start"] - curr["end"]
            tail_raw = curr["text"].rstrip()
            tail     = tail_raw[-1:]    # last character ("," or "." or letter)
            head     = nxt["text"].lstrip()[:1]

            # allow merge if small gap
            if gap > gap_threshold:
                break

            # if it ends in a comma, always merge
            if tail == ",":
                mergeable = True
            # otherwise, only merge if not ending in a hard stop and next starts lowercase
            else:
                mergeable = (tail not in terminal_stop) and head.islower()

            if not mergeable:
                break

            # --- perform merge keeping the comma this time ------------
            if tail == ",":
                curr["text"] = tail_raw + join_punct + nxt["text"].lstrip()
            else:
                curr["text"] = tail_raw + join_punct + nxt["text"].lstrip()

            curr["end"] = nxt["end"]
            i += 1

        merged.append(curr)

    return merged

def auto_sync_subs(
    subs_a: List[Dict[str, object]],
    subs_b: List[Dict[str, object]],
    *,
    aligner: Callable = align_subtitles_optimal_hungarian,
    offset_estimator: Callable = find_offset_between_subtitles_percentile,
    eps_offset: float = 0.05,           # 50 ms
    max_plateau_rounds: int = 2,
) -> Tuple[List[Dict[str, object]], float]:
    """
    Iteratively shift *subs_a* so that it lines-up with *subs_b*.

    Returns
    -------
    shifted_a : list[dict]
        The time-corrected subtitle blocks for A.
    final_offset : float
        The last applied offset in seconds (positive ⇒ A shifted earlier).

    Notes
    -----
    • Stops when |avg_offset| < eps_offset **and**
      max_offset < 2*eps_offset, **or** when coverage & mean score
      plateau (rounded to 3-decimals) for `max_plateau_rounds` passes.
    • Works in-place on a *copy* of subs_a; subs_b is never modified.
    """
    subs_a = [row.copy() for row in subs_a]   # work on a clone

    plateau_rounds   = 0
    prev_cov_round   = prev_score_round = None

    while True:
        aligned = aligner(subs_a, subs_b)
        avg_off, max_off, cov, mean = offset_estimator(aligned)

        cov_r, score_r = round(cov, 3), round(mean, 3)
        print(f"Offset {avg_off:+.3f}s (max {max_off:.3f})  "
              f"Coverage {cov_r:.3f}  Mean {score_r:.3f}")

        # good enough?
        if abs(avg_off) < eps_offset and max_off < 2*eps_offset:
            break

        # plateau?
        if (cov_r, score_r) == (prev_cov_round, prev_score_round):
            plateau_rounds += 1
            if plateau_rounds >= max_plateau_rounds:
                print("plateau → stop")
                break
        else:
            plateau_rounds = 0

        prev_cov_round, prev_score_round = cov_r, score_r

        # shift *A* by the signed average offset
        subs_a = shift_srt(subs_a, avg_off)

    return subs_a, avg_off

# def main():
#     file_a = 'subtitles_a.srt'
#     file_b = 'subtitles_b.srt'  
#     subs_a = load_subtitles(file_a)
#     subs_b = load_subtitles(file_b)
#     aligned = align_subtitles_optimal_hungarian(subs_a, subs_b)
#     for a, b, score in aligned:
#         print(f"A: [{a['start']}] {a['text']}")
#         if b: print(f"B: [{b['start']}] {b['text']} (Score: {score:.2f})")
#         else: print("B: [No Match]")
#         print("-" * 50)

# if __name__ == "__main__": main()