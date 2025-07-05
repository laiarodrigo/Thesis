import pysrt
from datetime import timedelta
from fuzzywuzzy import fuzz
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

TIME_WINDOW = timedelta(seconds=60)  # Maximum allowable time difference (start with a large value, aim at around 0.5s)
WEIGHT_TIME = 0.3
WEIGHT_TEXT = 0.7
SIMILARITY_THRESHOLD = 30  # Minimum fuzzy ratio for alignment, considering similarity metric frm fuzzywuzzy

def load_subtitles(filepath):
    subs = pysrt.open(filepath)
    return [
        {
            "start": pd.to_timedelta(sub.start.ordinal, unit="ms"),
            "end":   pd.to_timedelta(sub.end.ordinal,   unit="ms"),
            "text":  sub.text.strip(),
        } for sub in subs
    ]

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
    
    for a, b, score in aligned_pairs:
        if b is not None and score > min_score:
            time_differences.append(abs((a['start'] - b['start']).total_seconds()))
    
    if len(time_differences) == 0:
        return 0.0, 0.0
    
    average_offset = np.mean(time_differences)
    maximum_offset = np.max(time_differences)
    
    return average_offset, maximum_offset


def shift_srt(subs, offset_seconds):
    delta = pd.to_timedelta(offset_seconds)
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