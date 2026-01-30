import duckdb
import pandas as pd, pathlib, itertools, textwrap, re, gc
import numpy as np
import random
import unicodedata
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from fuzzywuzzy import fuzz
from rapidfuzz import fuzz, distance
from simalign import SentenceAligner
_device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    _ALIGNER = SentenceAligner(model="xlmr", token_type="word", matching_methods="a", device=_device)
except TypeError:
    _ALIGNER = SentenceAligner(model="xlmr", token_type="word", matching_methods="a")

print("CUDA available:", torch.cuda.is_available(), 
      "visible GPUs:", torch.cuda.device_count(),
      "device:", (_device if torch.cuda.is_available() else "cpu-only"))
from typing import Optional, Dict, Any, Tuple, List, Iterable, Union

ROOT = pathlib.Path(__file__).resolve().parents[1]
DB = os.environ.get("DB", str(ROOT / "data/duckdb/subs.duckdb"))
TAG_RE = re.compile(r'<[^>]+>|\{[^}]+\}')
NL_RE  = re.compile(r'\s*\n\s*')
SENT_SPLIT_RE = re.compile(r'(?<=[\.\?\!…])\s+')

ABBREVS = {"dr","dra","sr","sra","srta","prof","profa","etc","av","nº","n.º","vs","p.ex"}
ABBR_RX = re.compile(r'\b(' + '|'.join(re.escape(x) for x in ABBREVS) + r')\.', re.IGNORECASE)

# PREPROCESSING
QUOTE_PAIRS = [
    ('"', '"'), ("'", "'"),
    ('“', '”'), ('‘', '’'),
    ('«', '»'), ('„', '“'), ('‹', '›')
]
QUOTE_CHARS = set(ch for L,R in QUOTE_PAIRS for ch in (L,R))

HARD_STOPS = ".?!…"
SOFT_TAILS = ",;:—–-"
DASHES     = "-–—"

LETTER      = r"[^\W\d_]"                                   # any letter, no digits/underscore
NAME_TOKEN  = rf"{LETTER}(?:{LETTER}|[.'\-])*"              # e.g., Steve, O'Neill, João-Pedro
NAME_PHRASE = rf"{NAME_TOKEN}(?:\s+{NAME_TOKEN}){{0,2}}"    # up to 3-word names

WS = r"(?:\s|\u00A0|\u202F)*"                               # normal/narrow/nbsp
SPEAKER_LABEL_DROP = re.compile(
    rf"(^|[^\w]){NAME_PHRASE}{WS}[:\uFF1A]{WS}",            # keep boundary, drop label
    re.UNICODE
)

WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
# Lightweight PT stopword set for "content-token" accounting
PT_PREPS = {"de","do","da","dos","das","em","no","na","nos","nas","com","para","por","a","ao","à","às","aos"}
PT_DETS  = {"o","a","os","as","um","uma","uns","umas","este","esta","estes","estas","esse","essa","esses","essas","aquele","aquela","aqueles","aquelas"}
PT_CLITICS={"me","te","se","lhe","nos","vos","lhes"}
PT_CONJ  = {"e","ou","mas","nem","que","porque","pois","porém","porem"}
PT_NEG   = {"não","nao"}
PT_STOPWORDS = (PT_PREPS | PT_DETS | PT_CLITICS | PT_CONJ | PT_NEG)

# sentence splitter (no look-behind) for light stats only
_SENT_RE = re.compile(r'.*?[.!?…]+(?:["”»\'\)\]\}]+)?(?=\s|$)|.+?(?=\s|$)', re.UNICODE)

ALIGN_METHOD = "inter"  # alternatives: "inter" (↑recall), "mwmf", "itermax", "union"

# === alignment tokenizer (use ONLY for SimAlign) ===
WS_TOKEN = re.compile(r"\S+", re.UNICODE)

# ---------- cleaning + similarity ----------
def clean_text(s: str) -> str:
    if not s: return ""
    return TAG_RE.sub('', NL_RE.sub(' ', s)).strip()

def sim(a: str, b: str) -> float:
    a = clean_text(a); b = clean_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0

    # edit-distance core signals (all penalize insertions/deletions)
    s_edit  = fuzz.ratio(a, b) / 100.0
    s_sort  = fuzz.token_sort_ratio(a, b) / 100.0         # order-insensitive but still length-aware
    s_lev   = distance.Levenshtein.normalized_similarity(a, b)  # 0..1

    # penalize big length mismatches (e.g., a is much longer than b)
    lp = min(len(a), len(b)) / max(len(a), len(b))  # 0..1

    # blend; weights are tame and easy to tune
    base = 0.5*s_edit + 0.2*s_sort + 0.3*s_lev
    return base * (0.5 + 0.5*lp)   # shrink score when lengths differ a lot

def new_sim():
    pass

# ---------- clause split (sentences first, comma/dash fallback) ----------
def mask_abbrevs(t: str) -> str: return ABBR_RX.sub(lambda m: m.group(1)+"§", t or "")
def unmask_abbrevs(t: str) -> str: return (t or "").replace("§",".")

def sentence_split(t: str):
    tt = mask_abbrevs(t or "")
    parts = [unmask_abbrevs(p).strip() for p in SENT_SPLIT_RE.split(tt.strip()) if p.strip()]
    return parts

def split_tail_clause(text: str, max_tail_chars=60):
    parts = sentence_split(text)
    if len(parts) >= 2:
        head = ' '.join(parts[:-1]).strip(); tail = parts[-1].strip()
        if head and tail: return head, tail
    t = (text or "").strip()
    for token in [",", " - ", " – ", " — "]:
        k = t.rfind(token)
        if k != -1 and 1 <= len(t) - (k+len(token)) <= max_tail_chars:
            return t[:k].strip(), t[k+len(token):].strip()
    return None, None

def split_head_clause(text: str, max_head_chars=60):
    parts = sentence_split(text)
    if len(parts) >= 2:
        head = parts[0].strip(); rest = ' '.join(parts[1:]).strip()
        if head and rest: return head, rest
    t = (text or "").strip()
    for token in [",", " - ", " – ", " — "]:
        k = t.find(token)
        if k != -1 and 1 <= k+1 <= max_head_chars:
            return t[:k+len(token)].strip(), t[k+len(token):].strip()
    return None, None

def ok_piece(seg: str, min_chars=6, min_tokens=2):
    toks = [w for w in re.findall(r'\b\w+\b', seg or "", flags=re.UNICODE) if any(c.isalpha() for c in w)]
    return bool(seg) and len(seg) >= min_chars and len(toks) >= min_tokens

def _py_int(x):
    # robust cast for numpy/pandas scalars and plain ints
    if isinstance(x, (np.generic,)):  # np.int64, np.int32, etc.
        return int(x.item())
    return int(x)

def load_opus_window(start_line: int, window: int = 600) -> pd.DataFrame:
    start_line = _py_int(start_line)
    window     = _py_int(window)
    with duckdb.connect(str(DB)) as con:
        df = con.execute("""
            SELECT line_no, pair_id, sent_pt_br, sent_pt_pt
            FROM opus_moses
            WHERE line_no BETWEEN ? AND ?
            ORDER BY line_no
        """, [start_line, start_line + window - 1]).df()
    return df.fillna("")


# ---------- PASS A: neighbor MOVES (choose tail→next or head←next if it increases sum) ----------
def apply_neighbor_moves(df: pd.DataFrame, margin=0.04, max_clause_chars=60):
    df2 = df.copy()
    log = []
    n = len(df2)
    for i in range(n-1):
        for lang, other in (("sent_pt_pt","sent_pt_br"), ("sent_pt_br","sent_pt_pt")):
            L_i,  L_ip1  = df2.at[i,lang],     df2.at[i+1,lang]
            R_i,  R_ip1  = df2.at[i,other],    df2.at[i+1,other]
            keep_sum = sim(L_i,R_i) + sim(L_ip1,R_ip1)

            # option 1: move tail of i -> front of i+1
            head, tail = split_tail_clause(L_i, max_tail_chars=max_clause_chars)
            gain1 = -1e9
            if ok_piece(tail) and ok_piece(head, min_chars=4):
                move_sum1 = sim(head, R_i) + sim((tail + " " + (L_ip1 or "")).strip(), R_ip1)
                gain1 = move_sum1 - keep_sum

            # option 2: move head of i+1 -> end of i
            head2, rest2 = split_head_clause(L_ip1, max_head_chars=max_clause_chars)
            gain2 = -1e9
            if ok_piece(head2) and ok_piece(rest2, min_chars=4):
                move_sum2 = sim(((L_i or "") + (" " if L_i else "") + head2).strip(), R_i) + sim(rest2, R_ip1)
                gain2 = move_sum2 - keep_sum

            # apply the better positive option
            if gain1 > margin and gain1 >= gain2:
                df2.at[i,lang]     = head
                df2.at[i+1,lang]   = (tail + " " + (L_ip1 or "")).strip()
                log.append({"i": i, "lang": lang, "op": "tail_to_next", "gain": float(gain1)})
            elif gain2 > margin and gain2 > gain1:
                df2.at[i,lang]     = (((L_i or "") + (" " if L_i else "") + head2).strip())
                df2.at[i+1,lang]   = rest2
                log.append({"i": i, "lang": lang, "op": "head_from_next", "gain": float(gain2)})
            # else: no move
    return df2, pd.DataFrame(log)

def _moved_piece(bi, ai, bip1, aip1, op):
    """Best-effort extract of the moved fragment from before/after strings."""
    if op == "tail_to_next":
        # ai = head; piece = suffix removed from bi
        if bi.startswith(ai):
            return bi[len(ai):].strip()
        # fallback: prefix added to next
        added = max(0, len(aip1) - len(bip1))
        return aip1[:added].strip()
    else:  # "head_from_next"
        # aip1 = rest; piece = prefix removed from bip1
        if len(bip1) > len(aip1):
            return bip1[:len(bip1) - len(aip1)].strip()
        # fallback: suffix added to i
        if ai.startswith(bi):
            return ai[len(bi):].strip()
        return ""

def preview_moves(df_before, df_after, move_log, k=8):
    """
    Show top-k moves with before/after texts, moved fragment, and score deltas.
    Assumes df_before/df_after are the same window (same line_no order).
    """
    if move_log is None or move_log.empty:
        print("No moves to preview.")
        return

    log = move_log.sort_values("gain", ascending=False).head(k)

    for _, r in log.iterrows():
        i   = int(r["i"])
        op  = r["op"]
        lang = r["lang"]
        other = "sent_pt_pt" if lang == "sent_pt_br" else "sent_pt_br"

        # pull rows
        bi   = df_before.at[i,   lang]
        bip1 = df_before.at[i+1, lang]
        ai   = df_after.at[i,    lang]
        aip1 = df_after.at[i+1,  lang]

        Ri   = df_before.at[i,   other]
        Rip1 = df_before.at[i+1, other]  # other side doesn't change during move

        piece = _moved_piece(bi, ai, bip1, aip1, op)

        keep_sum = sim(bi, Ri) + sim(bip1, Rip1)
        new_sum  = sim(ai, Ri) + sim(aip1, Rip1)
        d_i   = sim(ai, Ri)   - sim(bi, Ri)
        d_ip1 = sim(aip1, Rip1) - sim(bip1, Rip1)

        line_i   = int(df_before.at[i,   "line_no"])
        line_ip1 = int(df_before.at[i+1, "line_no"])

        print("\n────────────────────────────────────────")
        print(f"lines {line_i} → {line_ip1} | {lang} | {op} | gain {float(r['gain']):.3f}")
        print(f"moved piece: [{piece}]")
        print(f"sum sim: {keep_sum:.3f} → {new_sum:.3f}  (Δi={d_i:+.3f}, Δi+1={d_ip1:+.3f})")

        print("\n— BEFORE —")
        print(f"i   ({lang}): {bi}")
        print(f"i+1 ({lang}): {bip1}")
        print(f"i   ({other}): {Ri}")
        print(f"i+1 ({other}): {Rip1}")

        print("\n— AFTER —")
        print(f"i   ({lang}): {ai}")
        print(f"i+1 ({lang}): {aip1}")

def _strip_wrapping_quotes_once(s: str) -> str:
    """Remove exactly one balanced pair of wrapping quotes if present."""
    if not s:
        return s
    t = s.strip()
    for left, right in QUOTE_PAIRS:
        if t.startswith(left) and t.endswith(right):
            inner = t[len(left):-len(right)].strip()
            # keep only if there's some non-quote content inside
            if inner and any(c not in QUOTE_CHARS for c in inner):
                return inner
    return t

def _strip_wrapping_quotes(s: str) -> str:
    """Peel multiple layers, e.g., “ 'foo' ” -> foo."""
    prev, cur = None, s
    while cur != prev:
        prev = cur
        cur = _strip_wrapping_quotes_once(cur)
    return cur

def _strip_edge_quotes_unbalanced(s: str) -> str:
    """
    Also remove lone leading/trailing quotes if they remain (unbalanced).
    Repeats until no edge quote remains or only quotes are left.
    """
    if not s:
        return s
    t = s.strip()
    lefts  = {L for L,_ in QUOTE_PAIRS}
    rights = {R for _,R in QUOTE_PAIRS}

    changed = True
    while changed and t:
        changed = False
        if t and t[0] in lefts:
            t = t[1:].lstrip(); changed = True
        if t and t[-1] in rights:
            t = t[:-1].rstrip(); changed = True
        # stop if the remainder is only quotes/spaces
        if t and all((c in QUOTE_CHARS) or c.isspace() for c in t):
            break
    return t


def _drop_speaker_labels_keep_content(s: str) -> str:
    if not s:
        return s
    # normalize space variants first
    s = (s.replace("\u00A0", " ")
           .replace("\u202F", " ")
           .replace("\u2007", " ")
           .replace("\u2009", " "))

    # keep the boundary, drop the label
    s = SPEAKER_LABEL_DROP.sub(r"\1", s)

    # tidy spacing
    s = re.sub(r"\s+([,.;:!?…)\]\}}])", r"\1", s)
    s = re.sub(r"([(\[\{{«“\"'])\s+", r"\1", s)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s

def _rstrip_quotes(s): return re.sub(r'[\s"\']+$', '', s or "")
def _last_char(s): 
    t = _rstrip_quotes(s or "").rstrip()
    return t[-1:] if t else ""
def _first_alpha_case(s):
    for ch in (s or ""):
        if ch.isalpha(): return "upper" if ch.isupper() else "lower"
    return None

def _starts_with_dash(s): return bool(re.match(r'^\s*['+re.escape(DASHES)+r']\s*', s or ""))
def _strip_leading_dash(s): return re.sub(r'^\s*['+re.escape(DASHES)+r']\s*', '', s or "")
def _remove_dash_after_punct(s): return re.sub(r'([,\.!\?])\s*['+re.escape(DASHES)+r']\s*', r'\1 ', s or "")

def _normalize_spaces(s):
    s = (s or "").replace("\u00A0", " ")
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'\s+([,.;:?!…])', r'\1', s)
    return s

def _is_all_caps_alpha(s):
    letters = [c for c in (s or "") if c.isalpha()]
    return bool(letters) and all(c.isupper() for c in letters)

def _sentence_case_from_lower(s):
    t, out, cap = (s or "").lower(), [], True
    for ch in t:
        if cap and ch.isalpha(): out.append(ch.upper()); cap=False
        else: out.append(ch)
        if ch in HARD_STOPS: cap=True
    return "".join(out)

def _capitalize_first_alpha(s):
    if not s: return s
    chars=list(s)
    for i,ch in enumerate(chars):
        if ch.isalpha(): chars[i]=ch.upper(); break
    return "".join(chars)

def _normalize_line_text(s):
    if not s: return ""
    s = _strip_leading_dash(s)
    s = _drop_speaker_labels_keep_content(s)
    s = _remove_dash_after_punct(s)
    s = _normalize_spaces(s)
    s = _strip_wrapping_quotes(s)
    s = _strip_edge_quotes_unbalanced(s)
    if _is_all_caps_alpha(s):
        s = _sentence_case_from_lower(s)
    return s

def _join_text(a,b):
    b2 = _strip_leading_dash(b).lstrip()
    if not a: return b2
    if not b2: return a
    return (a.rstrip() + " " + b2).strip()

def _has_inner_hard_stop(s):  # two sentences in one row
    return bool(re.search(r'[.?!…].+\S.*[.?!…]', s or ""))

def _should_merge_pair(br_a, br_b, pt_a, pt_b):
    a_end_br = _last_char(br_a); a_end_pt = _last_char(pt_a)
    b_head_br = _first_alpha_case(br_b); b_head_pt = _first_alpha_case(pt_b)
    hard_br = a_end_br in HARD_STOPS; hard_pt = a_end_pt in HARD_STOPS

    cont_br = ((a_end_br in SOFT_TAILS) or (len(br_a) < 40)) and (b_head_br=="lower" or _starts_with_dash(br_b))
    cont_pt = ((a_end_pt in SOFT_TAILS) or (len(pt_a) < 40)) and (b_head_pt=="lower" or _starts_with_dash(pt_b))

    underseg = (_has_inner_hard_stop(pt_a) and not _has_inner_hard_stop(br_a)) or \
               (_has_inner_hard_stop(br_a) and not _has_inner_hard_stop(pt_a))

    if hard_br and hard_pt and (b_head_br=="upper") and (b_head_pt=="upper") and not underseg:
        return False
    return bool(cont_br or cont_pt or underseg)

import duckdb, pandas as pd, gc

def _run_moves_df(df: pd.DataFrame, margin=0.04, max_clause_chars=60, max_iters=5) -> pd.DataFrame:
    """Repeat apply_neighbor_moves on a DataFrame until no more moves or max_iters."""
    cur = df.loc[:, ["line_no","pair_id","sent_pt_br","sent_pt_pt"]].copy()
    for _ in range(int(max_iters)):
        nxt, log = apply_neighbor_moves(cur, margin=margin, max_clause_chars=max_clause_chars)
        if log is None or log.empty:
            break
        cur = nxt
    return cur

def apply_neighbor_moves_corpus_inplace(
    block_size: int = 50_000,
    overlap: int = 3,                 # rows kept between blocks so moves can cross the seam
    margin: float = 0.04,
    max_clause_chars: int = 60,
    max_iters: int = 5,
):
    """
    Stream over opus_moses and apply your neighbor-move heuristic in-place.
    - Processes in blocks with 'overlap' rows carried forward.
    - Only updates rows that actually changed.
    - No row-count changes (this pass only moves clauses).
    """
    with duckdb.connect(str(DB)) as con:
        lo, hi = con.execute("SELECT min(line_no), max(line_no) FROM opus_moses").fetchone()
        lo, hi = int(lo), int(hi)

        cur_start = lo
        carry_df = None  # last 'overlap' rows of the previous processed block (already moved)

        while cur_start <= hi:
            # choose fetch start/count so we include the carried rows
            if carry_df is None:
                fetch_start = cur_start
                fetch_count = min(block_size, hi - fetch_start + 1)
            else:
                fetch_start = int(carry_df["line_no"].iloc[0])
                fetch_count = min(block_size + overlap, hi - fetch_start + 1)

            # load from DB
            df = con.execute("""
                SELECT line_no, pair_id, sent_pt_br, sent_pt_pt
                FROM opus_moses
                WHERE line_no BETWEEN ? AND ?
                ORDER BY line_no
            """, [fetch_start, fetch_start + fetch_count - 1]).df()

            # overlay carried texts onto the front (so we start from the already-moved boundary)
            if carry_df is not None and not carry_df.empty:
                df = df.merge(carry_df[["line_no","sent_pt_br","sent_pt_pt"]],
                              on="line_no", how="left", suffixes=("","_car"))
                for col in ("sent_pt_br","sent_pt_pt"):
                    rep = df[col + "_car"]
                    df[col] = rep.where(rep.notna(), df[col])
                    df.drop(columns=[col + "_car"], inplace=True)

            # keep a copy for diffing
            orig = df.loc[:, ["line_no","sent_pt_br","sent_pt_pt"]].copy()

            # run your move heuristic on this combined block
            moved = _run_moves_df(df, margin=margin, max_clause_chars=max_clause_chars, max_iters=max_iters)

            # decide how many rows to flush now (keep the last 'overlap' rows for the next block)
            is_last_block = (fetch_start + len(df) - 1) >= hi
            flush_n = len(moved) if is_last_block else max(0, len(moved) - overlap)

            if flush_n:
                out = moved.iloc[:flush_n]
                base = orig.iloc[:flush_n]

                # diffs → only update changed rows
                changed = (out["sent_pt_br"] != base["sent_pt_br"]) | (out["sent_pt_pt"] != base["sent_pt_pt"])
                upd = out.loc[changed, ["line_no","sent_pt_br","sent_pt_pt"]]

                if not upd.empty:
                    con.register("upd", upd)
                    con.execute("""
                        UPDATE opus_moses AS o
                        SET sent_pt_br = u.sent_pt_br,
                            sent_pt_pt = u.sent_pt_pt
                        FROM upd AS u
                        WHERE o.line_no = u.line_no
                    """)
                    con.unregister("upd")

                # next fetch should begin right after the last flushed line
                cur_start = int(out["line_no"].iloc[-1]) + 1
            else:
                # nothing flushed (tiny last block)
                cur_start = fetch_start + len(df)

            # carry the tail (overlap) forward (already moved)
            carry_df = moved.iloc[flush_n:].copy()

            # tidy memory
            del df, orig, moved
            gc.collect()

        # flush any leftover carried rows (end of file)
        if carry_df is not None and not carry_df.empty:
            con.register("upd_tail", carry_df.loc[:, ["line_no","sent_pt_br","sent_pt_pt"]])
            con.execute("""
                UPDATE opus_moses AS o
                SET sent_pt_br = u.sent_pt_br,
                    sent_pt_pt = u.sent_pt_pt
                FROM upd_tail AS u
                WHERE o.line_no = u.line_no
            """)
            con.unregister("upd_tail")

        # reclaim disk space
        con.execute("CHECKPOINT")

# ==============================
# B) REPEATED-PREFIX CLEANER
# ==============================
def _split_sents(s: str) -> List[str]:
    s = (s or "").strip()
    return [m.group(0).strip() for m in _SENT_RE.finditer(s)]

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def _norm_for_match(s: str) -> str:
    s = _strip_accents(s.lower())
    s = re.sub(r"[^\w]+", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()

def _tokens(s: str) -> List[str]:
    return _norm_for_match(s).split()

def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))

def _looks_like_sentence_start(s: str) -> bool:
    t = (s or "").lstrip()
    while t and t[0] in "«“\"([{'’”»": t = t[1:].lstrip()
    return (not t) or t[0].isupper()

def _adjacent_dedup(sents: List[str], jacc=0.96) -> List[str]:
    out: List[str] = []
    for s in sents:
        if out:
            a = set(_tokens(out[-1])); b = set(_tokens(s))
            if _jaccard(a, b) >= jacc:
                continue
        out.append(s)
    return out

def dedup_repeated_prefix_block(
    br_prev: str, pt_prev: str,
    br_here: str, pt_here: str,
    *,
    prev_window: int = 6,
    min_prefix_tokens: int = 6,
    coverage_thresh: float = 0.92,
    require_both: bool = False,
    collapse_adjacent_dups: bool = True
) -> Tuple[str, str, bool, int]:
    """
    Trim from the START of (br_here, pt_here) the longest sentence-aligned prefix
    whose tokens are largely contained in the TAIL of (br_prev, pt_prev).
    Returns: (br_trimmed, pt_trimmed, applied, n_sentences_removed)
    """
    def _count_to_remove(prev: str, nxt: str) -> int:
        prev_s = _split_sents(prev); nxt_s = _split_sents(nxt)
        if not prev_s or not nxt_s: return 0
        tail = " ".join(prev_s[-prev_window:]) if prev_window > 0 else " ".join(prev_s)
        tail_tok = set(_tokens(tail))
        best_k = 0
        for k in range(1, len(nxt_s) + 1):
            pref = " ".join(nxt_s[:k])
            toks = _tokens(pref)
            if len(toks) < min_prefix_tokens: continue
            cov = len(set(toks) & tail_tok) / max(1, len(set(toks)))
            if cov >= coverage_thresh: best_k = k
        return best_k

    k_br = _count_to_remove(br_prev, br_here)
    k_pt = _count_to_remove(pt_prev, pt_here)
    k = min(k_br, k_pt) if require_both else max(k_br, k_pt)
    if k <= 0: return br_here, pt_here, False, 0

    br_s = _split_sents(br_here)[k:]; pt_s = _split_sents(pt_here)[k:]
    if collapse_adjacent_dups:
        br_s = _adjacent_dedup(br_s); pt_s = _adjacent_dedup(pt_s)

    br_out = " ".join(br_s).strip(); pt_out = " ".join(pt_s).strip()
    if br_out and not _looks_like_sentence_start(br_out): return br_here, pt_here, False, 0
    if pt_out and not _looks_like_sentence_start(pt_out): return br_here, pt_here, False, 0
    return br_out, pt_out, True, k

def run_repeated_prefix_cleaner_chunked(
    *,
    db_path=DB,
    table: str = "opus_moses",
    order_col: str = "line_no",
    text_br_col: str = "sent_pt_br",
    text_pt_col: str = "sent_pt_pt",
    id_pair_col: str = "pair_id",
    # knobs (forwarded)
    prev_window: int = 6,
    min_prefix_tokens: int = 6,
    coverage_thresh: float = 0.92,
    require_both: bool = False,
    collapse_adjacent_dups: bool = True,
    # deletion policy
    delete_on_trigger: bool = True,
    delete_if_empty_only: bool = False,
    # execution
    chunk_size: int = 50_000,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    apply_changes: bool = False,
    print_updates: bool = False,
    trace_lines: Optional[Iterable[int]] = None
):
    """
    Walk rows; if dedup applies (either language unless require_both=True):
      - delete whole row (default), or
      - update with trimmed text.
    Prints deleted (line_no, pair_id). Processes in chunks.
    """
    assert not (delete_on_trigger and delete_if_empty_only), \
        "Choose delete_on_trigger=True OR delete_if_empty_only=True (not both)."

    where = []; args = []
    if start_line is not None: where.append(f"{order_col} >= ?"); args.append(int(start_line))
    if end_line   is not None: where.append(f"{order_col} <= ?"); args.append(int(end_line))
    WHERE = ("WHERE " + " AND ".join(where)) if where else ""

    with duckdb.connect(str(db_path)) as con:
        mn, mx = con.execute(
            f"SELECT min({order_col}), max({order_col}) FROM {table} {WHERE}", args
        ).fetchone()
        if mn is None or mx is None:
            print("No rows match selection."); return

        prev_br, prev_pt = "", ""
        cur = int(mn)
        while cur <= int(mx):
            hi = min(cur + int(chunk_size) - 1, int(mx))
            df = con.execute(f"""
                SELECT {order_col} AS line_no,
                       {id_pair_col} AS pair_id,
                       {text_br_col} AS br,
                       {text_pt_col} AS pt
                FROM {table}
                WHERE {order_col} BETWEEN ? AND ?
                ORDER BY {order_col}
            """, [cur, hi]).df()

            updates = []; deletes = []
            deleted_ids_print = []; updated_ids_print = []

            for i in range(len(df)):
                line_no = int(df.line_no.iloc[i])
                pair_id = int(df.pair_id.iloc[i]) if "pair_id" in df.columns else None
                br_here = df.br.iloc[i] or ""; pt_here = df.pt.iloc[i] or ""

                br_new, pt_new, applied, k = dedup_repeated_prefix_block(
                    prev_br, prev_pt, br_here, pt_here,
                    prev_window=prev_window,
                    min_prefix_tokens=min_prefix_tokens,
                    coverage_thresh=coverage_thresh,
                    require_both=require_both,
                    collapse_adjacent_dups=collapse_adjacent_dups
                )

                if trace_lines and (line_no in set(trace_lines)):
                    print(f"[trace {line_no}] applied={applied} k={k}")

                will_delete = False
                if applied:
                    if delete_on_trigger:
                        will_delete = True
                    elif delete_if_empty_only and (not br_new.strip() and not pt_new.strip()):
                        will_delete = True

                if will_delete:
                    deletes.append((line_no,))
                    deleted_ids_print.append((line_no, pair_id))
                    # don't advance prev_* on deletion (use last kept row)
                else:
                    if applied and (br_new != br_here or pt_new != pt_here):
                        updates.append((br_new, pt_new, line_no))
                        if print_updates: updated_ids_print.append((line_no, pair_id))
                        prev_br, prev_pt = br_new, pt_new
                    else:
                        prev_br, prev_pt = br_here, pt_here

            if apply_changes and (updates or deletes):
                con.execute("BEGIN TRANSACTION")
                if updates:
                    con.executemany(
                        f"UPDATE {table} SET {text_br_col} = ?, {text_pt_col} = ? WHERE {order_col} = ?",
                        updates
                    )
                if deletes:
                    con.executemany(
                        f"DELETE FROM {table} WHERE {order_col} = ?",
                        deletes
                    )
                con.execute("COMMIT")

            print(f"[{cur}..{hi}] updates={len(updates)} deletes={len(deletes)}")
            if deleted_ids_print:
                print("  Deleted rows (line_no, pair_id):")
                for j in range(0, len(deleted_ids_print), 1000):
                    block = deleted_ids_print[j:j+1000]
                    print("   ", ", ".join(f"({ln},{pid})" for ln,pid in block))
            if print_updates and updated_ids_print:
                print("  Updated rows (line_no, pair_id):")
                for j in range(0, len(updated_ids_print), 1000):
                    block = updated_ids_print[j:j+1000]
                    print("   ", ", ".join(f"({ln},{pid})" for ln,pid in block))

            cur = hi + 1

import re, gc, duckdb, pandas as pd
from typing import List, Tuple

# ---------------------------
# Small utilities / schema
# ---------------------------
def _as_int_or_none(x):
    if x is None: return None
    try:
        if pd.isna(x): return None
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        try:
            return int(str(x).strip())
        except Exception:
            return None

def _ensure_opus_ops_tables(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS opus_ops_update(
            line_no BIGINT PRIMARY KEY,
            sent_pt_br TEXT,
            sent_pt_pt TEXT
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS opus_ops_delete(
            line_no BIGINT PRIMARY KEY
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS opus_ops_progress(
            done_through BIGINT
        )
    """)
    if con.execute("SELECT COUNT(*) FROM opus_ops_progress").fetchone()[0] == 0:
        con.execute("INSERT INTO opus_ops_progress VALUES (0)")

def _ensure_simple_filter_table(con: duckdb.DuckDBPyConnection):
    con.execute("""
        CREATE TABLE IF NOT EXISTS opus_filter_simple(
            line_no BIGINT PRIMARY KEY,
            pair_id BIGINT,
            reason  TEXT,
            base_sim DOUBLE,
            thr     DOUBLE,
            L       INTEGER
        )
    """)

# ---------------------------
# A) PREPROCESSING
# ---------------------------
def plan_ops_over_corpus(block_size=50_000, reset=False, *, db_path=None):
    with duckdb.connect(str(db_path)) as con:
        _ensure_opus_ops_tables(con)
        lo, hi = con.execute("SELECT MIN(line_no), MAX(line_no) FROM opus_moses").fetchone()
        lo, hi = int(lo), int(hi)

        if reset:
            con.execute("DELETE FROM opus_ops_update")
            con.execute("DELETE FROM opus_ops_delete")
            con.execute("DELETE FROM opus_ops_progress")
            con.execute("INSERT INTO opus_ops_progress VALUES (0)")

        done = int(con.execute("SELECT done_through FROM opus_ops_progress").fetchone()[0])
        cur  = max(lo, done + 1)

        carry = None
        last_br, last_pt = None, None

        while cur <= hi:
            win = min(block_size, hi - cur + 1)
            df = con.execute("""
                SELECT line_no, pair_id, sent_pt_br, sent_pt_pt
                FROM opus_moses
                WHERE line_no BETWEEN ? AND ?
                ORDER BY line_no
            """, [cur, cur+win-1]).df()

            rows = df.to_dict("records")
            if carry is not None:
                rows = [carry] + rows
                carry = None

            updates, deletes = [], []
            i, n = 0, len(rows)
            while i < n:
                base = rows[i]; i += 1
                br = _normalize_line_text(base.get("sent_pt_br"))
                pt = _normalize_line_text(base.get("sent_pt_pt"))
                group_lines = [int(base["line_no"])]

                while i < n:
                    nxt = rows[i]
                    br2 = _normalize_line_text(nxt.get("sent_pt_br"))
                    pt2 = _normalize_line_text(nxt.get("sent_pt_pt"))
                    if _should_merge_pair(br, br2, pt, pt2):
                        br = _join_text(br, br2)
                        pt = _join_text(pt, pt2)
                        group_lines.append(int(nxt["line_no"]))
                        i += 1
                    else:
                        break

                if i >= n:
                    carry = {
                        "line_no": group_lines[0],
                        "pair_id": _as_int_or_none(base.get("pair_id")),
                        "sent_pt_br": br,
                        "sent_pt_pt": pt
                    }
                    break

                if (br.strip() == "") or (pt.strip() == ""):
                    for ln in group_lines:
                        deletes.append({"line_no": int(ln)})
                    last_br, last_pt = None, None
                    continue

                if last_br and _last_char(last_br) in HARD_STOPS:
                    br = _capitalize_first_alpha(br)
                if last_pt and _last_char(last_pt) in HARD_STOPS:
                    pt = _capitalize_first_alpha(pt)

                head = group_lines[0]
                if br != base.get("sent_pt_br") or pt != base.get("sent_pt_pt") or len(group_lines) > 1:
                    updates.append({"line_no": int(head), "sent_pt_br": br, "sent_pt_pt": pt})
                for ln in group_lines[1:]:
                    deletes.append({"line_no": int(ln)})

                last_br, last_pt = br, pt

            if updates:
                con.register("upd", pd.DataFrame(updates))
                con.execute("""
                    INSERT INTO opus_ops_update (line_no, sent_pt_br, sent_pt_pt)
                    SELECT line_no, sent_pt_br, sent_pt_pt FROM upd
                    ON CONFLICT(line_no) DO UPDATE SET
                        sent_pt_br = EXCLUDED.sent_pt_br,
                        sent_pt_pt = EXCLUDED.sent_pt_pt
                """)
                con.unregister("upd")
            if deletes:
                con.register("del", pd.DataFrame(deletes))
                con.execute("""
                    INSERT INTO opus_ops_delete (line_no)
                    SELECT DISTINCT line_no FROM del
                    ON CONFLICT(line_no) DO NOTHING
                """)
                con.unregister("del")

            del df, rows, updates, deletes
            gc.collect()

            cur += win
            con.execute("UPDATE opus_ops_progress SET done_through = ?", [cur - 1])

        if carry is not None:
            if last_br and _last_char(last_br) in HARD_STOPS:
                carry["sent_pt_br"] = _capitalize_first_alpha(carry["sent_pt_br"])
            if last_pt and _last_char(last_pt) in HARD_STOPS:
                carry["sent_pt_pt"] = _capitalize_first_alpha(carry["sent_pt_pt"])
            con.register("tail_upd", pd.DataFrame([{
                "line_no": int(carry["line_no"]),
                "sent_pt_br": carry["sent_pt_br"],
                "sent_pt_pt": carry["sent_pt_pt"],
            }]))
            con.execute("""
                INSERT INTO opus_ops_update (line_no, sent_pt_br, sent_pt_pt)
                SELECT line_no, sent_pt_br, sent_pt_pt FROM tail_upd
                ON CONFLICT(line_no) DO UPDATE SET
                    sent_pt_br = EXCLUDED.sent_pt_br,
                    sent_pt_pt = EXCLUDED.sent_pt_pt
            """)
            con.unregister("tail_upd")

def apply_ops_ctas_swap(*, db_path=None, force_checkpoint=True):
    with duckdb.connect(str(db_path)) as con:
        try: con.execute("ROLLBACK")
        except: pass

        con.execute("""
            DELETE FROM opus_ops_delete
            WHERE line_no IN (SELECT line_no FROM opus_ops_update)
        """)

        con.execute("BEGIN")
        try:
            con.execute("DROP TABLE IF EXISTS opus_moses_new")
            con.execute("""
                CREATE TABLE opus_moses_new AS
                SELECT
                    o.line_no,
                    o.pair_id,
                    COALESCE(u.sent_pt_br, o.sent_pt_br) AS sent_pt_br,
                    COALESCE(u.sent_pt_pt, o.sent_pt_pt) AS sent_pt_pt
                FROM opus_moses o
                LEFT JOIN opus_ops_update u USING (line_no)
                WHERE o.line_no NOT IN (SELECT line_no FROM opus_ops_delete)
                ORDER BY o.line_no
            """)
            con.execute("DROP TABLE opus_moses")
            con.execute("ALTER TABLE opus_moses_new RENAME TO opus_moses")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS opus_moses_line_pk ON opus_moses(line_no)")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS opus_moses_pair_uq  ON opus_moses(pair_id)")
            con.execute("COMMIT")
        except:
            con.execute("ROLLBACK"); raise

        if force_checkpoint:
            con.execute("FORCE CHECKPOINT")

# ---------------------------
# B) SIMPLE ALIGNMENT FILTER
# ---------------------------
ALIGN_METHOD = "inter"  # alternatives: "inter", "mwmf", "itermax", "union"

WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|\d+", re.UNICODE)
PT_STOPWORDS = {
    "a","à","ao","aos","as","o","os","um","uma","de","da","das","do","dos","em","no","na","nos","nas",
    "e","ou","que","se","por","para","com","como","mais","mas","não","sim","já","sua","seu","suas","seus",
    "eu","tu","ele","ela","nós","vós","eles","elas","me","te","lhe","nos","vos","lhes","isso","isto","aquilo"
}

def tokenize(s: str) -> List[str]:
    return WORD.findall(s or "")

def ali_tokenize(s: str) -> List[str]:
    return tokenize(s)

def is_content_token(tok: str) -> bool:
    t = (tok or "").lower()
    return (t not in PT_STOPWORDS) and (len(t) > 1)

try:
    from simalign import SentenceAligner
    _ALIGNER = SentenceAligner(model="xlmr", token_type="word", matching_methods="a")
except Exception as e:
    raise RuntimeError(
        "SimAlign required. Install:\n"
        "  pip install simalign torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n"
        f"Import error: {e}"
    )

def _raw_pairs(l_tokens, r_tokens, method: str = ALIGN_METHOD):
    """
    Ask SimAlign for alignments, but never crash.
    - Catches SimAlign internal errors (IndexError, ValueError, etc.)
    - Handles missing keys ('itermax'/'inter'/'mwmf') and falls back.
    """
    try:
        out = _ALIGNER.get_word_aligns(l_tokens, r_tokens)  # dict-like
    except Exception:
        return []

    if not isinstance(out, dict):
        return []

    # normalize keys robustly
    keys = {k.lower(): k for k in out.keys()}
    def get(name: str):
        return out.get(keys.get(name.lower()), [])

    if method.lower() == "union":
        return list({*get("inter"), *get("itermax"), *get("mwmf")})

    pairs = get(method)
    if pairs:
        return pairs

    # graceful fallbacks
    for alt in ("itermax", "inter", "mwmf"):
        pairs = get(alt)
        if pairs:
            return pairs
    return []


def _form_norm(t: str) -> str:
    import unicodedata as ud
    return ud.normalize("NFC", t or "").casefold()

def _type_sim_from_pairs(L_tokens, R_tokens, pairs):
    Ln = [_form_norm(t) for t in L_tokens]
    Rn = [_form_norm(t) for t in R_tokens]
    union = set(Ln) | set(Rn)
    if not union: return 1.0
    aligned = set()
    for i, j in pairs:
        if 0 <= i < len(Ln): aligned.add(Ln[i])
        if 0 <= j < len(Rn): aligned.add(Rn[j])
    return len(aligned) / len(union)

def _smooth_small_gaps(covered: set, n_tokens: int, max_gap: int = 1):
    C = set(covered); i = 0
    while i < n_tokens:
        if i not in C:
            j = i
            while j < n_tokens and j not in C: j += 1
            gap = j - i
            if 0 < gap <= max_gap and i > 0 and j < n_tokens:
                for k in range(i, j): C.add(k)
            i = j
        else:
            i += 1
    return C

def _interior_uncovered(tokens: List[str], covered: set) -> List[Tuple[int,int]]:
    runs, cur = [], []
    for i in range(len(tokens)):
        if i not in covered: cur.append(i)
        elif cur: runs.append((cur[0], cur[-1])); cur=[]
    if cur: runs.append((cur[0], cur[-1]))
    n = len(tokens)
    return [(i0,i1) for (i0,i1) in runs if i0>0 and i1<n-1]

def _content_count(tokens: List[str]) -> int:
    return sum(1 for t in tokens if is_content_token(t))

def _type_jaccard(L_tokens, R_tokens) -> float:
    Ln = {_form_norm(t) for t in L_tokens}
    Rn = {_form_norm(t) for t in R_tokens}
    U = Ln | Rn
    return (len(Ln & Rn) / len(U)) if U else 1.0

def new_sim(br: str, pt: str, method: str = ALIGN_METHOD) -> float:
    L = ali_tokenize(br or ""); R = ali_tokenize(pt or "")
    if not L and not R: return 1.0
    if not L or not R:  return 0.0
    try:
        pairs = _raw_pairs(L, R, method=method)
    except Exception:
        pairs = []
    if not pairs:
        return _type_jaccard(L, R)
    return _type_sim_from_pairs(L, R, pairs)


def alignment_quality_features(
    br_prev: str, br_here: str, br_next: str,
    pt_prev: str, pt_here: str, pt_next: str,
    *, use_window: bool = True, sim_fn=None
) -> dict:
    if sim_fn is None:
        sim_fn = new_sim

    br_toks = ali_tokenize(br_here); pt_toks = ali_tokenize(pt_here)
    if use_window:
        pt_win = ali_tokenize(" ".join(x for x in [pt_prev, pt_here, pt_next] if x))
        br_pairs = _raw_pairs(br_toks, pt_win, method=ALIGN_METHOD)
        br_cov = {i for i,_ in br_pairs}
        br_win = ali_tokenize(" ".join(x for x in [br_prev, br_here, br_next] if x))
        pt_pairs = _raw_pairs(pt_toks, br_win, method=ALIGN_METHOD)
        pt_cov = {i for i,_ in pt_pairs}
    else:
        pairs  = _raw_pairs(br_toks, pt_toks, method=ALIGN_METHOD)
        br_cov = {i for i,_ in pairs}
        pt_cov = {j for _,j in pairs}

    br_cov = _smooth_small_gaps(br_cov, len(br_toks), 1)
    pt_cov = _smooth_small_gaps(pt_cov, len(pt_toks), 1)

    br_len = len(br_toks); pt_len = len(pt_toks)
    br_cov_ratio = len(br_cov) / max(1, br_len)
    pt_cov_ratio = len(pt_cov) / max(1, pt_len)
    cov_min = min(br_cov_ratio, pt_cov_ratio)
    cov_gap = abs(br_cov_ratio - pt_cov_ratio)

    br_int_spans = _interior_uncovered(br_toks, br_cov)
    pt_int_spans = _interior_uncovered(pt_toks, pt_cov)

    def _content_in_runs(tokens, runs):
        return sum(1 for i0,i1 in runs for t in tokens[i0:i1+1] if is_content_token(t))

    br_ct = _content_count(br_toks)
    pt_ct = _content_count(pt_toks)
    br_int_ratio = _content_in_runs(br_toks, br_int_spans) / max(1, br_ct)
    pt_int_ratio = _content_in_runs(pt_toks, pt_int_spans) / max(1, pt_ct)

    base_sim = float(new_sim(br_here, pt_here, method=ALIGN_METHOD))

    return {
        "br_cov": br_cov_ratio, "pt_cov": pt_cov_ratio,
        "cov_min": cov_min, "cov_gap": cov_gap,
        "br_int_content_ratio": br_int_ratio, "pt_int_content_ratio": pt_int_ratio,
        "br_content_total": br_ct, "pt_content_total": pt_ct,
        "br_len": br_len, "pt_len": pt_len,
        "base_sim": base_sim
    }

def alignment_similarity_only_flag(
    feats: dict,
    *,
    min_sim_ok: float | None = None,
    length_mode: str = "total",
    base_min_sim: float = 0.30,
    long_min_sim: float = 0.70,
    short_len: int = 8,
    long_len: int = 28,
    **kwargs,
) -> Tuple[bool,str,float,int]:
    sim = float(feats["base_sim"])

    if min_sim_ok is not None:
        thr = float(min_sim_ok)
        L = max(int(feats.get("br_len", 0)), int(feats.get("pt_len", 0)))
    else:
        if length_mode == "total":
            L = max(int(feats.get("br_len", 0)), int(feats.get("pt_len", 0)))
        elif length_mode == "char":
            L = max(int(feats.get("br_len", 0)), int(feats.get("pt_len", 0)))
        else:  # "content"
            L = max(int(feats.get("br_content_total", 0)), int(feats.get("pt_content_total", 0)))

        if L <= short_len:
            thr = base_min_sim
        elif L >= long_len:
            thr = long_min_sim
        else:
            t = (L - short_len) / max(1, (long_len - short_len))
            thr = base_min_sim + t * (long_min_sim - base_min_sim)

    activate = sim < thr
    reason = (f"low_similarity@simple(thr={thr:.2f},L={L})" if activate else "ok@simple")
    return activate, reason, thr, L

# ---------------------------
# C) Simple filter across the corpus
# ---------------------------
def build_simple_filter_flags_chunked(
    *,
    db_path,
    chunk_size: int = 50_000,
    use_window: bool = True,
    thresholds: dict | None = None,
):
    if thresholds is None:
        thresholds = dict(min_sim_ok=None, length_mode="total",
                          base_min_sim=0.30, long_min_sim=0.70,
                          short_len=8, long_len=28)

    with duckdb.connect(str(db_path)) as con:
        _ensure_simple_filter_table(con)
        mn, mx = con.execute("SELECT MIN(line_no), MAX(line_no) FROM opus_moses").fetchone()
        if mn is None or mx is None:
            print("[simple-filter] opus_moses empty; nothing to do.")
            return
        mn, mx = int(mn), int(mx)

        cur = mn
        total_flagged = 0
        while cur <= mx:
            hi = min(cur + chunk_size - 1, mx)

            df = con.execute("""
                SELECT line_no, pair_id, sent_pt_br, sent_pt_pt
                FROM opus_moses
                WHERE line_no BETWEEN ? AND ?
                ORDER BY line_no
            """, [max(mn, cur-1), min(mx, hi+1)]).df()

            rows = []
            n = len(df)
            for idx in range(n):
                line_no = int(df.line_no.iloc[idx])
                if line_no < cur or line_no > hi:
                    continue
                br_prev = df.sent_pt_br.iloc[idx-1] if idx > 0 else ""
                br_here = df.sent_pt_br.iloc[idx]
                br_next = df.sent_pt_br.iloc[idx+1] if idx+1 < n else ""
                pt_prev = df.sent_pt_pt.iloc[idx-1] if idx > 0 else ""
                pt_here = df.sent_pt_pt.iloc[idx]
                pt_next = df.sent_pt_pt.iloc[idx+1] if idx+1 < n else ""

                feats = alignment_quality_features(
                    br_prev, br_here, br_next,
                    pt_prev, pt_here, pt_next,
                    use_window=use_window, sim_fn=new_sim
                )
                activate, reason, thr, L = alignment_similarity_only_flag(feats, **thresholds)
                if activate:
                    rows.append({
                        "line_no": line_no,
                        "pair_id": _as_int_or_none(df.pair_id.iloc[idx]),
                        "reason": reason,
                        "base_sim": float(feats["base_sim"]),
                        "thr": float(thr),
                        "L": int(L),
                    })

            if rows:
                con.register("flags", pd.DataFrame(rows))
                con.execute("""
                    INSERT INTO opus_filter_simple (line_no, pair_id, reason, base_sim, thr, L)
                    SELECT line_no, pair_id, reason, base_sim, thr, L FROM flags
                    ON CONFLICT(line_no) DO UPDATE SET
                        pair_id = EXCLUDED.pair_id,
                        reason  = EXCLUDED.reason,
                        base_sim= EXCLUDED.base_sim,
                        thr     = EXCLUDED.thr,
                        L       = EXCLUDED.L
                """)
                con.unregister("flags")
                total_flagged += len(rows)

            print(f"[simple-filter] [{cur}..{hi}] flagged={len(rows)} (cum={total_flagged})")
            cur = hi + 1

        print(f"[simple-filter] DONE. total flagged lines: {total_flagged}")

def apply_simple_filter_ctas_swap(*, db_path, drop_flags=False):
    with duckdb.connect(str(db_path)) as con:
        con.execute("BEGIN")
        try:
            con.execute("DROP TABLE IF EXISTS opus_moses_new")
            con.execute("""
                CREATE TABLE opus_moses_new AS
                SELECT o.*
                FROM opus_moses o
                LEFT JOIN opus_filter_simple f USING (line_no)
                WHERE f.line_no IS NULL
                ORDER BY o.line_no
            """)
            con.execute("DROP TABLE opus_moses")
            con.execute("ALTER TABLE opus_moses_new RENAME TO opus_moses")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS opus_moses_line_pk ON opus_moses(line_no)")
            con.execute("CREATE UNIQUE INDEX IF NOT EXISTS opus_moses_pair_uq  ON opus_moses(pair_id)")
            con.execute("COMMIT")
        except:
            con.execute("ROLLBACK"); raise

        con.execute("FORCE CHECKPOINT")
        kept = con.execute("SELECT COUNT(*) FROM opus_moses").fetchone()[0]
        removed = con.execute("SELECT COUNT(*) FROM opus_filter_simple").fetchone()[0]
        print(f"[simple-filter] applied. kept={kept:,} removed={removed:,}")
        if drop_flags:
            con.execute("DROP TABLE opus_filter_simple")
            con.execute("FORCE CHECKPOINT")

# ---------------------------
# D) One-call convenience runner (with simple reset)
# ---------------------------
def run_opus_pipeline_simple(
    *,
    db_path,
    block_size: int = 50_000,
    neighbor_overlap: int = 3,
    neighbor_margin: float = 0.04,
    neighbor_max_clause_chars: int = 60,
    neighbor_max_iters: int = 5,
    dedup_chunk_size: int = 50_000,
    delete_on_trigger: bool = True,
    filter_chunk_size: int = 50_000,
    filter_use_window: bool = True,
    filter_thresholds: dict | None = None,
    apply_filter: bool = True,
    reset: bool = False,          # <<< simple switch
):
    # reset planning progress + clear flags if asked
    if reset:
        with duckdb.connect(str(db_path)) as con:
            _ensure_opus_ops_tables(con)
            _ensure_simple_filter_table(con)
            con.execute("DELETE FROM opus_ops_update")
            con.execute("DELETE FROM opus_ops_delete")
            con.execute("DELETE FROM opus_ops_progress")
            con.execute("INSERT INTO opus_ops_progress VALUES (0)")
            con.execute("DELETE FROM opus_filter_simple")
        print("[pipeline] reset: cleared ops progress and flags.")

    # 1) plan + swap (merge lines, clean)
    plan_ops_over_corpus(block_size=block_size, reset=False, db_path=db_path)
    apply_ops_ctas_swap(db_path=db_path)

    # 2) neighbor moves (assumes your function exists)
    apply_neighbor_moves_corpus_inplace(
        block_size=block_size,
        overlap=neighbor_overlap,
        margin=neighbor_margin,
        max_clause_chars=neighbor_max_clause_chars,
        max_iters=neighbor_max_iters,
    )

    # 3) repeated-prefix cleaner (assumes your function exists)
    run_repeated_prefix_cleaner_chunked(
        db_path=db_path,
        prev_window=6,
        min_prefix_tokens=6,
        coverage_thresh=0.92,
        require_both=False,
        collapse_adjacent_dups=True,
        delete_on_trigger=delete_on_trigger,
        delete_if_empty_only=False,
        chunk_size=dedup_chunk_size,
        start_line=None, end_line=None,
        apply_changes=True,
        print_updates=False,
        trace_lines=None
    )

    # 4) simple alignment filter → write flags
    build_simple_filter_flags_chunked(
        db_path=db_path,
        chunk_size=filter_chunk_size,
        use_window=filter_use_window,
        thresholds=filter_thresholds
    )

    # 5) apply filter (CTAS swap) if requested
    if apply_filter:
        apply_simple_filter_ctas_swap(db_path=db_path, drop_flags=False)

    print("[pipeline] complete.")

import os, sys, subprocess, math, duckdb, pandas as pd, pathlib

def _shard_ranges(lo, hi, n):
    size = hi - lo + 1
    base, rem = divmod(size, n)
    out = []
    s = lo
    for i in range(n):
        e = s + base - 1 + (1 if i < rem else 0)
        out.append((s, e))
        s = e + 1
    return out

def build_simple_filter_flags_chunked_to_parquet(
    *, db_path, out_parquet, chunk_size=50_000, use_window=True, thresholds=None,
    start_line=None, end_line=None
):
    """
    Same logic as your build_simple_filter_flags_chunked, but COLLECTS rows and writes Parquet.
    Relies on your already-defined: alignment_quality_features + alignment_similarity_only_flag.
    """
    if thresholds is None:
        thresholds = dict(min_sim_ok=None, length_mode="total",
                          base_min_sim=0.30, long_min_sim=0.70,
                          short_len=8, long_len=28)

    con = duckdb.connect(str(db_path))
    where, args = [], []
    if start_line is not None: where.append("line_no >= ?"); args.append(int(start_line))
    if end_line   is not None: where.append("line_no <= ?"); args.append(int(end_line))
    WHERE = ("WHERE " + " AND ".join(where)) if where else ""

    mn, mx = con.execute(
        f"SELECT MIN(line_no), MAX(line_no) FROM opus_moses {WHERE}", args
    ).fetchone()
    if mn is None or mx is None:
        print("[simple-filter] nothing to do in this shard"); con.close(); return

    cur = int(mn); mx = int(mx)
    rows = []

    while cur <= mx:
        hi = min(cur + int(chunk_size) - 1, mx)
        # grab with context rows (prev,next) to build window
        df = con.execute("""
            SELECT line_no, pair_id, sent_pt_br AS br, sent_pt_pt AS pt
            FROM opus_moses
            WHERE line_no BETWEEN ? AND ?
            ORDER BY line_no
        """, [max(int(mn), cur-1), min(mx, hi+1)]).df()

        n = len(df)
        for idx in range(n):
            line_no = int(df.line_no.iloc[idx])
            if line_no < cur or line_no > hi: 
                continue

            br_prev = df.br.iloc[idx-1] if idx > 0 else ""
            br_here = df.br.iloc[idx]
            br_next = df.br.iloc[idx+1] if idx+1 < n else ""
            pt_prev = df.pt.iloc[idx-1] if idx > 0 else ""
            pt_here = df.pt.iloc[idx]
            pt_next = df.pt.iloc[idx+1] if idx+1 < n else ""

            feats = alignment_quality_features(
                br_prev, br_here, br_next,
                pt_prev, pt_here, pt_next,
                use_window=use_window, sim_fn=new_sim  # your new_sim
            )
            activate, reason, thr, L = alignment_similarity_only_flag(feats, **thresholds)
            if activate:
                rows.append({
                    "line_no": line_no,
                    "pair_id": int(df.pair_id.iloc[idx]) if pd.notna(df.pair_id.iloc[idx]) else None,
                    "reason": reason,
                    "base_sim": float(feats["base_sim"]),
                    "thr": float(thr),
                    "L": int(L),
                })

        print(f"[{cur}..{hi}] flagged={sum(1 for r in rows if cur<=r['line_no']<=hi)}")
        cur = hi + 1

    con.close()
    out = pathlib.Path(out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print("wrote shard parquet:", out)

def launch_parallel_filter_to_parquet(db_path, gpu_ids, out_dir, use_window=False, thresholds=None):
    """
    One subprocess per GPU, each writing a shard parquet to out_dir.
    """
    with duckdb.connect(str(db_path)) as con:
        lo, hi = con.execute("SELECT MIN(line_no), MAX(line_no) FROM opus_moses").fetchone()
    lo, hi = int(lo), int(hi)
    shards = _shard_ranges(lo, hi, len(gpu_ids))
    print(f"Shards: {shards}")

    procs = []
    for (s,e), gpu in zip(shards, gpu_ids):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        code = (
            "from __main__ import build_simple_filter_flags_chunked_to_parquet; "
            f"build_simple_filter_flags_chunked_to_parquet(db_path={db_path!r}, "
            f"out_parquet={str(pathlib.Path(out_dir)/f'shard_{s}_{e}.parquet')!r}, "
            f"chunk_size=50000, use_window={bool(use_window)}, thresholds={thresholds!r}, "
            f"start_line={s}, end_line={e})"
        )
        procs.append(subprocess.Popen([sys.executable, "-u", "-c", code], env=env))
    for p in procs: p.wait()
    print("All shard processes finished.")

