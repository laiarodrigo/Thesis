# src/shard_filter.py
import os, sys, json, re, time, signal
import duckdb
import pandas as pd

# --- tiny tokenizer ---
WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+(?:[-'][A-Za-zÀ-ÖØ-öø-ÿ]+)*|\d+", re.UNICODE)
def ali_tokens(s: str): return WORD.findall(s or "")

def _norm(t: str):
    import unicodedata as ud
    return ud.normalize("NFC", t or "").casefold()

def _type_sim_from_pairs(L, R, pairs):
    Ln = [_norm(t) for t in L]; Rn = [_norm(t) for t in R]
    U = set(Ln) | set(Rn)
    if not U: return 1.0
    aligned = set()
    for i, j in pairs:
        if 0 <= i < len(Ln): aligned.add(Ln[i])
        if 0 <= j < len(Rn): aligned.add(Rn[j])
    return len(aligned) / len(U)

# --- SimAlign wrapper ---
def _build_aligner(device: str):
    from simalign import SentenceAligner
    try:
        return SentenceAligner(model="xlmr", token_type="word",
                               matching_methods="a", device=device)
    except TypeError:
        # older simalign doesn't accept device=
        return SentenceAligner(model="xlmr", token_type="word",
                               matching_methods="a")

def _raw_pairs(L, R, ALIGNER):
    try:
        out = ALIGNER.get_word_aligns(L, R)
    except Exception:
        return []
    for key in ("inter","itermax","mwmf"):
        if key in out and out[key]:
            return out[key]
    return []

def new_sim(a: str, b: str, ALIGNER) -> float:
    L = ali_tokens(a); R = ali_tokens(b)
    if not L and not R: return 1.0
    if not L or not R:  return 0.0
    pairs = _raw_pairs(L, R, ALIGNER)
    if not pairs:
        Ln = {_norm(t) for t in L}; Rn = {_norm(t) for t in R}; U = Ln | Rn
        return (len(Ln & Rn) / len(U)) if U else 1.0
    return _type_sim_from_pairs(L, R, pairs)

def length_adaptive_flag(sim: float, L: int, base_min=0.30, long_min=0.70, short_len=8, long_len=28):
    if L <= short_len: thr = base_min
    elif L >= long_len: thr = long_min
    else:
        t = (L - short_len) / max(1, (long_len - short_len))
        thr = base_min + t * (long_min - base_min)
    return sim < thr, thr

def main():
    if len(sys.argv) != 6:
        print("usage: shard_filter.py DB_PATH START_LINE END_LINE OUT_PARQUET THRESHOLDS_JSON", file=sys.stderr)
        sys.exit(2)

    db_path, start_line, end_line, out_parquet = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    thr = json.loads(sys.argv[5])

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    # choose device (we’ll pass FORCE_CPU=1 from the launcher)
    try:
        import torch
        if os.environ.get("FORCE_CPU") == "1":
            device = "cpu"
            # let each process use the threads the parent set (or default 1)
            try: torch.set_num_threads(int(os.environ.get("TORCH_CPU_THREADS","1")))
            except Exception: pass
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    ALIGNER = _build_aligner(device)
    print(f"[shard {start_line}-{end_line}] device={device}", flush=True)

    con = duckdb.connect(db_path, read_only=True)
    mn, mx = con.execute("SELECT MIN(line_no), MAX(line_no) FROM opus_moses").fetchone()
    start_line = max(int(mn), start_line); end_line = min(int(mx), end_line)

    total = end_line - start_line + 1
    processed = 0
    flagged = 0
    last_print = 0
    PROG_EVERY = 50_000
    t0 = time.time()

    rows, cur, CHUNK = [], start_line, 50_000
    interrupted = False
    def _sigint(_s,_f):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, _sigint)

    try:
        while cur <= end_line and not interrupted:
            hi = min(cur + CHUNK - 1, end_line)
            df = con.execute("""
                SELECT line_no, pair_id, sent_pt_br AS br, sent_pt_pt AS pt
                FROM opus_moses
                WHERE line_no BETWEEN ? AND ?
                ORDER BY line_no
            """, [cur, hi]).df()

            for i in range(len(df)):
                ln  = int(df.line_no.iloc[i])
                brh = df.br.iloc[i]
                pth = df.pt.iloc[i]

                sim = new_sim(brh, pth, ALIGNER)
                L   = max(len(ali_tokens(brh)), len(ali_tokens(pth)))
                bad, thrv = length_adaptive_flag(sim, L,
                    base_min=thr["base_min_sim"], long_min=thr["long_min_sim"],
                    short_len=thr["short_len"], long_len=thr["long_len"])
                if bad:
                    pid = df.pair_id.iloc[i]
                    rows.append({
                        "line_no": ln,
                        "pair_id": int(pid) if pd.notna(pid) else None,
                        "reason": f"low_similarity@simple(thr={thrv:.2f},L={L})",
                        "base_sim": float(sim),
                        "thr": float(thrv),
                        "L": int(L),
                    })
                    flagged += 1

            processed += len(df)
            cur = hi + 1

            if (processed - last_print) >= PROG_EVERY or processed == total:
                spd = processed / max(1e-6, (time.time()-t0))
                pct = 100.0 * processed / max(1, total)
                print(f"[progress {start_line}-{end_line}] {processed}/{total} ({pct:.2f}%)  "
                      f"~{int(spd)} rows/s  flagged={flagged}", flush=True)
                last_print = processed
    finally:
        con.close()

    tmp = out_parquet + ".tmp"
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    os.replace(tmp, out_parquet)
    print(f"[done {start_line}-{end_line}] wrote {out_parquet} rows {len(rows)}", flush=True)

if __name__ == "__main__":
    main()
