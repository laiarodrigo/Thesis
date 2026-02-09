#!/usr/bin/env python3
# scripts/ptbrvarid/filter_ptbrvarid.py
import argparse
import hashlib
import os
import re
from pathlib import Path

import duckdb
import numpy as np
from bs4 import BeautifulSoup
import nltk
from cleantext import clean

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "data/duckdb/subs.duckdb"
BATCH_ROWS = 20_000
NLTK_USER_DIR = os.path.expanduser("~/nltk_data")

# ----------------------------
# NLTK setup
# ----------------------------
if NLTK_USER_DIR not in nltk.data.path:
    nltk.data.path.append(NLTK_USER_DIR)


def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt/portuguese.pickle")
    except LookupError:
        nltk.download("punkt", download_dir=NLTK_USER_DIR, quiet=True)
        try:
            nltk.data.find("tokenizers/punkt/portuguese.pickle")
        except LookupError:
            nltk.download("punkt_tab", download_dir=NLTK_USER_DIR, quiet=True)
            nltk.data.find("tokenizers/punkt/portuguese.pickle")


# ----------------------------
# Author-style regex + helpers
# ----------------------------
HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#…])*")
HASHTAG_RE = re.compile(r"#(\w+)")
QUOTE_SPACE_START_RE = re.compile(r"^\"\s")
QUOTE_SPACE_END_RE = re.compile(r"\s\"$")
MENTION_RE = re.compile(r"@(\w+)")
RETWEET_RE = re.compile(r"RT @(\w+):")
COD_RE = re.compile(r"COD _ (\w+) ")
BULLET_RE = re.compile(r"^(\d)+.\s")
THREE_DASH_RE = re.compile(r"---.*---")
MORE_THAN_THREE_POINTS_RE = re.compile(r"\.{4,}")

VALID_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzàáâãåāèéêëěėēîïíìįīĵłñńôöòóōõšśûüùúūÿýźçćčñń!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~«»“”ºª€ \t\n\r\x0b\x0c"

INVALID_START = [
    "List of recent changes", "Sort by", "Home |", "> Home", "useful tips", "Licenses:", "Search in: ",
    "Terms of Use - ", "Home page", "Home Page", "Copyright", "Results/Page",
    "!", "#", "$", "%", "&", "*", "+",
    ",", "-", ".", "/", ":", ";", "<", "=",
    ">", "?", "@", "[", "\\", "]", "^", "_", "`", "{", "|", "}", "~",
]
INVALID_MIDDLE = [" @ ", " / ", " | ", "[...]", "(...)"]
INVALID_END = [" ("]

MONTHS = ["january","february","march","april","may","june","july","august","september","october","november","december"]

SPEAKER_LABEL_SINGLE_LETTER = re.compile(
    r'(?m)(^|\s|[(\["“])'          # boundary to preserve
    r'[A-Za-zÀ-ÖØ-öø-ÿ]\.'           # single letter + dot
    r'\s*(?:--|[-–—]{1,2})\s*'
)
SPEAKER_LABEL_WORD_DASH = re.compile(
    r'(?m)(^|\s)[A-Za-zÀ-ÖØ-öø-ÿ]+(?:\.)?\s+--\s*'
)
DIALOGUE_LEADING_DASH = re.compile(
    r'(?m)(^|[\.!\?\:\;…])\s*--\s*'
)
CAP_SENT_START = re.compile(
    r'(?m)(^|[\.!\?\:\;…]\s+)([\"\'“”«»\(\[\{]*)([a-zà-öø-ÿ])'
)


def remove_html_tags(text):
    return BeautifulSoup(text, "html.parser").get_text()


def remove_hashtags(text):
    return HASHTAG_RE.sub("", text).strip()


def remove_mentions(text):
    return MENTION_RE.sub("", text).strip()


def remove_retweets(text):
    return RETWEET_RE.sub("", text).strip()


def remove_urls(text):
    return URL_RE.sub("", text).strip()


def remove_cod_literature(text):
    return COD_RE.sub("", text).strip()


def remove_bullets(text):
    return BULLET_RE.sub("", text).strip()


def remove_three_dashes(text):
    return THREE_DASH_RE.sub("", text).strip()


def remove_quote_space_start(text):
    return QUOTE_SPACE_START_RE.sub('"', text)


def remove_quote_space_end(text):
    if text.endswith(' "'):
        return text[:-2] + '"'
    return text


def has_more_than_three_points(text):
    return bool(MORE_THAN_THREE_POINTS_RE.search(text))


def starts_with_month(text):
    return text.lower().startswith(tuple(MONTHS))


def has_too_long_word(text):
    return any(word for word in text.split(" ") if len(word) > 20)


def has_invalid_start(text):
    return text.startswith(tuple(INVALID_START))


def has_invalid_middle(text):
    return any(True for word in INVALID_MIDDLE if word in text)


def has_invalid_end(text):
    return text.endswith(tuple(INVALID_END))


def has_valid_brackets(text):
    return (text.count("(") == text.count(")") and
            text.count("[") == text.count("]") and
            text.count("{") == text.count("}"))


def has_valid_quotes(text):
    return text.count('"') % 2 == 0 and text.count("“") == text.count("”")


def is_empty(text):
    return len(text) == 0


def has_invalid_character(text):
    for char in text:
        if char.lower() not in VALID_CHARS:
            return True
    return False


def normalize_double_quotes(text: str) -> str:
    if not text:
        return text
    return (text.replace("«", '"').replace("»", '"')
                .replace("“", '"').replace("”", '"')
                .replace("„", '"'))


def remove_single_letter_speaker_labels(text: str) -> str:
    if not text:
        return text
    return SPEAKER_LABEL_SINGLE_LETTER.sub(r"\1", text)


def remove_speaker_label_word_dash(text: str) -> str:
    if not text:
        return text
    return SPEAKER_LABEL_WORD_DASH.sub(r'\1', text)


def remove_dialogue_leading_double_dash(text: str) -> str:
    if not text:
        return text
    return DIALOGUE_LEADING_DASH.sub(lambda m: (m.group(1) or '') + ' ', text)


def capitalize_sentence_starts(text: str) -> str:
    if not text:
        return text
    m = re.match(r'^([\"\'“”«»\(\[\{]*)([a-zà-öø-ÿ])', text)
    if m:
        text = m.group(1) + m.group(2).upper() + text[m.end():]
    def repl(m):
        return (m.group(1) or '') + (m.group(2) or '') + m.group(3).upper()
    return CAP_SENT_START.sub(repl, text)


def author_transform_chain(text: str) -> str:
    text = remove_retweets(text)
    text = remove_mentions(text)
    text = remove_hashtags(text)
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = normalize_double_quotes(text)
    text = remove_single_letter_speaker_labels(text)
    text = remove_speaker_label_word_dash(text)
    text = remove_dialogue_leading_double_dash(text)
    text = remove_cod_literature(text)
    text = remove_bullets(text)
    text = remove_three_dashes(text)
    text = remove_quote_space_start(text)
    text = remove_quote_space_end(text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    text = capitalize_sentence_starts(text)
    return text


def apply_clean_text_ascii(s: str) -> str:
    return clean(s, fix_unicode=True, to_ascii=True, lower=False,
                 no_line_breaks=False, no_urls=False, no_emails=False,
                 no_phone_numbers=False, no_numbers=False, no_digits=False, no_currency_symbols=False)


def apply_clean_text_unicode_only(s: str) -> str:
    return clean(s, fix_unicode=True, to_ascii=False, lower=False,
                 no_line_breaks=False, no_urls=False, no_emails=False,
                 no_phone_numbers=False, no_numbers=False, no_digits=False, no_currency_symbols=False)


_FALLBACK_TOKEN_RE = re.compile(r"\w+|[^\w\s]")

def pt_word_count(s: str) -> int:
    try:
        return len(nltk.word_tokenize(s, language="portuguese"))
    except LookupError:
        return len(_FALLBACK_TOKEN_RE.findall(s))


# ===== jusText wrapper =====

def web_justext(text: str,
                *,
                drop_if_no_good: bool = True,
                min_chars: int = 30,
                min_ratio: float = 0.20) -> str:
    if not text:
        return ""
    try:
        import justext
        paras = justext.justext(text, justext.get_stoplist("Portuguese"))
        good = [p.text for p in paras if p.class_type == "good"]
        if not good:
            return "" if drop_if_no_good else text
        jt = "\n".join(good).strip()
        if len(jt) < min_chars:
            return ""
        if len(jt) / max(1, len(text)) < min_ratio:
            return ""
        return jt
    except Exception:
        return text


def _pass_filters(t: str) -> bool:
    return (not starts_with_month(t)
            and not has_too_long_word(t)
            and not has_invalid_start(t)
            and not has_invalid_middle(t)
            and not has_invalid_end(t)
            and not has_more_than_three_points(t)
            and not is_empty(t)
            and not has_invalid_character(t)
            and has_valid_brackets(t)
            and has_valid_quotes(t))


def _safe_tbl(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _iter_raw_rows(con: duckdb.DuckDBPyConnection, domain: str, split: str, batch_size: int = 10_000):
    cur = con.execute(
        """
        SELECT label, text_pt_br, text_pt_pt
        FROM ptbrvarid
        WHERE dataset='PtBrVId-Raw' AND domain=? AND split=?
        """,
        [domain, split],
    )
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for lbl, br, pt in rows:
            yield lbl, br, pt


def _normalize_label(lbl) -> str:
    """
    Normalize raw labels to notebook-compatible targets.
    Supports string labels and legacy numeric labels.
    """
    if lbl is None:
        return "pt-PT"

    if isinstance(lbl, str):
        s = lbl.strip().lower().replace("_", "-")
        if s in {"pt-br", "br", "ptbr", "brazil", "brazilian"}:
            return "pt-BR"
        if s in {"pt-pt", "pt", "eu", "european"}:
            return "pt-PT"

    try:
        return "pt-BR" if int(lbl) == 1 else "pt-PT"
    except Exception:
        return "pt-PT"


def process_domain_split(db_path: Path, domain: str, split: str) -> None:
    print(f"[filter_ptbrvarid] start domain={domain} split={split}")
    stage = f"__ptbr_stage_{_safe_tbl(domain)}_{_safe_tbl(split)}"

    raw_n = n_nonempty = n_after_jt = n_after_author = 0
    n_after_clean = n_after_dedup = n_after_filters = 0
    seen, buf = set(), []
    with duckdb.connect(db_path.as_posix()) as con:
        # prepare stage table
        con.execute(f"DROP TABLE IF EXISTS {stage}")
        con.execute(f"""
            CREATE TABLE {stage} (
                domain TEXT,
                split  TEXT,
                label  TEXT,
                text   TEXT,
                len_tokens BIGINT
            )
        """)

        # pass 1: stream raw -> stage
        for lbl_raw, br, pt in _iter_raw_rows(con, domain, split):
            raw_n += 1
            t = (br or pt or "")
            t = (t or "").strip()
            if not t:
                continue
            n_nonempty += 1

            t = web_justext(t)
            if not t:
                continue
            n_after_jt += 1

            t = author_transform_chain(t)
            if not t:
                continue
            n_after_author += 1

            t = apply_clean_text_unicode_only(t)
            t = (t or "").strip()
            if not t:
                continue
            n_after_clean += 1

            h = hashlib.md5(t.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            n_after_dedup += 1

            if not _pass_filters(t):
                continue
            n_after_filters += 1

            lbl = _normalize_label(lbl_raw)
            L = pt_word_count(t)
            buf.append((domain, split, lbl, t, int(L)))

            if len(buf) >= BATCH_ROWS:
                con.executemany(f"INSERT INTO {stage} VALUES (?,?,?,?,?)", buf)
                buf.clear()

        if buf:
            con.executemany(f"INSERT INTO {stage} VALUES (?,?,?,?,?)", buf)
            buf.clear()

        n_stage = con.execute(f"SELECT COUNT(*) FROM {stage}").fetchone()[0]
        if n_stage == 0:
            metrics_row = (
                "PtBrVId", domain, split,
                raw_n, n_nonempty, n_after_jt, n_after_author, n_after_clean,
                n_after_dedup, n_after_filters, 0,
                raw_n - n_nonempty, n_nonempty - n_after_jt, n_after_jt - n_after_author,
                n_after_author - n_after_clean, n_after_clean - n_after_dedup,
                n_after_dedup - n_after_filters, n_after_filters - 0,
                0.0, 0.0,
            )
            con.execute("""
                INSERT INTO ptbrvarid_metrics (
                    dataset, domain, split,
                    raw, after_nonempty, after_jusText, after_author, after_clean,
                    after_dedup, after_filters, after_IQR,
                    drop_empty, drop_jusText, drop_author, drop_clean, drop_dedup, drop_filters, drop_IQR,
                    IQR_lo, IQR_hi
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, metrics_row)
            con.execute(f"DROP TABLE {stage}")
            print(f"{domain}/{split}: raw={raw_n:,} → after_IQR=0 (0.0%)")
            return

        q1, q3 = con.execute(
            f"SELECT quantile_cont(len_tokens,0.25), quantile_cont(len_tokens,0.75) FROM {stage}"
        ).fetchone()
        iqr = q3 - q1
        lo = float(q1 - 1.5 * iqr)
        hi = float(q3 + 1.5 * iqr)

        n_after_iqr = con.execute(
            f"SELECT COUNT(*) FROM {stage} WHERE len_tokens BETWEEN ? AND ?", [lo, hi]
        ).fetchone()[0]

        con.execute(f"""
            INSERT INTO ptbrvarid (dataset, domain, split, label, text_pt_br, text_pt_pt)
            SELECT
              'PtBrVId' AS dataset,
              domain, split, label,
              CASE WHEN label='pt-BR' THEN text ELSE NULL END AS text_pt_br,
              CASE WHEN label='pt-PT' THEN text ELSE NULL END AS text_pt_pt
            FROM {stage}
            WHERE len_tokens BETWEEN ? AND ?
        """, [lo, hi])

        metrics_row = (
            "PtBrVId", domain, split,
            raw_n, n_nonempty, n_after_jt, n_after_author, n_after_clean,
            n_after_dedup, n_after_filters, int(n_after_iqr),
            raw_n - n_nonempty, n_nonempty - n_after_jt, n_after_jt - n_after_author,
            n_after_author - n_after_clean, n_after_clean - n_after_dedup,
            n_after_dedup - n_after_filters, n_after_filters - int(n_after_iqr),
            lo, hi,
        )
        con.execute("""
            INSERT INTO ptbrvarid_metrics (
                dataset, domain, split,
                raw, after_nonempty, after_jusText, after_author, after_clean,
                after_dedup, after_filters, after_IQR,
                drop_empty, drop_jusText, drop_author, drop_clean, drop_dedup, drop_filters, drop_IQR,
                IQR_lo, IQR_hi
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, metrics_row)

        con.execute(f"DROP TABLE {stage}")

        pct = round(100.0 * (n_after_iqr / max(1, raw_n)), 2)
        print(f"{domain}/{split}: raw={raw_n:,} → after_IQR={n_after_iqr:,} ({pct}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run PtBrVarId filtering pipeline (jusText + clean + IQR).")
    ap.add_argument("--db", default=str(DEFAULT_DB), help="DuckDB path (default: data/duckdb/subs.duckdb)")
    args = ap.parse_args()

    print(f"[filter_ptbrvarid] db={args.db}")
    ensure_punkt()

    db_path = Path(args.db).expanduser().resolve()
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")

    with duckdb.connect(db_path.as_posix()) as con:
        # sanity
        n_raw = con.execute("SELECT COUNT(*) FROM ptbrvarid WHERE dataset='PtBrVId-Raw'").fetchone()[0]
        if n_raw == 0:
            raise SystemExit("No PtBrVId-Raw rows found. Run ingest_ptbrvarid.py first.")

        con.execute("""
            CREATE TABLE IF NOT EXISTS ptbrvarid_metrics (
                dataset     TEXT,
                domain      TEXT,
                split       TEXT,
                raw                 BIGINT,
                after_nonempty      BIGINT,
                after_jusText       BIGINT,
                after_author        BIGINT,
                after_clean         BIGINT,
                after_dedup         BIGINT,
                after_filters       BIGINT,
                after_IQR           BIGINT,
                drop_empty          BIGINT,
                drop_jusText        BIGINT,
                drop_author         BIGINT,
                drop_clean          BIGINT,
                drop_dedup          BIGINT,
                drop_filters        BIGINT,
                drop_IQR            BIGINT,
                IQR_lo              DOUBLE,
                IQR_hi              DOUBLE
            )
        """)
        con.execute("DELETE FROM ptbrvarid WHERE dataset='PtBrVId'")
        con.execute("DELETE FROM ptbrvarid_metrics WHERE dataset='PtBrVId'")

        domains = con.execute(
            "SELECT DISTINCT domain FROM ptbrvarid WHERE dataset='PtBrVId-Raw' ORDER BY domain"
        ).fetchall()
        domains = [d[0] for d in domains]

        print(f"[filter_ptbrvarid] domains={len(domains)}")
        for domain in domains:
            splits = con.execute(
                "SELECT DISTINCT split FROM ptbrvarid WHERE dataset='PtBrVId-Raw' AND domain=? ORDER BY split",
                [domain],
            ).fetchall()
            splits = [s[0] for s in splits]
            print(f"[filter_ptbrvarid] domain={domain} splits={splits}")
            for split in splits:
                try:
                    process_domain_split(db_path, domain, split)
                except KeyboardInterrupt:
                    print(f"[filter_ptbrvarid] interrupted at domain={domain} split={split}")
                    raise SystemExit(130)
                except Exception as e:
                    print(f"[filter_ptbrvarid] ERROR domain={domain} split={split}: {e}")
                    raise

    print("[ok] PtBrVId filtering complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
