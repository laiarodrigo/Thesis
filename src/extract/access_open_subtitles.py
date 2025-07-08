import os, gzip, requests, time
import pandas as pd
import pathlib

from datetime import datetime, timezone
from dotenv import load_dotenv         
from pathlib import Path
from opensubtitlescom import OpenSubtitles
from xmlrpc.client import ServerProxy, Error as XMLRPCError

load_dotenv()

OUTPUT_DIR_BR = Path("../data/raw/test_br_subs")
OUTPUT_DIR_PT = Path("../data/raw/test_pt_subs")
os.makedirs(OUTPUT_DIR_BR, exist_ok=True)
os.makedirs(OUTPUT_DIR_PT, exist_ok=True)

API_BASE = "https://api.opensubtitles.com/api/v1"
API_KEY = os.getenv("OPENSUBTITLES_API_KEY")
USERNAME = os.getenv("OPENSUBTITLES_USER", "")
PASSWORD = os.getenv("OPENSUBTITLES_PASS", "")
USER_AGENT = "MySubtitleApp/1.0"

# REST headers
# REST_HEADERS = {
#     "Api-Key":     API_KEY,
#     "User-Agent":  USER_AGENT,
#     "Content-Type":"application/json"
# }

# HEAD = {
#     "Api-Key": os.getenv("OPENSUBTITLES_API_KEY"),
#     "User-Agent": "TeseCollector/0.1",
#     "Accept": "application/json",
# }

BASE_HEADERS = {
    "Api-Key":     API_KEY,
    "User-Agent":  "MySubtitleApp/1.0",          # <- descriptive!
    "Accept":      "application/json",           # <- important
    "Content-Type":"application/json",
}
#AUTH_HEADERS = dict(BASE_HEADERS)       # will gain 'Authorization' below

# XML-RPC client (fallback path)
ost = OpenSubtitles(user_agent=USER_AGENT, api_key=API_KEY)
# optional login for higher quotas
if USERNAME and PASSWORD:
    try:
        ost.login(USERNAME, PASSWORD)
    except Exception:
        pass

# YEARS AND LANGUAGES
YEARS = range(2023, 2024)          
LANGS = {"pt-br", "pt-pt"}             
by_lang = {lang: {} for lang in LANGS}

TIMEOUT = 15

AUTH_HEADERS = BASE_HEADERS.copy()

def login() -> str | None:
    """Return 'Bearer <token>' or None; never raise on 5xx."""
    if not USERNAME or not PASSWORD:
        return None                    # anonymous (API-key-only)

    r = requests.post(f"{API_BASE}/login",
                      headers=BASE_HEADERS,
                      json={"username": USERNAME, "password": PASSWORD},
                      timeout=15)
    if r.status_code == 401:           # bad creds → skip
        print("OpenSubtitles: wrong user/password; staying anonymous.")
        return None
    r.raise_for_status()               # will raise on 5xx
    return f"Bearer {r.json()['token']}"

# --- safest call -----------------------------------------------------
try:
    token = login()
    if token:
        AUTH_HEADERS["Authorization"] = token
        print("OpenSubtitles: logged-in session OK.")
    else:
        print("OpenSubtitles: using API-key-only mode.")
except requests.HTTPError as e:
    # 5xx or network glitch – just warn and keep going
    print("OpenSubtitles login failed:", e, "→ continuing without token.")
    
def fetch_subtitles_for_br(imdb_id: str):
    resp = requests.get(
        f"{API_BASE}/subtitles",
        headers=AUTH_HEADERS,          #  ← 1) send Api-Key & UA
        params={
            "imdb_id":   imdb_id,
            "languages": "pt-br",    #  ← 2) valid PT codes
            "order_by":  "downloads",
        },
        timeout=TIMEOUT
    )
    resp.raise_for_status()
    hit = next(iter(resp.json().get("data", [])), None)
    if not hit:
        return None            # nothing in that language

    file_id = hit["attributes"]["files"][0]["file_id"]

    # 2) Download
    dl = requests.post(f"{API_BASE}/download",
                       headers=AUTH_HEADERS,
                       json={"file_id": file_id},
                       timeout=TIMEOUT)

    if dl.status_code == 406:          # show the API explanation
        print("Download refused:", dl.json())
        return None
    dl.raise_for_status()

    url  = dl.json()["link"]
    blob = requests.get(url, timeout=TIMEOUT).content
    if url.endswith(".gz"):
        blob = gzip.decompress(blob)

    return blob.decode("utf-8", errors="replace")
        
def fetch_subtitles_for_pt(imdb_id: str):
    resp = requests.get(
        f"{API_BASE}/subtitles",
        headers=AUTH_HEADERS,          #  ← 1) send Api-Key & UA
        params={
            "imdb_id":   imdb_id,
            "languages": "pt-pt",    #  ← 2) valid PT codes
            "order_by":  "downloads",
        },
        timeout=TIMEOUT
    )
    resp.raise_for_status()
    hit = next(iter(resp.json().get("data", [])), None)
    if not hit:
        return None            # nothing in that language

    file_id = hit["attributes"]["files"][0]["file_id"]

    # 2) Download
    dl = requests.post(f"{API_BASE}/download",
                       headers=AUTH_HEADERS,
                       json={"file_id": file_id},
                       timeout=TIMEOUT)

    if dl.status_code == 406:          # show the API explanation
        print("Download refused:", dl.json())
        return None
    dl.raise_for_status()

    url  = dl.json()["link"]
    blob = requests.get(url, timeout=TIMEOUT).content
    if url.endswith(".gz"):
        blob = gzip.decompress(blob)

    return blob.decode("utf-8", errors="replace")

# def fetch_subtitles_for_pt(imdb_id: str, max_alternatives: int = 5) -> str | None:
 
#     search_q = {
#         "imdb_id": imdb_id,
#         "languages": "pt-pt",
#         "order_by": "downloads",      # best first
#         "order_direction": "desc",
#         "page": 1
#     }

#     try:
#         r = safe_get(f"{API_BASE}/subtitles", headers=BASE_HEADERS, params=search_q)
#     except Exception as e:
#         print(f"search failed for {imdb_id}: {e}")
#         return None

#     releases = r.json()["data"]
#     if not releases:
#         return None

#     # try up to N different releases until one downloads successfully
#     for rel in releases[:max_alternatives]:
#         file_id = rel["id"]

#         # ask for download link
#         try:
#             dl = safe_get(f"{API_BASE}/download",
#                           headers=BASE_HEADERS, params={"file_id": file_id})
#         except Exception:
#             continue                        # transient error → try next release

#         link = dl.json().get("link")
#         if not link:                        # 403/404 or other refusal
#             continue

#         # grab the .srt
#         try:
#             blob = safe_get(link).content
#             return blob.decode("utf-8", errors="replace")
#         except Exception:
#             continue                        # network error → next release

#     # none of the tested releases were downloadable
#     return None

def save_batch(movies_df: pd.DataFrame, lang: str, n: int = 10):
    """
    Fetch subtitles for the first *n* movies in *movies_df*.
    Rows must expose .imdb_id and .title
    """
    fetch = fetch_subtitles_for_br if lang.lower() == "pt-br" else fetch_subtitles_for_pt
    out_dir = OUTPUT_DIR_BR         if lang.lower() == "pt-br" else OUTPUT_DIR_PT

    for movie in movies_df.head(n).itertuples(index=False):   # ← fixed
        imdb_id = movie.imdb_id
        title   = movie.title

        out_file = out_dir / f"{imdb_id}.srt"
        if out_file.exists():
            print(f"✓ Already have {imdb_id} ({title})")
            continue

        blob = fetch(imdb_id)
        if not blob:
            print(f"✗ No {lang} subs for {imdb_id} ({title})")
            continue

        out_file.write_text(blob, encoding="utf-8")
        print(f"✓ Saved {lang}: {imdb_id} ({title})")


from hashlib import md5

RAW_DIR = Path("../data/subs_in_duckdb")          # one canonical folder for .srt
RAW_DIR.mkdir(parents=True, exist_ok=True)


def _hash(b: bytes) -> str:
    return md5(b).hexdigest()


# ------------ pipeline-facing helper ------------
def download_srt(imdb_id: str, lang: str, *, title: str | None = None) -> Path:
    """
    Unified entry point for the ETL.
    • lang must be 'pt-PT' or 'pt-BR'
    • returns Path to <data/raw>/<hash>.srt    (creates it if missing)
    """
    if lang.lower() == "pt-br":
        blob = fetch_subtitles_for_br(imdb_id)
    elif lang.lower() == "pt-pt":
        blob = fetch_subtitles_for_pt(imdb_id)
    else:
        raise ValueError("lang must be 'pt-PT' or 'pt-BR'")

    if blob is None:
        raise RuntimeError(f"No {lang} subs for {imdb_id} ({title or ''})")

    h   = _hash(blob.encode("utf-8"))
    fp  = RAW_DIR / f"{h}.srt"
    if not fp.exists():
        fp.write_text(blob, encoding="utf-8")

    return fp
