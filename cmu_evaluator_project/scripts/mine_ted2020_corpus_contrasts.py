#!/usr/bin/env python3
"""Mine stable BP/PT TED2020 contrasts and rank researcher-review passages.

PT TED is a matched-topic contrast corpus only.  It is never a conservative
rewrite, annotation reference, candidate, or gold target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sqlite3
import statistics
import time
import unicodedata
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rank_ted2020_bp_passages import LEXICAL_SIGNALS, ORTHOGRAPHIC_SIGNALS, STRUCTURAL_SIGNALS


TOKEN_RE = re.compile(r"[^\W\d_]+(?:[-’'][^\W\d_]+)*", re.UNICODE)
STAGE_RE = re.compile(r"^\s*[\[(](?:aplausos?|risos?|m[úu]sica|v[íi]deo|áudio|audio|inaudível|silêncio)[^)\]]*[)\]]", re.I)
SPEAKER_RE = re.compile(r"^\s*[A-ZÁÉÍÓÚÂÊÔÃÕÇ]{1,6}:\s")
SUSPICIOUS_RE = re.compile(r"\ufffd|[\u0000-\u0008\u000b\u000c\u000e-\u001f]|[\ue000-\uf8ff]")
FUNCTION_WORDS = {"a", "à", "ao", "aos", "as", "às", "o", "os", "de", "da", "das", "do", "dos", "e", "é", "em", "na", "nas", "no", "nos", "para", "por", "que", "se", "um", "uma", "uns", "umas", "com", "como", "não", "mais", "mas", "ou", "eu", "ele", "ela", "eles", "elas"}
MAIN_FAMILIES = ("surface_form", "lemma", "pos_trigram")
ALL_FAMILIES = MAIN_FAMILIES + ("word_bigram", "word_trigram")
WINDOW_LENGTHS = (8, 9, 10)
FOLD_COUNT = 5
JM_LAMBDA = 0.05
SIGNIFICANCE_P = 0.001
LOG_ODDS_ALPHA = 0.5
SETTINGS = {
    "surface_form": {"min_count": 20, "min_docs": 8, "retain": 350},
    "lemma": {"min_count": 20, "min_docs": 8, "retain": 350},
    "pos_trigram": {"min_count": 40, "min_docs": 12, "retain": 250},
    "word_bigram": {"min_count": 15, "min_docs": 7, "retain": 250},
    "word_trigram": {"min_count": 10, "min_docs": 5, "retain": 200},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bp-corpus", type=Path, default=Path("data/external/TED2020/TED2020/raw/pt_br"))
    p.add_argument("--pt-corpus", type=Path, default=Path("data/external/TED2020/TED2020/raw/pt"))
    p.add_argument("--old-shortlist", type=Path, default=Path("cmu_evaluator_project/data/passage_selection/ted2020_larger_pool_shortlist.json"))
    p.add_argument("--output-dir", type=Path, default=Path("cmu_evaluator_project/data/passage_selection"))
    p.add_argument("--stanza-model-dir", type=Path, default=Path("/tmp/ted2020-stanza-resources"))
    p.add_argument("--nlp-cache", type=Path, default=Path("/tmp/ted2020-stanza-bosque.sqlite3"))
    p.add_argument("--shortlist-size", type=int, default=100)
    p.add_argument("--negative-audit-size", type=int, default=24)
    p.add_argument("--seed", type=int, default=20201008)
    p.add_argument("--cache-only", action="store_true", help="Build one resumable NLP-cache shard and stop.")
    p.add_argument("--cache-shard-count", type=int, default=1)
    p.add_argument("--cache-shard-index", type=int, default=0)
    return p.parse_args()


def norm(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def strip_accents(value:str)->str:
    return "".join(c for c in unicodedata.normalize("NFD",value) if unicodedata.category(c)!="Mn")


def edit_distance(a:str,b:str)->int:
    previous=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        current=[i]
        for j,cb in enumerate(b,1): current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(ca!=cb)))
        previous=current
    return previous[-1]


def surface_tokens(text: str) -> list[str]:
    return [norm(m.group(0)) for m in TOKEN_RE.finditer(text)]


def ngrams(values: list[str], n: int) -> Iterable[tuple[str, ...]]:
    return zip(*(values[i:] for i in range(n)))


def read_xml(path: Path) -> list[dict[str, Any]]:
    root = ET.parse(path).getroot()
    return [{"sentence_id": str(node.attrib.get("id", pos)), "position": pos, "source_bp": "".join(node.itertext())} for pos, node in enumerate(root.findall(".//s"), 1)]


def corpus_paths(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        raise SystemExit(f"Corpus directory missing: {root}")
    paths = {p.stem: p for p in sorted(root.glob("ted2020-*.xml"))}
    if not paths:
        raise SystemExit(f"No TED XML files found in {root}")
    return paths


def stable_fold(document_id: str) -> int:
    return int(hashlib.sha256(document_id.encode()).hexdigest()[:8], 16) % FOLD_COUNT


def comparability(bp: dict[str, Path], pt: dict[str, Path]) -> dict[str, Any]:
    common = sorted(set(bp) & set(pt)); overlap = union = bp_n = pt_n = same_ids = same_count = 0
    examples = []
    for doc in common:
        br, eu = read_xml(bp[doc]), read_xml(pt[doc]); bi = {r["sentence_id"] for r in br}; pi = {r["sentence_id"] for r in eu}
        bp_n += len(br); pt_n += len(eu); overlap += len(bi & pi); union += len(bi | pi)
        same_ids += bi == pi; same_count += len(br) == len(eu)
        if len(examples) < 12 and bi != pi:
            examples.append({"document_id": doc, "bp_sentences": len(br), "pt_sentences": len(eu), "id_overlap": len(bi & pi)})
    return {
        "bp_document_count": len(bp), "pt_document_count": len(pt), "common_document_count": len(common),
        "bp_only_document_count": len(set(bp)-set(pt)), "pt_only_document_count": len(set(pt)-set(bp)),
        "bp_only_document_ids": sorted(set(bp)-set(pt)), "pt_only_document_ids": sorted(set(pt)-set(bp)),
        "common_document_ids": common, "matched_bp_sentence_count": bp_n, "matched_pt_sentence_count": pt_n,
        "sentence_id_overlap_count": overlap, "sentence_id_union_count": union,
        "sentence_id_jaccard": overlap/union, "common_documents_same_sentence_count": same_count,
        "common_documents_same_sentence_ids": same_ids,
        "common_documents_with_segmentation_or_id_difference": len(common)-same_ids,
        "segmentation_difference_examples": examples,
    }


def init_cache(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    con.execute("CREATE TABLE IF NOT EXISTS documents(lang TEXT, document_id TEXT, sentence_count INTEGER, PRIMARY KEY(lang,document_id))")
    con.execute("CREATE TABLE IF NOT EXISTS sentences(lang TEXT, document_id TEXT, position INTEGER, sentence_id TEXT, text_sha256 TEXT, annotation TEXT, PRIMARY KEY(lang,document_id,position))")
    con.execute("CREATE INDEX IF NOT EXISTS sentence_ids ON sentences(lang,document_id,sentence_id)")
    return con


def stanza_pipeline(model_dir: Path):
    try:
        import stanza
        from stanza.models.common.doc import Document
    except ImportError as exc:
        raise SystemExit("Stanza is required. Run this script in the prepared Stanza environment.") from exc
    pipeline = stanza.Pipeline(
        "pt", processors="tokenize,mwt,pos,lemma",
        package={"tokenize":"bosque", "mwt":"bosque", "pos":"bosque_nocharlm", "lemma":"bosque_nocharlm"},
        dir=str(model_dir), tokenize_no_ssplit=True, use_gpu=False, verbose=False,
        download_method=None, pos_batch_size=256,
    )
    return stanza, Document, pipeline


def build_nlp_cache(con: sqlite3.Connection, bp: dict[str, Path], pt: dict[str, Path], model_dir: Path, shard_count:int=1, shard_index:int=0) -> dict[str, Any]:
    stanza, Document, pipeline = stanza_pipeline(model_dir)
    wanted = [("bp", doc, path) for doc, path in bp.items()] + [("pt", doc, path) for doc, path in pt.items()]
    if not 0 <= shard_index < shard_count: raise SystemExit("cache shard index must be smaller than shard count")
    wanted=[row for row in wanted if int(hashlib.sha256(f"{row[0]}:{row[1]}".encode()).hexdigest()[:8],16)%shard_count==shard_index]
    done = {(r[0],r[1]):r[2] for r in con.execute("SELECT lang,document_id,sentence_count FROM documents")}
    start = time.time(); processed = 0; pending=[]; pending_sentences=0

    def flush(batch:list[tuple[str,str,list[dict[str,Any]]]])->None:
        nonlocal processed
        flat=[(bi,ri,row) for bi,(_,_,rows) in enumerate(batch) for ri,row in enumerate(rows) if row["source_bp"].strip()]
        annotations=[[[] for _ in rows] for _,_,rows in batch]
        for offset in range(0,len(flat),1024):
            chunk=flat[offset:offset+1024]
            docs=[Document([],text=row["source_bp"]) for _,_,row in chunk]
            for (bi,ri,_),doc_out in zip(chunk,pipeline.bulk_process(docs)):
                words=[w for sentence in doc_out.sentences for w in sentence.words]
                annotations[bi][ri]=[[w.text,norm(w.lemma or w.text),w.upos or "X"] for w in words]
        for (lang,doc,rows),anns in zip(batch,annotations):
            con.execute("DELETE FROM sentences WHERE lang=? AND document_id=?",(lang,doc))
            con.executemany("INSERT OR REPLACE INTO sentences VALUES(?,?,?,?,?,?)",[(lang,doc,row["position"],row["sentence_id"],hashlib.sha256(row["source_bp"].encode()).hexdigest(),json.dumps(ann,ensure_ascii=False,separators=(",",":"))) for row,ann in zip(rows,anns)])
            con.execute("INSERT OR REPLACE INTO documents VALUES(?,?,?)",(lang,doc,len(rows))); processed+=1
        con.commit()
        if processed and processed%50 < len(batch): print(f"NLP_CACHE documents={processed} total={len(wanted)} elapsed_min={(time.time()-start)/60:.1f}",flush=True)

    for lang,doc,path in wanted:
        rows=read_xml(path)
        if done.get((lang,doc))==len(rows): continue
        pending.append((lang,doc,rows)); pending_sentences+=len(rows)
        if pending_sentences>=3000:
            flush(pending); pending=[]; pending_sentences=0
    if pending: flush(pending)
    info = {
        "stanza_version": stanza.__version__, "language": "pt", "treebank_package": "Bosque",
        "processors": ["tokenize","mwt","pos","lemma"],
        "models": {"tokenize":"bosque", "mwt":"bosque", "pos":"bosque_nocharlm", "lemma":"bosque_nocharlm"},
        "tokenize_no_ssplit": True, "device": "cpu",
        "deviation_from_alves": "Official Stanza Bosque no-charlm POS/lemma models were used for feasible full-corpus CPU analysis; dependency and word-order analyses were not required and were omitted.",
    }
    for key,value in info.items(): con.execute("INSERT OR REPLACE INTO metadata VALUES(?,?)",(key,json.dumps(value)))
    con.commit(); return info


def load_nlp_doc(con: sqlite3.Connection, lang: str, doc: str) -> list[list[list[str]]]:
    return [json.loads(r[0]) for r in con.execute("SELECT annotation FROM sentences WHERE lang=? AND document_id=? ORDER BY position",(lang,doc))]


def sentence_features(text: str, annotation: list[list[str]]) -> dict[str, list[str]]:
    surf = surface_tokens(text)
    lemmas = [x[1] for x in annotation if x[2] not in {"PUNCT","SYM","X"} and TOKEN_RE.search(x[0])]
    pos = [x[2] for x in annotation if x[2] not in {"PUNCT","SYM","X"}]
    return {
        "surface_form": surf,
        "lemma": lemmas,
        "pos_trigram": [" ".join(x) for x in ngrams(pos,3)],
        "word_bigram": [" ".join(x) for x in ngrams(surf,2)],
        "word_trigram": [" ".join(x) for x in ngrams(surf,3)],
    }


def empty_aggregate() -> dict[str, Any]:
    return {f:{"bp":Counter(),"pt":Counter(),"bp_docs":Counter(),"pt_docs":Counter(),"bp_total":0,"pt_total":0} for f in ALL_FAMILIES}


def aggregate_matched(con: sqlite3.Connection, bp: dict[str,Path], pt: dict[str,Path], common: list[str]) -> tuple[dict[str,Any],dict[str,dict[str,Counter]]]:
    agg=empty_aggregate(); proper={"surface_form":{"bp":Counter(),"pt":Counter()},"lemma":{"bp":Counter(),"pt":Counter()}}
    for di,doc in enumerate(common,1):
        for lang,paths in (("bp",bp),("pt",pt)):
            rows=read_xml(paths[doc]); anns=load_nlp_doc(con,lang,doc); seen={f:set() for f in ALL_FAMILIES}
            for row,ann in zip(rows,anns):
                feats=sentence_features(row["source_bp"],ann)
                for family,vals in feats.items(): agg[family][lang].update(vals); agg[family][f"{lang}_total"]+=len(vals); seen[family].update(vals)
                for text,lemma,upos in ann:
                    if upos=="PROPN": proper["surface_form"][lang][norm(text)]+=1; proper["lemma"][lang][lemma]+=1
            for family in ALL_FAMILIES: agg[family][f"{lang}_docs"].update(seen[family])
        if di%250==0: print(f"AGGREGATE matched_documents={di}/{len(common)}",flush=True)
    return agg,proper


def smoothed_prob(c:int,total:int,pooled:float)->float:
    return (1-JM_LAMBDA)*(c/total if total else 0.0)+JM_LAMBDA*pooled


def weighted_log_odds(a:int,b:int,at:int,bt:int)->tuple[float,float]:
    d=math.log((a+LOG_ODDS_ALPHA)/(at-a+LOG_ODDS_ALPHA))-math.log((b+LOG_ODDS_ALPHA)/(bt-b+LOG_ODDS_ALPHA))
    return d,d/math.sqrt(1/(a+LOG_ODDS_ALPHA)+1/(b+LOG_ODDS_ALPHA))


def chi_square(a:int,b:int,at:int,bt:int)->tuple[float,float]:
    table=((a,at-a),(b,bt-b)); n=at+bt; r=(at,bt); c=(a+b,n-a-b); value=0.0
    for i in range(2):
        for j in range(2):
            expected=r[i]*c[j]/n if n else 0
            if expected: value+=(table[i][j]-expected)**2/expected
    return value,math.erfc(math.sqrt(value/2))


def initial_features(agg:dict[str,Any],proper:dict[str,dict[str,Counter]])->tuple[dict[str,Any],dict[str,dict[str,list[dict[str,Any]]]]]:
    divergence={}; output={}
    for family in ALL_FAMILIES:
        data=agg[family]; bc,pc=data["bp"],data["pt"]; bt,pt=data["bp_total"],data["pt_total"]; vocab=set(bc)|set(pc)
        d_bp=d_pt=0.0; candidates={"bp_characteristic":[],"pt_characteristic":[]}; setting=SETTINGS[family]
        for feature in vocab:
            pooled=(bc[feature]+pc[feature])/(bt+pt); pb=smoothed_prob(bc[feature],bt,pooled); pp=smoothed_prob(pc[feature],pt,pooled)
            kb=pb*math.log2(pb/pp); kp=pp*math.log2(pp/pb); d_bp+=kb; d_pt+=kp
            direction="bp" if pb>pp else "pt"; own=bc[feature] if direction=="bp" else pc[feature]; docs=data[f"{direction}_docs"][feature]
            if own<setting["min_count"] or docs<setting["min_docs"]: continue
            delta,z=weighted_log_odds(bc[feature],pc[feature],bt,pt); chi,p=chi_square(bc[feature],pc[feature],bt,pt)
            proper_ratio=proper.get(family,{}).get(direction,Counter())[feature]/max(own,1)
            flags=[]
            if proper_ratio>=0.70: flags.append("possible_named_entity")
            if family=="surface_form" and feature in FUNCTION_WORDS: flags.append("high_frequency_function_word")
            row={
                "feature":feature,"feature_type":family,"direction":direction,"bp_count":bc[feature],"pt_count":pc[feature],
                "bp_frequency_per_million":bc[feature]*1e6/bt,"pt_frequency_per_million":pc[feature]*1e6/pt,
                "bp_document_count":data["bp_docs"][feature],"pt_document_count":data["pt_docs"][feature],
                "bp_smoothed_probability":pb,"pt_smoothed_probability":pp,
                "pointwise_kld_bp_to_pt_bits":kb,"pointwise_kld_pt_to_bp_bits":kp,
                "weighted_log_odds_delta":delta,"weighted_log_odds_z":z,"chi_square":chi,"chi_square_p":p,
                "significant_p_lt_0_001":p<SIGNIFICANCE_P,"flags":flags,
                "in_old_lexical_inventory": any(signal.pattern.search(feature) for signal in LEXICAL_SIGNALS),
                "in_any_old_seeded_inventory": any(signal.pattern.search(feature) for signal in LEXICAL_SIGNALS) or any(pattern.search(feature) for _,_,pattern in ORTHOGRAPHIC_SIGNALS+STRUCTURAL_SIGNALS),
            }
            if p<SIGNIFICANCE_P: candidates[f"{direction}_characteristic"].append(row)
        for direction in ("bp_characteristic","pt_characteristic"):
            key="pointwise_kld_bp_to_pt_bits" if direction.startswith("bp") else "pointwise_kld_pt_to_bp_bits"
            candidates[direction].sort(key=lambda x:(x[key],abs(x["weighted_log_odds_z"])),reverse=True)
            candidates[direction]=candidates[direction][:setting["retain"]]
        divergence[family]={"bp_to_pt_bits":d_bp,"pt_to_bp_bits":d_pt,"bp_total_events":bt,"pt_total_events":pt,"vocabulary_size":len(vocab)}
        output[family]=candidates
    for direction in ("bp_characteristic","pt_characteristic"):
        entities={r["feature"] for r in output["surface_form"][direction] if "possible_named_entity" in r["flags"]}
        for family in ("word_bigram","word_trigram"):
            for row in output[family][direction]:
                if any(token in entities for token in row["feature"].split()): row["flags"].append("contains_possible_named_entity")
    return divergence,output


def selected_keys(features:dict[str,Any])->dict[str,set[str]]:
    return {f:{r["feature"] for d in ("bp_characteristic","pt_characteristic") for r in features[f][d]} for f in ALL_FAMILIES}


def enrich_stability_and_examples(con:sqlite3.Connection,bp:dict[str,Path],pt:dict[str,Path],common:list[str],agg:dict[str,Any],features:dict[str,Any])->None:
    keys=selected_keys(features); fold_counts={f:{l:[Counter() for _ in range(FOLD_COUNT)] for l in ("bp","pt")} for f in ALL_FAMILIES}; fold_totals={f:{l:[0]*FOLD_COUNT for l in ("bp","pt")} for f in ALL_FAMILIES}
    examples={f:{l:defaultdict(list) for l in ("bp","pt")} for f in ALL_FAMILIES}; surface_lemmas=defaultdict(Counter)
    for di,doc in enumerate(common,1):
        fold=stable_fold(doc)
        for lang,paths in (("bp",bp),("pt",pt)):
            rows=read_xml(paths[doc]); anns=load_nlp_doc(con,lang,doc)
            for row,ann in zip(rows,anns):
                feats=sentence_features(row["source_bp"],ann)
                if lang=="bp":
                    for text,lemma,upos in ann:
                        if TOKEN_RE.search(text): surface_lemmas[norm(text)][lemma]+=1
                for family,vals in feats.items():
                    fold_totals[family][lang][fold]+=len(vals); present=Counter(v for v in vals if v in keys[family]); fold_counts[family][lang][fold].update(present)
                    for feat in present:
                        if len(examples[family][lang][feat])<8:
                            examples[family][lang][feat].append({"document_id":doc,"sentence_id":row["sentence_id"],"position":row["position"],"source_text":row["source_bp"],"nlp_annotation_sha256":hashlib.sha256(json.dumps(ann,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()})
        if di%300==0: print(f"STABILITY documents={di}/{len(common)}",flush=True)
    for family in ALL_FAMILIES:
        index={r["feature"]:r for d in ("bp_characteristic","pt_characteristic") for r in features[family][d]}
        for feat,row in index.items():
            overall=row["direction"]; directions=[]; strengths=[]
            for held in range(FOLD_COUNT):
                bc=sum(fold_counts[family]["bp"][i][feat] for i in range(FOLD_COUNT) if i!=held); pc=sum(fold_counts[family]["pt"][i][feat] for i in range(FOLD_COUNT) if i!=held)
                bt=sum(fold_totals[family]["bp"][i] for i in range(FOLD_COUNT) if i!=held); ptot=sum(fold_totals[family]["pt"][i] for i in range(FOLD_COUNT) if i!=held)
                pooled=(bc+pc)/(bt+ptot); pb=smoothed_prob(bc,bt,pooled); pp=smoothed_prob(pc,ptot,pooled); ratio=math.log2(pb/pp)
                directions.append("bp" if ratio>0 else "pt"); strengths.append(ratio)
            same=sum(d==overall for d in directions); row["stability"]={"scheme":"deterministic five-fold document-level leave-one-fold-out","fold_directions":directions,"same_direction_folds":same,"log2_ratio_by_replication":strengths,"effect_stddev":statistics.pstdev(strengths)}
            # Frequency is evidence, not a defect.  Keep the function-word
            # marker as descriptive metadata, but only entity-related flags
            # block passage scoring.
            blocking_flags={"possible_named_entity","contains_possible_named_entity"}
            row["selection_eligible"]=(same>=4 and not (set(row["flags"])&blocking_flags))
            row["examples"]={"bp":examples[family]["bp"][feat],"pt":examples[family]["pt"][feat]}
            if family=="surface_form": row["dominant_stanza_lemma"]=(surface_lemmas[feat].most_common(1)[0][0] if surface_lemmas[feat] else feat)
        for direction in ("bp_characteristic","pt_characteristic"):
            eligible=[r for r in features[family][direction] if r["selection_eligible"]]
            point_key="pointwise_kld_bp_to_pt_bits" if direction.startswith("bp") else "pointwise_kld_pt_to_bp_bits"
            maximum=max((r[point_key] for r in eligible),default=1.0)
            scale=max(len(eligible)*0.25,1.0)
            for rank,row in enumerate(eligible):
                point_ratio=math.sqrt(max(row[point_key],0.0)/maximum)
                row["normalized_strength_within_family"]=point_ratio*math.exp(-rank/scale)


def feature_lookup(features:dict[str,Any],direction:str="bp_characteristic")->dict[str,dict[str,dict[str,Any]]]:
    return {f:{r["feature"]:r for r in features[f][direction] if r.get("selection_eligible")} for f in ALL_FAMILIES}


def high_confidence_pairs(bp_rows:list[dict[str,Any]],pt_rows:list[dict[str,Any]])->list[tuple[dict[str,Any],dict[str,Any]]]:
    bm=defaultdict(list); pm=defaultdict(list)
    for r in bp_rows: bm[r["sentence_id"]].append(r)
    for r in pt_rows: pm[r["sentence_id"]].append(r)
    pairs=[]
    for sid in bm.keys()&pm.keys():
        if len(bm[sid])!=1 or len(pm[sid])!=1: continue
        b,p=bm[sid][0],pm[sid][0]; bl=len(surface_tokens(b["source_bp"])); pl=len(surface_tokens(p["source_bp"]))
        clean=not STAGE_RE.search(b["source_bp"]) and not STAGE_RE.search(p["source_bp"]) and not SUSPICIOUS_RE.search(b["source_bp"]) and not SUSPICIOUS_RE.search(p["source_bp"])
        if clean and min(bl,pl)>=3 and 0.45<=bl/pl<=2.2: pairs.append((b,p))
    return pairs


def aligned_associations(con:sqlite3.Connection,bp:dict[str,Path],pt:dict[str,Path],common:list[str],features:dict[str,Any])->tuple[dict[str,list[dict[str,Any]]],dict[str,dict[str,dict[str,Any]]],dict[str,Any]]:
    look_bp=feature_lookup(features,"bp_characteristic"); look_pt=feature_lookup(features,"pt_characteristic"); families=MAIN_FAMILIES
    pair_counts={f:Counter() for f in families}; bx={f:Counter() for f in families}; py={f:Counter() for f in families}; docs={f:defaultdict(set) for f in families}; folds={f:defaultdict(set) for f in families}; fold_supports={f:defaultdict(lambda:[0]*FOLD_COUNT) for f in families}; examples={f:defaultdict(list) for f in families}; accepted=0; rejected=0
    for di,doc in enumerate(common,1):
        br=read_xml(bp[doc]); pr=read_xml(pt[doc]); ba=load_nlp_doc(con,"bp",doc); pa=load_nlp_doc(con,"pt",doc); bam={r["position"]:a for r,a in zip(br,ba)}; pam={r["position"]:a for r,a in zip(pr,pa)}
        pairs=high_confidence_pairs(br,pr); rejected += len(set(r["sentence_id"] for r in br)&set(r["sentence_id"] for r in pr))-len(pairs); accepted+=len(pairs)
        for b,p in pairs:
            bf=sentence_features(b["source_bp"],bam[b["position"]]); pf=sentence_features(p["source_bp"],pam[p["position"]]); fold=stable_fold(doc)
            for family in families:
                xs=set(bf[family])&look_bp[family].keys(); ys=set(pf[family])&look_pt[family].keys()
                bx[family].update(xs); py[family].update(ys)
                for x in xs:
                    for y in ys:
                        key=(x,y); pair_counts[family][key]+=1; docs[family][key].add(doc); folds[family][key].add(fold); fold_supports[family][key][fold]+=1
                        if len(examples[family][key])<3: examples[family][key].append({"document_id":doc,"sentence_id":b["sentence_id"],"bp_text":b["source_bp"],"pt_text":p["source_bp"]})
        if di%300==0: print(f"ALIGNMENT documents={di}/{len(common)} accepted_pairs={accepted}",flush=True)
    result={}; by_bp={}
    for family in families:
        rows=[]
        for (x,y),support in pair_counts[family].items():
            docn=len(docs[family][(x,y)]); foldn=len(folds[family][(x,y)])
            if support<5 or docn<5 or foldn<3: continue
            conditional=support/bx[family][x]; baseline=py[family][y]/accepted; lift=conditional/baseline if baseline else 0
            if conditional<0.05 or lift<2.0: continue
            per_fold=fold_supports[family][(x,y)]; mean=statistics.mean(per_fold); subtype="general"
            if family=="surface_form" and x!=y and ((strip_accents(x)==strip_accents(y)) or (max(len(x),len(y))>=4 and edit_distance(x,y)<=2)): subtype="orthographic_near_form_candidate"
            rows.append({"bp_feature":x,"pt_feature":y,"feature_family":family,"association_subtype":subtype,"aligned_pair_support":support,"distinct_document_support":docn,"conditional_probability_pt_given_bp":conditional,"lift":lift,"pmi_bits":math.log2(lift) if lift>0 else None,"folds_with_support":foldn,"aligned_pair_support_by_fold":per_fold,"fold_support_coefficient_of_variation":statistics.pstdev(per_fold)/mean if mean else None,"examples":examples[family][(x,y)],"interpretation":"recurrent aligned contrast association; not a replacement rule"})
        rows.sort(key=lambda r:(r["distinct_document_support"]*math.log1p(max(r["lift"],0)),r["aligned_pair_support"]),reverse=True); result[family]=rows[:100 if family!="pos_trigram" else 80]
        best={}
        for r in result[family]:
            if r["bp_feature"] not in best: best[r["bp_feature"]]=r
        by_bp[family]=best
    return result,by_bp,{"accepted_sentence_pairs":accepted,"rejected_or_anomalous_id_matches":rejected,"requirements":"same document and unique exact sentence ID; both texts present; token-length ratio 0.45–2.2; no forced alignment"}


def quality_thresholds(bp:dict[str,Path])->dict[str,float]:
    lengths=[]
    for path in bp.values(): lengths.extend(len(surface_tokens(r["source_bp"])) for r in read_xml(path))
    lengths.sort()
    def q(p:float)->float: return lengths[min(int((len(lengths)-1)*p),len(lengths)-1)]
    return {"sentence_token_p99":q(.99),"sentence_token_p99_5":q(.995),"sentence_token_p99_9":q(.999),"hard_max_sentence_tokens":max(250,q(.999)*1.5)}


def quality_profile(rows:list[dict[str,Any]],thresholds:dict[str,float])->dict[str,Any]:
    lengths=[len(surface_tokens(r["source_bp"])) for r in rows]; n=len(rows); joined=" ".join(r["source_bp"] for r in rows)
    stage=sum(bool(STAGE_RE.search(r["source_bp"])) for r in rows); speakers=sum(bool(SPEAKER_RE.search(r["source_bp"])) for r in rows); suspicious=sum(bool(SUSPICIOUS_RE.search(r["source_bp"])) for r in rows)
    opens=sum(joined.count(x) for x in "([{“«"); closes=sum(joined.count(x) for x in ")] }”»".replace(" ","")); punctuation_imbalance=abs(opens-closes)
    straight_quotes=joined.count('"'); quote_imbalance=straight_quotes%2
    very_short=sum(x<=2 for x in lengths); flags=[]; severe=[]; penalty=0.0
    if max(lengths,default=0)>thresholds["hard_max_sentence_tokens"]: severe.append("extreme_sentence_length")
    elif max(lengths,default=0)>thresholds["sentence_token_p99_5"]: flags.append("long_sentence_outlier"); penalty+=10
    if stage/n>.35: flags.append("high_stage_direction_density"); penalty+=12
    if stage/n>.65: severe.append("overwhelming_stage_directions")
    if speakers/n>.45: flags.append("high_speaker_label_density"); penalty+=8
    if suspicious: severe.append("suspicious_unicode_or_control_character")
    if punctuation_imbalance>3: flags.append("punctuation_imbalance"); penalty+=5
    if quote_imbalance: flags.append("quote_imbalance"); penalty+=3
    if very_short/n>.5: flags.append("probable_segmentation_anomaly"); penalty+=10
    return {"sentence_token_counts":lengths,"maximum_sentence_tokens":max(lengths,default=0),"mean_sentence_tokens":statistics.mean(lengths) if lengths else 0,"extreme_sentence_length_outlier":max(lengths,default=0)>thresholds["sentence_token_p99_5"],"stage_direction_density":stage/n,"speaker_label_density":speakers/n,"suspicious_unicode_count":suspicious,"punctuation_imbalance":punctuation_imbalance,"quote_imbalance":bool(quote_imbalance),"probable_segmentation_anomaly":very_short/n>.5,"flags":flags,"hard_reject_reasons":severe,"quality_score":max(0.0,100.0-penalty-(100 if severe else 0))}


def assign_feature_families(features:dict[str,Any])->None:
    bp=feature_lookup(features,"bp_characteristic")
    for row in features["lemma"]["bp_characteristic"]: row["correlated_family"]="lexical::"+row["feature"]
    for row in features["surface_form"]["bp_characteristic"]: row["correlated_family"]="lexical::"+row.get("dominant_stanza_lemma",row["feature"])
    for family in ("word_bigram","word_trigram"):
        for row in features[family]["bp_characteristic"]:
            anchors=[bp["surface_form"][t] for t in row["feature"].split() if t in bp["surface_form"]]
            row["correlated_family"]=(max(anchors,key=lambda x:x["normalized_strength_within_family"])["correlated_family"] if anchors else "multiword::"+row["feature"])
    for row in features["pos_trigram"]["bp_characteristic"]: row["correlated_family"]="pos::"+row["feature"]


def evidence_for_sentence(text:str,ann:list[list[str]],lookup:dict[str,dict[str,dict[str,Any]]],associations:dict[str,dict[str,dict[str,Any]]])->list[dict[str,Any]]:
    feats=sentence_features(text,ann); evidence=[]
    for family,vals in feats.items():
        counts=Counter(v for v in vals if v in lookup[family])
        for feature,count in counts.items():
            row=lookup[family][feature]; assoc=associations.get(family,{}).get(feature)
            evidence.append({"feature":feature,"feature_view":family,"correlated_family":row["correlated_family"],"occurrences_in_sentence":count,"normalized_strength":row["normalized_strength_within_family"],"weighted_log_odds_z":row["weighted_log_odds_z"],"pointwise_kld_bits":row["pointwise_kld_bp_to_pt_bits"],"same_direction_folds":row["stability"]["same_direction_folds"],"bp_document_count":row["bp_document_count"],"aligned_association":({k:assoc[k] for k in ("pt_feature","aligned_pair_support","distinct_document_support","lift","folds_with_support")} if assoc else None)})
    return evidence


def score_window(rows:list[dict[str,Any]],thresholds:dict[str,float])->dict[str,Any]:
    q=quality_profile(rows,thresholds); family_entries=defaultdict(list); evidence_sentences=0; total_hits=0; views=set(); aligned=0; sentence_strengths=[]
    for row in rows:
        ev=row["contrast_evidence"]; sentence_families={}
        for e in ev:
            sentence_families[e["correlated_family"]]=max(sentence_families.get(e["correlated_family"],0.0),e["normalized_strength"])
        sentence_raw=sum(sentence_families.values()); sentence_strengths.append(1-math.exp(-sentence_raw/3.0))
        if sentence_raw>=0.5: evidence_sentences+=1
        for e in ev: family_entries[e["correlated_family"]].append(e); total_hits+=e["occurrences_in_sentence"]; views.add(e["feature_view"]); aligned+=bool(e["aligned_association"])
    contributions={}; stable=[]
    for family,entries in family_entries.items():
        hit_count=sum(e["occurrences_in_sentence"] for e in entries); base=max(e["normalized_strength"] for e in entries); repeat=min(2.0,1+0.35*math.log1p(hit_count)); view_count=len({e["feature_view"] for e in entries}); cross=1+0.08*(view_count-1); align=1.1 if any(e["aligned_association"] for e in entries) else 1.0
        contribution=base*repeat*cross*align; stable.append(max(e["same_direction_folds"] for e in entries)); contributions[family]={"hit_count":hit_count,"views":sorted({e["feature_view"] for e in entries}),"base_strength":base,"saturating_repeat_multiplier":repeat,"cross_view_multiplier":cross,"alignment_multiplier":align,"contribution":contribution}
    contribution_sum=sum(x["contribution"] for x in contributions.values())
    strength=45*(1-math.exp(-contribution_sum/20))
    density_ratio=statistics.mean(sentence_strengths) if sentence_strengths else 0.0; density=25*density_ratio
    effective_diversity=sum(1-math.exp(-x["contribution"]/0.35) for x in contributions.values())
    diversity=15*(1-math.exp(-effective_diversity/20))+10*(1-math.exp(-len(views)/2))
    quality=20*q["quality_score"]/100
    length_preference=max(0,len(rows)-8)*0.75
    total=strength+density+diversity+quality+length_preference
    concentration=max((x["contribution"] for x in contributions.values()),default=0)/max(sum(x["contribution"] for x in contributions.values()),1e-9)
    if q["hard_reject_reasons"] or q["quality_score"]<70 or concentration>.60: confidence="REVIEW-NEEDED"
    elif (statistics.mean(stable) if stable else 0)>=4.5 and len(views)>=3 and density_ratio>=.55 and aligned: confidence="HIGH"
    else: confidence="MEDIUM"
    return {"total_passage_score":total,"contrast_strength_component":strength,"contrast_density_component":density,"contrast_diversity_component":diversity,"source_quality_component":quality,"length_preference_component":length_preference,"evidence_bearing_sentence_count":evidence_sentences,"total_sentences":len(rows),"evidence_density":density_ratio,"sentence_evidence_strengths":sentence_strengths,"total_evidence_hits":total_hits,"distinct_contrast_family_count":len(family_entries),"effective_contrast_family_count":effective_diversity,"repeated_family_density":total_hits/max(len(family_entries),1),"evidence_views":sorted(views),"aligned_association_hits":aligned,"mean_feature_stability":statistics.mean(stable) if stable else 0,"largest_family_contribution_share":concentration,"family_contributions":contributions,"source_quality":q,"confidence":confidence}


def make_passage(doc:str,path:Path,rows:list[dict[str,Any]],profile:dict[str,Any],old:bool,updated:bool)->dict[str,Any]:
    concerns=list(profile["source_quality"]["flags"])+list(profile["source_quality"]["hard_reject_reasons"])
    if profile["largest_family_contribution_share"]>.60: concerns.append("evidence_concentrated_in_one_correlated_family")
    if profile["mean_feature_stability"]<4: concerns.append("some_evidence_is_fold-unstable")
    unique={}
    for sentence in rows:
        for evidence in sentence["contrast_evidence"]:
            key=(evidence["feature_view"],evidence["feature"])
            unique.setdefault(key,evidence)
    strongest=sorted(unique.values(),key=lambda e:(e["normalized_strength"],abs(e["weighted_log_odds_z"])),reverse=True)[:12]
    by_view={}
    for view in ALL_FAMILIES:
        items=[e for (v,_),e in unique.items() if v==view]
        if items:
            by_view[view]={
                "distinct_feature_count":len(items),
                "occurrence_count":sum(
                    e["occurrences_in_sentence"]
                    for sentence in rows for e in sentence["contrast_evidence"]
                    if e["feature_view"]==view
                ),
                "strongest_features":[e["feature"] for e in sorted(items,key=lambda x:x["normalized_strength"],reverse=True)[:8]],
            }
    aligned=[]
    for evidence in sorted(unique.values(),key=lambda e:e["normalized_strength"],reverse=True):
        if evidence["aligned_association"]:
            aligned.append({"bp_feature":evidence["feature"],"feature_view":evidence["feature_view"],**evidence["aligned_association"]})
    return {"passage_id":f"{doc}_s{rows[0]['sentence_id']}-s{rows[-1]['sentence_id']}","window_id":f"{doc}_s{rows[0]['sentence_id']}-s{rows[-1]['sentence_id']}","document_id":doc,"source_xml":str(path),"sentence_ids":[r["sentence_id"] for r in rows],"sentence_count":len(rows),"sentences":rows,"score_components":profile,"evidence_by_feature_view":by_view,"strongest_stable_features":strongest,"aligned_association_evidence":aligned,"provenance":{"old_seeded_miner":old,"updated_corpus_miner":updated,"supported_by_both":old and updated},"confidence":profile["confidence"],"concerns":sorted(set(concerns))}


def mine_passages(con:sqlite3.Connection,bp:dict[str,Path],features:dict[str,Any],associations:dict[str,dict[str,dict[str,Any]]],thresholds:dict[str,float],old_data:dict[str,Any],limit:int)->tuple[list[dict[str,Any]],list[dict[str,Any]],list[dict[str,Any]],dict[str,int]]:
    lookup=feature_lookup(features,"bp_characteristic"); old_by_doc={p["document_id"]:p for p in old_data["passages"]}; corpus_best=[]; old_scored=[]; quality_stats=Counter()
    for di,(doc,path) in enumerate(bp.items(),1):
        rows=read_xml(path); anns=load_nlp_doc(con,"bp",doc)
        for row,ann in zip(rows,anns): row["contrast_evidence"]=evidence_for_sentence(row["source_bp"],ann,lookup,associations)
        candidates=[]
        for length in WINDOW_LENGTHS:
            for start in range(len(rows)-length+1):
                subset=rows[start:start+length]; profile=score_window(subset,thresholds)
                if profile["source_quality"]["hard_reject_reasons"]: quality_stats["hard_rejected_windows"]+=1; continue
                candidates.append((profile["total_passage_score"],length,-start,subset,profile))
        if candidates:
            _,_,_,subset,profile=max(candidates,key=lambda x:x[:3]); corpus_best.append(make_passage(doc,path,subset,profile,False,True))
            if profile["source_quality"]["quality_score"]<=90: quality_stats["corpus_best_materially_quality_downranked"]+=1
        else:
            quality_stats["documents_with_all_windows_hard_rejected"]+=1
        old=old_by_doc.get(doc)
        if old:
            positions=[int(s["position"]) for s in old["sentences"]]; subset=[rows[p-1] for p in positions]; profile=score_window(subset,thresholds); updated=not profile["source_quality"]["hard_reject_reasons"] and profile["distinct_contrast_family_count"]>=3 and profile["evidence_bearing_sentence_count"]>=2
            old_scored.append(make_passage(doc,path,subset,profile,True,updated))
            if profile["source_quality"]["hard_reject_reasons"]: quality_stats["old_seeded_passages_hard_rejected"]+=1
        if di%250==0: print(f"PASSAGES documents={di}/{len(bp)}",flush=True)
    combined=corpus_best+old_scored; by_doc={}
    for row in combined:
        previous=by_doc.get(row["document_id"])
        if previous is None or row["score_components"]["total_passage_score"]>previous["score_components"]["total_passage_score"]: by_doc[row["document_id"]]=row
    eligible=[r for r in by_doc.values() if r["score_components"]["distinct_contrast_family_count"]>=3 and r["score_components"]["evidence_bearing_sentence_count"]>=2 and not r["score_components"]["source_quality"]["hard_reject_reasons"]]
    eligible.sort(key=lambda r:(r["score_components"]["total_passage_score"],r["passage_id"]),reverse=True); shortlist=eligible[:limit]
    old_ids={p["window_id"] for p in old_data["passages"]}
    for rank,row in enumerate(shortlist,1):
        row["rank"]=rank
        if row["window_id"] in old_ids: row["provenance"]["old_seeded_miner"]=True; row["provenance"]["supported_by_both"]=row["provenance"]["updated_corpus_miner"]
        if row["score_components"]["source_quality"]["flags"]: quality_stats["selected_with_quality_flags"]+=1
    # HIGH is deliberately selective within this broad review pool.  The
    # total score already combines strength, continuous sentence coverage,
    # effective diversity, alignment, stability, and quality; the top 30%
    # without review concerns receive HIGH, the remainder MEDIUM.
    non_review=[r for r in shortlist if r["confidence"]!="REVIEW-NEEDED"]
    ordered_scores=sorted(r["score_components"]["total_passage_score"] for r in non_review)
    high_threshold=ordered_scores[min(int(0.70*len(ordered_scores)),len(ordered_scores)-1)] if ordered_scores else math.inf
    for row in non_review:
        row["confidence"]="HIGH" if row["score_components"]["total_passage_score"]>=high_threshold else "MEDIUM"
        row["score_components"]["confidence"]=row["confidence"]
        row["score_components"]["confidence_calibration"]={"scope":"selected researcher-review pool","high_score_quantile":0.70,"high_score_threshold":high_threshold}
    selected_docs={r["document_id"] for r in shortlist}; quality_stats["corpus_best_excluded_by_union_ranking"]=sum(r["document_id"] not in selected_docs for r in corpus_best)
    return shortlist,corpus_best,old_scored,dict(quality_stats)


def feature_report(payload:dict[str,Any])->str:
    c=payload["corpus_comparability"]; n=payload["nlp_pipeline"]; m=payload["methodology"]; lines=[
        "# TED2020 corpus-driven BP/PT contrast features","",
        "> **Interpretation boundary:** these are source-selection signals. They are not annotation errors, required rewrites, unique EP replacements, candidates, or gold references.","",
        "## 1. Corpus basis","",
        f"- BP: `{payload['paths']['bp']}` — {c['bp_document_count']:,} documents and {c['bp_sentence_count']:,} sentences.",
        f"- PT: `{payload['paths']['pt']}` — {c['pt_document_count']:,} documents and {c['pt_sentence_count']:,} sentences.",
        f"- Primary comparison: {c['common_document_count']:,} matched TED talks; {c['bp_only_document_count']:,} BP-only and {c['pt_only_document_count']:,} PT-only documents.","",
        "## 2. BP/PT overlap and alignment quality","",
        f"- Exact sentence-ID overlap: {c['sentence_id_overlap_count']:,}/{c['sentence_id_union_count']:,} (Jaccard {c['sentence_id_jaccard']:.3f}).",
        f"- Common documents with segmentation/ID differences: {c['common_documents_with_segmentation_or_id_difference']:,}.",
        f"- High-confidence aligned pairs retained: {payload['alignment']['accepted_sentence_pairs']:,}; rejected/anomalous ID matches: {payload['alignment']['rejected_or_anomalous_id_matches']:,}.",
        "- Shared IDs are used only for recurrent association evidence; no forced alignment is performed.","",
        "## 3. NLP pipeline and Stanza model","",
        f"- Stanza `{n['stanza_version']}`; Portuguese **Bosque** package.",
        f"- Processors: {', '.join(n['processors'])}.",
        f"- Models: `{json.dumps(n['models'],ensure_ascii=False)}`.",
        "- Every original TED sentence is passed as one no-split unit; its original ID and exact text remain authoritative.",
        f"- Deviation from Alves: {n['deviation_from_alves']}","",
        "## 4. KLD methodology","",
        "Methodological basis: Diego Alves (2025), *Information Theory and Linguistic Variation: A Study of Brazilian and European Portuguese*, VarDial 2025 ([ACL Anthology](https://aclanthology.org/2025.vardial-1.2/)). This experiment adapts the analysis for source mining rather than reproducing the paper's FRMT study.","",
        f"For every feature view, both `D_KL(BP || PT)` and `D_KL(PT || BP)` are computed in bits. Jelinek–Mercer smoothing uses λ={m['jelinek_mercer_lambda']}. The background distribution is the pooled matched-talk distribution, which gives non-zero support while retaining corpus composition.","",
        "| View | D_KL(BP || PT) | D_KL(PT || BP) | BP events | PT events |",
        "|---|---:|---:|---:|---:|",
    ]
    for f,d in payload["divergence"].items(): lines.append(f"| {f} | {d['bp_to_pt_bits']:.8f} | {d['pt_to_bp_bits']:.8f} | {d['bp_total_events']:,} | {d['pt_total_events']:,} |")
    lines += ["","## 5. Smoothing and significance","",f"Feature-level Pearson χ² tests use `p < {SIGNIFICANCE_P}`. Significance is necessary but not sufficient: minimum count, document spread, effect direction, pointwise KLD, weighted log-odds, and stability are also retained.","","## 6. Cross-document stability","",f"Matched documents are assigned deterministically to {FOLD_COUNT} folds by SHA-256 of document ID. Direction is recomputed in five leave-one-fold-out replications; passage-ranking features must preserve direction in at least four.",""]
    sections=[("7. BP-characteristic surface forms","surface_form","bp_characteristic"),("8. PT-characteristic surface forms","surface_form","pt_characteristic"),("9. BP-characteristic lemmas","lemma","bp_characteristic"),("10. PT-characteristic lemmas","lemma","pt_characteristic"),("11. BP-characteristic POS trigrams","pos_trigram","bp_characteristic"),("12. PT-characteristic POS trigrams","pos_trigram","pt_characteristic")]
    for title,family,direction in sections:
        lines += [f"## {title}",""]
        rows=[r for r in payload["features"][family][direction] if r["selection_eligible"]][:25]
        for r in rows:
            point=r["pointwise_kld_bp_to_pt_bits"] if direction.startswith("bp") else r["pointwise_kld_pt_to_bp_bits"]
            lines += [f"### `{r['feature']}`","",f"- BP {r['bp_count']:,} ({r['bp_frequency_per_million']:.2f}/M), {r['bp_document_count']:,} documents; PT {r['pt_count']:,} ({r['pt_frequency_per_million']:.2f}/M), {r['pt_document_count']:,} documents.",f"- Pointwise KLD: {point:.10f} bits; weighted log-odds z: {r['weighted_log_odds_z']:.3f}; χ² p: {r['chi_square_p']:.3g}.",f"- Stability: {r['stability']['same_direction_folds']}/{FOLD_COUNT} leave-one-fold-out replications; effect SD {r['stability']['effect_stddev']:.4f}.",f"- Retrospective metadata only: old lexical inventory **{'yes' if r['in_old_lexical_inventory'] else 'no'}**; any earlier seeded detector **{'yes' if r['in_any_old_seeded_inventory'] else 'no'}**."]
            ex=r["examples"]["bp" if direction.startswith("bp") else "pt"][:2]
            for e in ex: lines.append(f"  - `{e['document_id']}:{e['sentence_id']}` — {e['source_text'][:220]}")
            lines.append("")
    lines += ["## 13. Supplemental multiword features","","Word bigrams and trigrams remain visible, but passage scoring groups overlaps with their strongest lexical family and uses saturating repetition contributions.",""]
    for family in ("word_bigram","word_trigram"):
        vals=[r for r in payload["features"][family]["bp_characteristic"] if r["selection_eligible"]][:20]
        lines += [f"### {family}","",", ".join(f"`{r['feature']}`" for r in vals),""]
    for heading,families in (("14. Aligned lexical / orthographic associations",("surface_form","lemma")),("15. Aligned structural associations",("pos_trigram",))):
        lines += [f"## {heading}","","> Recurrent aligned contrast associations are source-selection evidence, not replacement rules.",""]
        rows=[r for f in families for r in payload["aligned_associations"][f]]; rows.sort(key=lambda r:(r["distinct_document_support"]*math.log1p(r["lift"]),r["aligned_pair_support"]),reverse=True)
        for r in rows[:25]: lines.append(f"- `{r['bp_feature']}` ↔ `{r['pt_feature']}` ({r['feature_family']}; {r['association_subtype']}): {r['aligned_pair_support']} pairs / {r['distinct_document_support']} documents; P(PT|BP)={r['conditional_probability_pt_given_bp']:.3f}; lift={r['lift']:.2f}; folds={r['folds_with_support']}/{FOLD_COUNT}; fold-support CV={r['fold_support_coefficient_of_variation']:.3f}.")
        lines.append("")
    lines += ["## 16. Stability diagnostics","","Stable high-frequency evidence is retained rather than penalized for frequency. Features reversing across folds remain in the machine-readable report with lower confidence and are excluded from passage ranking.","","## 17. Optional dependency / word-order results","","Not computed. The required surface, real-lemma, and real-POS-trigram views were prioritized; no heuristic dependency or POS labels were invented.","","## 18. Translation-bias limitations","","Matched talks control topic, but TED translations can encode translator conventions, paraphrase, age, transcription, or segmentation effects. Document spread, fold stability, aligned recurrence, multiple views, and researcher review reduce—but cannot eliminate—this risk.","","## 19. Source-mining interpretation boundary","","Source mining asks whether a passage is likely to contain useful BP/EP contrastive material. Later cESA annotation asks whether a specific model output requires correction. Statistical distinctiveness in the source does not predetermine the annotation judgment.",""]
    return "\n".join(lines)


def shortlist_report(payload:dict[str,Any])->str:
    lines=["# TED2020 revised corpus-contrast researcher-review pool","","> Broad review pool only. No final benchmark is frozen, and PT TED is not annotation gold.","",f"Passages: **{payload['passage_count']}** from distinct talks. Ranking combines contrast strength, density, diversity, fold stability, aligned evidence, saturating repetition, and source quality.",""]
    for p in payload["passages"]:
        s=p["score_components"]; prov=p["provenance"]
        view_summary="; ".join(f"{view}: {data['distinct_feature_count']} features/{data['occurrence_count']} hits" for view,data in p["evidence_by_feature_view"].items())
        strongest=", ".join(f"`{e['feature']}` ({e['feature_view']})" for e in p["strongest_stable_features"][:8])
        lines += [f"## {p['rank']}. `{p['passage_id']}`","",f"- Document: `{p['document_id']}`; sentence IDs: {', '.join(p['sentence_ids'])}; count: {p['sentence_count']}.",f"- Total **{s['total_passage_score']:.3f}** = strength {s['contrast_strength_component']:.3f} + density {s['contrast_density_component']:.3f} + diversity {s['contrast_diversity_component']:.3f} + quality {s['source_quality_component']:.3f} + length {s['length_preference_component']:.3f}.",f"- Evidence: {s['evidence_bearing_sentence_count']}/{s['total_sentences']} sentences; density {s['evidence_density']:.2f}; {s['distinct_contrast_family_count']} families; aligned hits {s['aligned_association_hits']}.",f"- Evidence by view: {view_summary}.",f"- Strongest stable features: {strongest}.",f"- Provenance: old seeded miner **{'yes' if prov['old_seeded_miner'] else 'no'}**; updated corpus miner **{'yes' if prov['updated_corpus_miner'] else 'no'}**; both **{'yes' if prov['supported_by_both'] else 'no'}**.",f"- Confidence: **{p['confidence']}**; concerns: {', '.join(p['concerns']) if p['concerns'] else 'none'}.","",]
        for r in p["sentences"]:
            key=sorted(r["contrast_evidence"],key=lambda e:e["normalized_strength"],reverse=True)[:6]
            detail="; ".join(f"`{e['feature']}` ({e['feature_view']}, {e['same_direction_folds']}/5 folds)" for e in key) or "none"
            lines += [f"**{r['sentence_id']}.** {r['source_bp']}",f"  - Key evidence: {detail}",""]
    return "\n".join(lines)


def compare_report(shortlist:list[dict[str,Any]],old_scored:list[dict[str,Any]],features:dict[str,Any],divergence:dict[str,Any],associations:dict[str,list[dict[str,Any]]],quality_stats:dict[str,int],previous_audit_ids:set[str])->str:
    old_ids={p["window_id"] for p in old_scored}; exact=[p for p in shortlist if p["window_id"] in old_ids]; old_docs={p["document_id"] for p in old_scored}; same_docs=[p for p in shortlist if p["document_id"] in old_docs]; both=[p for p in shortlist if p["provenance"]["supported_by_both"]]
    structural=[p for p in shortlist if "pos_trigram" in p["score_components"]["evidence_views"]]; mixed=[p for p in shortlist if {"lemma","pos_trigram"}.issubset(p["score_components"]["evidence_views"])]
    def pos_share(p:dict[str,Any])->float:
        values=p["score_components"]["family_contributions"]
        total=sum(v["contribution"] for v in values.values())
        return sum(v["contribution"] for k,v in values.items() if k.startswith("pos::"))/max(total,1e-9)
    structural_ranked=sorted(structural,key=lambda p:(pos_share(p),p["score_components"]["total_passage_score"]),reverse=True)
    selected_ids={p["passage_id"] for p in shortlist}; recovered=sorted(selected_ids&previous_audit_ids); unstable=[]
    for family in MAIN_FAMILIES:
        unstable.extend((family,r) for d in ("bp_characteristic","pt_characteristic") for r in features[family][d] if r["significant_p_lt_0_001"] and r["stability"]["same_direction_folds"]<4)
    unstable.sort(key=lambda x:abs(x[1]["weighted_log_odds_z"]),reverse=True)
    eligible=lambda family,direction:[r for r in features[family][direction] if r["selection_eligible"]]
    known=sorted((r for r in eligible("lemma","bp_characteristic") if r["in_any_old_seeded_inventory"]),key=lambda r:r["pointwise_kld_bp_to_pt_bits"],reverse=True)[:10]
    novel=sorted((r for family in ("surface_form","lemma") for r in eligible(family,"bp_characteristic") if not r["in_any_old_seeded_inventory"]),key=lambda r:r["pointwise_kld_bp_to_pt_bits"],reverse=True)[:12]
    pos_top=sorted(eligible("pos_trigram","bp_characteristic"),key=lambda r:r["pointwise_kld_bp_to_pt_bits"],reverse=True)[:12]
    frequent=sorted((r for family in MAIN_FAMILIES for r in eligible(family,"bp_characteristic") if r["stability"]["same_direction_folds"]==FOLD_COUNT),key=lambda r:r["bp_count"],reverse=True)[:12]
    lexical_assoc=sorted(associations["surface_form"]+associations["lemma"],key=lambda r:(r["distinct_document_support"]*math.log1p(r["lift"]),r["aligned_pair_support"]),reverse=True)
    structural_assoc=sorted(associations["pos_trigram"],key=lambda r:(r["distinct_document_support"]*math.log1p(r["lift"]),r["aligned_pair_support"]),reverse=True)
    recovered_text=", ".join(f"`{x}`" for x in recovered) or "none"
    known_text=", ".join(f"`{r['feature']}`" for r in known) or "none"
    novel_text=", ".join(f"`{r['feature']}`" for r in novel) or "none"
    pos_text=", ".join(f"`{r['feature']}`" for r in pos_top) or "none"
    pos_passages=", ".join(f"`{p['passage_id']}` ({pos_share(p):.1%} of family contribution)" for p in structural_ranked[:10]) or "none"
    lexical_assoc_text=", ".join(f"`{r['bp_feature']}` ↔ `{r['pt_feature']}` ({r['distinct_document_support']} docs)" for r in lexical_assoc[:8]) or "none"
    structural_assoc_text=", ".join(f"`{r['bp_feature']}` ↔ `{r['pt_feature']}` ({r['distinct_document_support']} docs)" for r in structural_assoc[:8]) or "none"
    frequent_text=", ".join(f"`{r['feature']}` ({r['bp_count']:,})" for r in frequent) or "none"
    lines=["# Comparison of TED2020 source-mining passes","","> PT TED remains discovery evidence only; none of these contrasts is a conservative rewrite rule or annotator-visible reference.","","## 1–3. Overlap, recovered original passages, and additions","",f"The revised union pool has **{len(shortlist)}** passages: **{len(exact)}** exact original windows, **{len(same_docs)}** talks represented in the original 50, and **{len(both)}** passages supported by both miners.",f"New passages outside original talks: **{len(shortlist)-len(same_docs)}**. Previous negative-audit windows recovered exactly: **{len(recovered)}** ({recovered_text}).","","## 4–5. Lemma recovery and independently discovered lexical/orthographic signals","",f"Stable BP lemmas retrospectively matching an old detector include: {known_text}.",f"Strong stable BP surface/lemma signals not matched by the old inventory include: {novel_text}.","","## 6–8. POS-trigram structure and cross-view passages","",f"Strong stable BP POS trigrams include: {pos_text}.",f"Passages with POS-trigram evidence: **{len(structural)}**; with both lemma and POS-trigram evidence: **{len(mixed)}**.",f"Passages most strengthened by POS evidence: {pos_passages}.","","## 9–10. Bidirectional divergence and complementary information",""]
    for f in MAIN_FAMILIES:
        d=divergence[f]; lines.append(f"- {f}: D_KL(BP||PT)={d['bp_to_pt_bits']:.8f} bits; D_KL(PT||BP)={d['pt_to_bp_bits']:.8f} bits.")
    lines += ["","Both directions are retained because asymmetry changes the feature contribution and exposes complementary BP- and PT-characteristic evidence.","","## 11–12. Recurrent aligned contrast associations","",f"Lexical/orthographic associations retained: **{len(lexical_assoc)}**. Leading examples: {lexical_assoc_text}.",f"Structural POS-trigram associations retained: **{len(structural_assoc)}**. Leading examples: {structural_assoc_text}.","","## 13–14. Stability","",f"Stable frequent evidence was retained when it preserved direction in all folds: {frequent_text}.","Statistically significant candidates with fewer than four stable replications are excluded from passage ranking; leading cases are listed below.","","### Strong revised candidates",""]
    for p in shortlist[:15]: lines.append(f"- `{p['passage_id']}` — {p['score_components']['total_passage_score']:.2f}; {p['score_components']['distinct_contrast_family_count']} families; {p['score_components']['evidence_bearing_sentence_count']}/{p['sentence_count']} evidence-bearing sentences; {p['confidence']}.")
    lines += ["","### Statistically strong but unstable examples",""]
    for family,row in unstable[:15]: lines.append(f"- `{row['feature']}` ({family}, {row['direction']}): |z|={abs(row['weighted_log_odds_z']):.2f}; direction preserved in {row['stability']['same_direction_folds']}/{FOLD_COUNT} replications; excluded from passage ranking.")
    if not unstable: lines.append("- None among retained candidates.")
    lines += ["","## 15. Source-quality intervention","",f"Source-quality statistics: `{json.dumps(quality_stats,sort_keys=True)}`. Severe failures are rejected; moderate flags apply a bounded score penalty and remain reviewable.","","## 16. Previous negative-audit recovery","",f"The revised shortlist recovered **{len(recovered)}** exact windows from the preceding deterministic negative audit. The new audit remains diagnostic and requires researcher inspection for residual false negatives.","","## 17. Researcher-pool assessment","","The revision materially broadens the evidence used for human source review: real Bosque lemmas and UPOS trigrams supplement surface evidence, while bidirectional KLD, document-level stability, aligned recurrence, saturation, and quality controls reduce obvious failure modes. It does **not** establish that every signal is an obligatory BP→EP change, and it does not freeze a benchmark.","","## Limitations","","- PT and BP TED texts are independent translations, not conservative rewrites.","- Pointwise signals may encode translator conventions despite stability controls.","- Aligned associations use only conservative exact-ID pairs and are not replacement rules.","- Dependency and word-order entropy were omitted to prioritize the required feature views.","- Human review remains necessary before source freezing.",""]
    return "\n".join(lines)


def negative_audit(corpus_best:list[dict[str,Any]],old_scored:list[dict[str,Any]],size:int,seed:int)->str:
    rng=random.Random(seed); ordered=sorted(corpus_best,key=lambda p:p["score_components"]["total_passage_score"]); zero=[p for p in ordered if p["score_components"]["distinct_contrast_family_count"]==0]; middle=ordered[len(ordered)//3:2*len(ordered)//3]; old_low=sorted(old_scored,key=lambda p:p["score_components"]["total_passage_score"]); concern=[p for p in reversed(ordered) if p["confidence"]=="REVIEW-NEEDED" or p["concerns"]]
    def view_share(p:dict[str,Any],prefix:str)->float:
        contributions=p["score_components"]["family_contributions"]
        total=sum(v["contribution"] for v in contributions.values())
        return sum(v["contribution"] for k,v in contributions.items() if k.startswith(prefix))/max(total,1e-9)
    structural=sorted((p for p in corpus_best if view_share(p,"pos::")>=.55 and view_share(p,"lexical::")<=.35),key=lambda p:(view_share(p,"pos::"),p["score_components"]["total_passage_score"]),reverse=True)
    low=zero if zero else ordered[:5]
    groups=[("Zero / very low",rng.sample(low,min(5,len(low)))),("Middle-ranked",rng.sample(middle,min(5,len(middle)))),("Original seeded but revised-low",old_low[:5]),("High-score with quality/diversity/stability concern",concern[:5]),("Raised by structural POS evidence",structural[:4])]
    lines=["# TED2020 revised source-mining negative audit","","> Deterministic diagnostic sample only—not benchmark material and not automatic promotion evidence.","",f"Seed: `{seed}`.",""]
    for title,rows in groups:
        lines += [f"## {title}",""]
        if not rows: lines += ["No eligible example in this category.",""]; continue
        for p in rows:
            s=p["score_components"]; lines += [f"### `{p['passage_id']}`","",f"Score {s['total_passage_score']:.2f}; {s['distinct_contrast_family_count']} families; views {', '.join(s['evidence_views']) or 'none'}; confidence {p['confidence']}; concerns {', '.join(p['concerns']) or 'none'}.",""]
            for r in p["sentences"]: lines.append(f"- **{r['sentence_id']}** {r['source_bp']}")
            lines.append("")
    return "\n".join(lines)


def archive_info(corpus:Path)->dict[str,Any]|None:
    path=corpus.parents[2]/f"{corpus.name}.zip"
    if not path.is_file(): return None
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return {"path":str(path),"size_bytes":path.stat().st_size,"sha256":h.hexdigest()}


def main()->None:
    args=parse_args(); bp=corpus_paths(args.bp_corpus); pt=corpus_paths(args.pt_corpus); comp=comparability(bp,pt); common=comp["common_document_ids"]
    comp["bp_sentence_count"]=sum(len(read_xml(p)) for p in bp.values()); comp["pt_sentence_count"]=sum(len(read_xml(p)) for p in pt.values())
    con=init_cache(args.nlp_cache); nlp=build_nlp_cache(con,bp,pt,args.stanza_model_dir,args.cache_shard_count,args.cache_shard_index)
    if args.cache_only:
        print(json.dumps({"cache":str(args.nlp_cache),"shard_index":args.cache_shard_index,"shard_count":args.cache_shard_count,"documents":con.execute("SELECT COUNT(*) FROM documents").fetchone()[0]},indent=2)); return
    agg,proper=aggregate_matched(con,bp,pt,common); divergence,features=initial_features(agg,proper); enrich_stability_and_examples(con,bp,pt,common,agg,features); assign_feature_families(features)
    associations,assoc_by_bp,alignment=aligned_associations(con,bp,pt,common,features); thresholds=quality_thresholds(bp); old=json.loads(args.old_shortlist.read_text(encoding="utf-8"))
    previous_audit_path=args.output_dir/"ted2020_source_mining_negative_audit.md"; previous_audit_ids=set(re.findall(r"### `([^`]+)`",previous_audit_path.read_text(encoding="utf-8"))) if previous_audit_path.exists() else set()
    shortlist,corpus_best,old_scored,quality_stats=mine_passages(con,bp,features,assoc_by_bp,thresholds,old,args.shortlist_size)
    feature_payload={"status":"revised_researcher_review_source_mining_not_gold_not_final","paths":{"bp":str(args.bp_corpus),"pt":str(args.pt_corpus)},"archives":{"bp":archive_info(args.bp_corpus),"pt":archive_info(args.pt_corpus)},"pt_role":"matched-topic contrast corpus only; never conservative gold/reference","corpus_comparability":comp,"nlp_pipeline":nlp,"methodology":{"basis":"Alves (2025), adapted for source mining","jelinek_mercer_lambda":JM_LAMBDA,"log_base":"2 (bits)","smoothing_background":"pooled matched-talk feature distribution","significance_test":"Pearson chi-square 2x2, df=1","significance_threshold":SIGNIFICANCE_P,"stability":"five deterministic document folds; leave-one-fold-out direction","feature_thresholds":SETTINGS,"dependency_analysis":"not run","word_order_entropy":"not run"},"divergence":divergence,"features":features,"aligned_associations":associations,"alignment":alignment,"source_quality_thresholds":thresholds,"interpretation_boundary":"source-selection evidence only; not annotation rules or required rewrites"}
    shortlist_payload={"status":"broad_researcher_review_pool_not_final_benchmark","source_language":"pt-BR","pt_contrast_role":"source discovery only; no annotator-visible reference","passage_count":len(shortlist),"source_quality_statistics":quality_stats,"passages":shortlist}
    args.output_dir.mkdir(parents=True,exist_ok=True); paths={"features_json":args.output_dir/"ted2020_corpus_contrast_features.json","features_md":args.output_dir/"ted2020_corpus_contrast_features.md","shortlist_json":args.output_dir/"ted2020_corpus_contrast_shortlist.json","shortlist_md":args.output_dir/"ted2020_corpus_contrast_shortlist.md","comparison_md":args.output_dir/"ted2020_source_mining_comparison.md","negative_md":args.output_dir/"ted2020_source_mining_negative_audit.md"}
    paths["features_json"].write_text(json.dumps(feature_payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); paths["features_md"].write_text(feature_report(feature_payload),encoding="utf-8"); paths["shortlist_json"].write_text(json.dumps(shortlist_payload,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); paths["shortlist_md"].write_text(shortlist_report(shortlist_payload),encoding="utf-8"); paths["comparison_md"].write_text(compare_report(shortlist,old_scored,features,divergence,associations,quality_stats,previous_audit_ids),encoding="utf-8"); paths["negative_md"].write_text(negative_audit(corpus_best,old_scored,args.negative_audit_size,args.seed),encoding="utf-8")
    summary={"stanza":nlp,"corpus":{k:comp[k] for k in ("bp_document_count","pt_document_count","common_document_count","bp_sentence_count","pt_sentence_count","sentence_id_overlap_count")},"divergence":divergence,"retained":{f:{d:sum(r["selection_eligible"] for r in features[f][d]) for d in ("bp_characteristic","pt_characteristic")} for f in ALL_FAMILIES},"aligned_associations":{f:len(v) for f,v in associations.items()},"passages":len(shortlist),"exact_old":sum(p["provenance"]["old_seeded_miner"] for p in shortlist),"supported_by_both":sum(p["provenance"]["supported_by_both"] for p in shortlist),"quality":quality_stats,"outputs":{k:str(v) for k,v in paths.items()}}
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
