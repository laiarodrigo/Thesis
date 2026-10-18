#!/usr/bin/env python3
"""Independently validate the revised TED2020 contrast-mining artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sqlite3
from pathlib import Path

from mine_ted2020_corpus_contrasts import (
    ALL_FAMILIES, FOLD_COUNT, JM_LAMBDA, SIGNIFICANCE_P,
    aggregate_matched, aligned_associations, chi_square, comparability,
    corpus_paths, enrich_stability_and_examples, evidence_for_sentence,
    feature_lookup, high_confidence_pairs, load_nlp_doc, quality_profile,
    read_xml, score_window, sentence_features, smoothed_prob, stable_fold,
)


def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("cmu_evaluator_project/data/passage_selection"))
    p.add_argument("--bp-corpus",type=Path,default=Path("data/external/TED2020/TED2020/raw/pt_br"))
    p.add_argument("--pt-corpus",type=Path,default=Path("data/external/TED2020/TED2020/raw/pt"))
    p.add_argument("--nlp-cache",type=Path,default=Path("/tmp/ted2020-stanza-bosque.sqlite3"))
    return p.parse_args()


def close(a:float,b:float,tol:float=1e-9)->bool:
    return math.isclose(a,b,rel_tol=tol,abs_tol=tol)


def main()->None:
    a=args(); feature=json.loads((a.root/"ted2020_corpus_contrast_features.json").read_text()); shortlist=json.loads((a.root/"ted2020_corpus_contrast_shortlist.json").read_text())
    bp=corpus_paths(a.bp_corpus); pt=corpus_paths(a.pt_corpus); common=sorted(set(bp)&set(pt)); recorded=feature["corpus_comparability"]; recount=comparability(bp,pt)
    for key in ("bp_document_count","pt_document_count","common_document_count","bp_only_document_count","pt_only_document_count","matched_bp_sentence_count","matched_pt_sentence_count","sentence_id_overlap_count","sentence_id_union_count","common_documents_same_sentence_ids"):
        assert recount[key]==recorded[key],key
    assert feature["methodology"]["jelinek_mercer_lambda"]==JM_LAMBDA
    assert feature["methodology"]["significance_threshold"]==SIGNIFICANCE_P
    assert feature["nlp_pipeline"]["treebank_package"]=="Bosque"
    assert feature["nlp_pipeline"]["models"]["pos"]=="bosque_nocharlm"
    assert feature["nlp_pipeline"]["models"]["lemma"]=="bosque_nocharlm"

    con=sqlite3.connect(a.nlp_cache)
    cached={(r[0],r[1]):r[2] for r in con.execute("SELECT lang,document_id,sentence_count FROM documents")}
    assert len(cached)==len(bp)+len(pt)
    for lang,paths in (("bp",bp),("pt",pt)):
        for doc,path in paths.items():
            rows=read_xml(path)
            assert cached[(lang,doc)]==len(rows)
            saved=list(con.execute(
                "SELECT position,sentence_id,text_sha256 FROM sentences "
                "WHERE lang=? AND document_id=? ORDER BY position",(lang,doc)
            ))
            assert len(saved)==len(rows)
            for row,(position,sentence_id,text_sha256) in zip(rows,saved):
                assert position==row["position"] and sentence_id==row["sentence_id"]
                assert text_sha256==hashlib.sha256(row["source_bp"].encode()).hexdigest()

    # Independently rebuild distributions, aggregate bidirectional KLD, and
    # feature-level pointwise values from the exact cached Bosque output.
    agg,_=aggregate_matched(con,bp,pt,common)
    for family in ALL_FAMILIES:
        data=agg[family]; bt=data["bp_total"]; tt=data["pt_total"]; db=dt=0.0
        for value in set(data["bp"])|set(data["pt"]):
            pooled=(data["bp"][value]+data["pt"][value])/(bt+tt); pb=smoothed_prob(data["bp"][value],bt,pooled); pp=smoothed_prob(data["pt"][value],tt,pooled)
            db+=pb*math.log2(pb/pp); dt+=pp*math.log2(pp/pb)
        assert close(db,feature["divergence"][family]["bp_to_pt_bits"])
        assert close(dt,feature["divergence"][family]["pt_to_bp_bits"])
        for direction in ("bp_characteristic","pt_characteristic"):
            for row in feature["features"][family][direction]:
                value=row["feature"]; bc=data["bp"][value]; tc=data["pt"][value]; pooled=(bc+tc)/(bt+tt); pb=smoothed_prob(bc,bt,pooled); pp=smoothed_prob(tc,tt,pooled)
                assert bc==row["bp_count"] and tc==row["pt_count"]
                assert close(pb*math.log2(pb/pp),row["pointwise_kld_bp_to_pt_bits"])
                assert close(pp*math.log2(pp/pb),row["pointwise_kld_pt_to_bp_bits"])
                chi,p=chi_square(bc,tc,bt,tt); assert close(chi,row["chi_square"]); assert close(p,row["chi_square_p"]); assert (p<SIGNIFICANCE_P)==row["significant_p_lt_0_001"]

    # Reproduce deterministic document-fold stability from the cache.
    folds=[set() for _ in range(FOLD_COUNT)]
    for doc in common: folds[stable_fold(doc)].add(doc)
    assert sum(len(x) for x in folds)==len(common)
    assert not any(folds[i]&folds[j] for i in range(FOLD_COUNT) for j in range(i+1,FOLD_COUNT))
    copied=copy.deepcopy(feature["features"]); enrich_stability_and_examples(con,bp,pt,common,agg,copied)
    for family in ALL_FAMILIES:
        rebuilt={r["feature"]:r for d in ("bp_characteristic","pt_characteristic") for r in copied[family][d]}
        for row in [r for d in ("bp_characteristic","pt_characteristic") for r in feature["features"][family][d]]:
            other=rebuilt[row["feature"]]; assert row["stability"]["fold_directions"]==other["stability"]["fold_directions"]
            assert row["stability"]["same_direction_folds"]==other["stability"]["same_direction_folds"]
            assert all(close(x,y) for x,y in zip(row["stability"]["log2_ratio_by_replication"],other["stability"]["log2_ratio_by_replication"]))

    # Verify every saved example maps to exact source text and actual Stanza
    # lemma/POS output, never heuristic labels.
    checked_examples=0
    for family in ALL_FAMILIES:
        for direction in ("bp_characteristic","pt_characteristic"):
            for row in feature["features"][family][direction]:
                for lang in ("bp","pt"):
                    paths=bp if lang=="bp" else pt
                    for ex in row["examples"][lang]:
                        rows=read_xml(paths[ex["document_id"]]); source=rows[ex["position"]-1]; anns=load_nlp_doc(con,lang,ex["document_id"]); ann=anns[ex["position"]-1]
                        assert source["sentence_id"]==ex["sentence_id"] and source["source_bp"]==ex["source_text"]
                        digest=hashlib.sha256(json.dumps(ann,ensure_ascii=False,separators=(",",":")).encode()).hexdigest(); assert digest==ex["nlp_annotation_sha256"]
                        assert row["feature"] in sentence_features(source["source_bp"],ann)[family]
                        checked_examples+=1

    # Associations must come from real conservative exact-ID pairs.
    rebuilt_associations,_,rebuilt_alignment=aligned_associations(con,bp,pt,common,copied)
    assert rebuilt_alignment==feature["alignment"]
    checked_associations=0
    for family,rows in feature["aligned_associations"].items():
        rebuilt={(r["bp_feature"],r["pt_feature"]):r for r in rebuilt_associations[family]}
        for assoc in rows:
            other=rebuilt[(assoc["bp_feature"],assoc["pt_feature"])]
            assert assoc["distinct_document_support"]>=5 and assoc["folds_with_support"]>=3
            assert assoc["conditional_probability_pt_given_bp"]>=0.05 and assoc["lift"]>=2.0
            assert len(assoc["aligned_pair_support_by_fold"])==FOLD_COUNT
            assert sum(assoc["aligned_pair_support_by_fold"])==assoc["aligned_pair_support"]
            assert sum(value>0 for value in assoc["aligned_pair_support_by_fold"])==assoc["folds_with_support"]
            for key in ("aligned_pair_support","distinct_document_support","folds_with_support","aligned_pair_support_by_fold"):
                assert assoc[key]==other[key]
            for key in ("conditional_probability_pt_given_bp","lift","fold_support_coefficient_of_variation"):
                assert close(assoc[key],other[key])
            for ex in assoc["examples"]:
                assert ex["document_id"] in bp and ex["document_id"] in pt
                pairs=high_confidence_pairs(read_xml(bp[ex["document_id"]]),read_xml(pt[ex["document_id"]]))
                pair=next((x for x in pairs if x[0]["sentence_id"]==ex["sentence_id"]),None); assert pair
                assert pair[0]["source_bp"]==ex["bp_text"] and pair[1]["source_bp"]==ex["pt_text"]
                checked_associations+=1

    # Recompute all shortlisted sentence evidence, quality diagnostics, and
    # score components against exact BP XML and cached Bosque annotations.
    lookup=feature_lookup(feature["features"],"bp_characteristic")
    assoc_by_bp={family:{} for family in ALL_FAMILIES}
    for family,rows in feature["aligned_associations"].items():
        for row in rows:
            assoc_by_bp[family].setdefault(row["bp_feature"],row)
    checked_sentences=0
    for passage in shortlist["passages"]:
        rows=read_xml(bp[passage["document_id"]]); anns=load_nlp_doc(con,"bp",passage["document_id"]); positions=[r["position"] for r in passage["sentences"]]
        assert positions==list(range(positions[0],positions[0]+len(positions)))
        rebuilt=[]
        for saved in passage["sentences"]:
            source=rows[saved["position"]-1]; assert source["sentence_id"]==saved["sentence_id"] and source["source_bp"]==saved["source_bp"]
            source["contrast_evidence"]=evidence_for_sentence(source["source_bp"],anns[source["position"]-1],lookup,assoc_by_bp); rebuilt.append(source); checked_sentences+=1
        score=score_window(rebuilt,feature["source_quality_thresholds"]); saved=passage["score_components"]
        for key in ("total_passage_score","contrast_strength_component","contrast_density_component","contrast_diversity_component","source_quality_component","length_preference_component","evidence_density","repeated_family_density","largest_family_contribution_share"):
            assert close(score[key],saved[key]),(passage["passage_id"],key,score[key],saved[key])
        assert score["distinct_contrast_family_count"]==saved["distinct_contrast_family_count"]
        assert score["distinct_contrast_family_count"]==len(score["family_contributions"])
        assert score["source_quality"]==saved["source_quality"]
        for value in score["family_contributions"].values(): assert value["saturating_repeat_multiplier"]<=2.0 and value["saturating_repeat_multiplier"]>=1.0

    for row in feature["features"]["lemma"]["bp_characteristic"]:
        assert row["correlated_family"]=="lexical::"+row["feature"]
    for row in feature["features"]["pos_trigram"]["bp_characteristic"]:
        assert row["correlated_family"]=="pos::"+row["feature"]

    serialized=json.dumps(shortlist,ensure_ascii=False).casefold()
    assert '"source_pt"' not in serialized and '"reference"' not in serialized and '"ref"' not in serialized
    assert shortlist["status"]=="broad_researcher_review_pool_not_final_benchmark"
    confidence={p["confidence"] for p in shortlist["passages"]}
    assert confidence.issubset({"HIGH","MEDIUM","REVIEW-NEEDED"}) and len(confidence)>=2
    for passage in shortlist["passages"]:
        if passage["confidence"]=="HIGH":
            calibration=passage["score_components"]["confidence_calibration"]
            assert passage["score_components"]["total_passage_score"]>=calibration["high_score_threshold"]
    print("VALIDATION_STATUS=PASS")
    print(f"corpus_documents={len(bp)}_bp,{len(pt)}_pt,{len(common)}_common")
    print(f"feature_examples_checked={checked_examples}")
    print(f"association_examples_checked={checked_associations}")
    print(f"passages_checked={len(shortlist['passages'])}")
    print(f"exact_bp_sentences_checked={checked_sentences}")
    print("bidirectional_kld=RECOMPUTED")
    print("significance=RECOMPUTED")
    print("five_fold_stability=RECOMPUTED")
    print("pt_annotation_reference_fields=0")
    print("final_benchmark_frozen=False")


if __name__=="__main__": main()
