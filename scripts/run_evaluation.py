"""Evaluate the 3 Progressive Literature Kinetics Mining Pipelines on the Gold Benchmark.

Pipelines evaluated:
  1. Baseline Single-Pass Extractor (`mode='baseline'`)
  2. Extractor + RDKit Structure Gate (`mode='rdkit_gate'`)
  3. ChemLit-Kinetics-Miner (`mode='full'`: Dual-Verification + RDKit + Unit/Log/Speciation Audit)

Outputs:
  - `results/miner_benchmark.json`: Aggregate & per-pipeline metrics + parity audit trace
  - `results/error_breakdown.csv`: Failure mode taxonomy x chemical family breakdown
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemlit_miner.chem_validator import lookup_reference_compound
from chemlit_miner.cross_verifier import CrossVerificationEngine, PipelineMode
from chemlit_miner.schema import ReactionKineticsRecord


PIPELINE_SPECS: List[Tuple[PipelineMode, str]] = [
    ("baseline", "Baseline Single-Pass Extractor"),
    ("rdkit_gate", "Extractor + RDKit Structure Gate"),
    (
        "full",
        "ChemLit-Kinetics-Miner (Full Dual-Verification + RDKit + Physics Audit)",
    ),
]


def canonical_name(name: str) -> str:
    ref = lookup_reference_compound(name)
    return ref.name if ref else name.strip()


def record_matches_gold(rec: ReactionKineticsRecord, gold: Dict[str, Any]) -> bool:
    """Strict joint verification of compound identity, oxidant, standardized k, inequality, pH, and T."""
    if canonical_name(rec.compound_name).lower() != gold["compound_name"].lower():
        return False
    # If RDKit InChIKey was computed, require exact InChIKey match
    if rec.inchikey is not None and rec.inchikey != gold["inchikey"]:
        return False
    # In baseline mode without RDKit, if raw_smiles was explicitly present in text, check if it matches canonical
    if rec.inchikey is None and rec.raw_smiles:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(rec.raw_smiles)
        if mol is None or Chem.MolToInchiKey(mol) != gold["inchikey"]:
            return False

    if rec.oxidant != gold["oxidant"]:
        return False
    if (rec.inequality or None) != (gold.get("inequality") or None):
        return False

    if rec.standardized_k <= 0 or gold["standardized_k"] <= 0:
        return False
    if abs(math.log10(rec.standardized_k) - math.log10(gold["standardized_k"])) > 0.05:
        return False

    if gold["ph"] is not None:
        if rec.ph is None or abs(rec.ph - gold["ph"]) > 0.20:
            return False

    if abs(rec.temperature_c - gold["temperature_c"]) > 1.0:
        return False

    return True


def record_matches_trap(rec: ReactionKineticsRecord, trap: Dict[str, Any]) -> bool:
    """Check if an accepted record corresponds to an injected hallucination/violation trap."""
    return (
        canonical_name(rec.compound_name).lower() == canonical_name(trap["compound_name"]).lower()
        and rec.oxidant == trap["oxidant"]
    )


def evaluate_pipeline(
    mode: PipelineMode,
    label: str,
    documents: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    engine = CrossVerificationEngine(mode=mode)

    total_gold = 0
    tp = 0
    fp = 0
    fn = 0
    total_accepted = 0

    total_traps = 0
    traps_rejected = 0
    struct_traps_total = 0
    struct_traps_rejected = 0
    physics_traps_total = 0
    physics_traps_rejected = 0

    unit_exp_cases_total = 0
    unit_exp_cases_caught = 0

    spec_swap_cases_total = 0
    spec_swap_cases_caught = 0

    provenance_span_exact = 0
    provenance_complete_with_inchikey = 0

    audit_trace_points: List[Dict[str, Any]] = []

    unit_exp_gold_cats = {
        "UNIT_TRAP_REPAIRABLE",
        "OCR_GLYPH_BASE10",
        "OCR_EXPONENT_REPAIRABLE",
    }
    unit_exp_trap_cats = {
        "PSEUDO_FIRST_ORDER_TRAP",
        "UNRECOVERABLE_LOG_K_MISMATCH",
    }

    for doc in documents:
        doc_id = doc["doc_id"]
        text = doc["text"]
        gold_list = doc["gold_records"]
        trap_list = doc["injected_traps"]

        out = engine.run(text=text, doc_id=doc_id, page_num=1)
        accepted = out.accepted_records
        total_accepted += len(accepted)
        total_gold += len(gold_list)

        matched_gold_indices = set()
        for rec in accepted:
            # Provenance checks on accepted records
            span_ok = rec.evidence.verify_against_source(text)
            prose_ok = (
                True
                if rec.prose_evidence is None
                else rec.prose_evidence.verify_against_source(text)
            )
            if span_ok and prose_ok:
                provenance_span_exact += 1
                if rec.inchikey is not None and len(rec.inchikey) == 27:
                    provenance_complete_with_inchikey += 1

            hit_idx: Optional[int] = None
            for g_idx, g in enumerate(gold_list):
                if g_idx in matched_gold_indices:
                    continue
                if record_matches_gold(rec, g):
                    hit_idx = g_idx
                    break
            if hit_idx is not None:
                tp += 1
                matched_gold_indices.add(hit_idx)
            else:
                fp += 1

        fn += len(gold_list) - len(matched_gold_indices)

        # Evaluate gold failure-category recovery rates
        for g_idx, g in enumerate(gold_list):
            cat = g.get("failure_category", "CLEAN_GOLD")
            was_recovered = g_idx in matched_gold_indices
            if cat in unit_exp_gold_cats:
                unit_exp_cases_total += 1
                if was_recovered:
                    unit_exp_cases_caught += 1
            if cat == "PH_PKA_SWAP_REPAIRABLE":
                spec_swap_cases_total += 1
                if was_recovered:
                    spec_swap_cases_caught += 1

        # Evaluate injected trap rejection rates
        for trap in trap_list:
            total_traps += 1
            ttype = trap["trap_type"]
            leaked = any(record_matches_trap(r, trap) for r in accepted)
            rejected_ok = not leaked
            if rejected_ok:
                traps_rejected += 1

            if ttype == "INVALID_VALENCE_HALLUCINATION":
                struct_traps_total += 1
                if rejected_ok:
                    struct_traps_rejected += 1
            else:
                physics_traps_total += 1
                if rejected_ok:
                    physics_traps_rejected += 1

            if ttype in unit_exp_trap_cats:
                unit_exp_cases_total += 1
                if rejected_ok:
                    unit_exp_cases_caught += 1
            if ttype == "SPECIATION_VIOLATION":
                spec_swap_cases_total += 1
                if rejected_ok:
                    spec_swap_cases_caught += 1

        # Collect candidate-level parity & anomaly points from full mode for Figure 2
        if mode == "full":
            # Also run baseline parser or inspect pre-repair vs post-repair values
            base_out = CrossVerificationEngine(mode="baseline").run(text=text, doc_id=doc_id, page_num=1)
            base_map = {
                (canonical_name(r.compound_name).lower(), r.oxidant): r
                for r in base_out.all_candidates
            }
            for cand in out.all_candidates:
                key = (canonical_name(cand.compound_name).lower(), cand.oxidant)
                b_rec = base_map.get(key)
                raw_naive_k = b_rec.standardized_k if b_rec else cand.k_value
                audit_trace_points.append({
                    "doc_id": doc_id,
                    "record_id": cand.record_id,
                    "compound_name": cand.compound_name,
                    "compound_class": cand.compound_class,
                    "oxidant": cand.oxidant,
                    "raw_k_value": cand.k_value,
                    "naive_k": raw_naive_k,
                    "standardized_k": cand.standardized_k,
                    "reported_log10_k": cand.reported_log10_k,
                    "ph": cand.ph,
                    "pka": cand.pka,
                    "temperature_c": cand.temperature_c,
                    "status": cand.audit_report.status if cand.audit_report else "UNKNOWN",
                    "error_flags": cand.audit_report.error_flags if cand.audit_report else [],
                    "repairs_applied": cand.audit_report.repairs_applied if cand.audit_report else [],
                    "calibrated_confidence": (
                        cand.audit_report.calibrated_confidence if cand.audit_report else 0.0
                    ),
                })

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, total_gold)
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    fp_contamination = fp / max(1, total_accepted)
    hallucination_rejection_rate = traps_rejected / max(1, total_traps)
    struct_rejection_rate = struct_traps_rejected / max(1, struct_traps_total)
    physics_rejection_rate = physics_traps_rejected / max(1, physics_traps_total)
    unit_exp_catch_rate = unit_exp_cases_caught / max(1, unit_exp_cases_total)
    spec_catch_rate = spec_swap_cases_caught / max(1, spec_swap_cases_total)
    prov_span_rate = provenance_span_exact / max(1, total_accepted)
    prov_complete_rate = provenance_complete_with_inchikey / max(1, total_accepted)

    metrics = {
        "pipeline_mode": mode,
        "pipeline_name": label,
        "total_gold_records": total_gold,
        "total_injected_traps": total_traps,
        "accepted_count": total_accepted,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_contamination_rate": round(fp_contamination, 4),
        "hallucination_rejection_rate": round(hallucination_rejection_rate, 4),
        "structural_hallucination_rejection_rate": round(struct_rejection_rate, 4),
        "physics_violation_rejection_rate": round(physics_rejection_rate, 4),
        "unit_and_exponent_error_catch_rate": round(unit_exp_catch_rate, 4),
        "speciation_and_ph_swap_catch_rate": round(spec_catch_rate, 4),
        "provenance_span_exact_match_rate": round(prov_span_rate, 4),
        "provenance_exact_match_rate": round(prov_complete_rate, 4),
    }
    return metrics, audit_trace_points


def build_error_breakdown_rows(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Tabulate all failure & repair modes across the 6 chemical families and 3 pipelines."""
    engine_base = CrossVerificationEngine(mode="baseline")
    engine_rdkit = CrossVerificationEngine(mode="rdkit_gate")
    engine_full = CrossVerificationEngine(mode="full")

    counts: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: {"total_occurrences": 0, "caught_baseline": 0, "caught_rdkit_gate": 0, "caught_full_miner": 0}
    )

    for doc in documents:
        doc_id = doc["doc_id"]
        text = doc["text"]
        out_b = engine_base.run(text=text, doc_id=doc_id)
        out_r = engine_rdkit.run(text=text, doc_id=doc_id)
        out_f = engine_full.run(text=text, doc_id=doc_id)

        for g in doc["gold_records"]:
            cat = g.get("failure_category", "CLEAN_GOLD")
            if cat == "CLEAN_GOLD":
                continue
            fam = g["compound_class"]
            bucket = counts[(fam, cat)]
            bucket["total_occurrences"] += 1
            if any(record_matches_gold(rec, g) for rec in out_b.accepted_records):
                bucket["caught_baseline"] += 1
            if any(record_matches_gold(rec, g) for rec in out_r.accepted_records):
                bucket["caught_rdkit_gate"] += 1
            if any(record_matches_gold(rec, g) for rec in out_f.accepted_records):
                bucket["caught_full_miner"] += 1

        for trap in doc["injected_traps"]:
            cat = trap["trap_type"]
            fam = trap["compound_class"]
            bucket = counts[(fam, cat)]
            bucket["total_occurrences"] += 1
            if not any(record_matches_trap(rec, trap) for rec in out_b.accepted_records):
                bucket["caught_baseline"] += 1
            if not any(record_matches_trap(rec, trap) for rec in out_r.accepted_records):
                bucket["caught_rdkit_gate"] += 1
            if not any(record_matches_trap(rec, trap) for rec in out_f.accepted_records):
                bucket["caught_full_miner"] += 1

    rows: List[Dict[str, Any]] = []
    for (fam, cat), vals in sorted(counts.items()):
        tot = vals["total_occurrences"]
        rows.append({
            "compound_class": fam,
            "failure_mode": cat,
            "total_occurrences": tot,
            "caught_baseline": vals["caught_baseline"],
            "caught_rdkit_gate": vals["caught_rdkit_gate"],
            "caught_full_miner": vals["caught_full_miner"],
            "full_catch_rate": round(vals["caught_full_miner"] / max(1, tot), 4),
        })
    return rows


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    corpus_path = ROOT / "data" / "benchmark_corpus.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    documents = corpus["documents"]

    pipeline_results: List[Dict[str, Any]] = []
    full_trace: List[Dict[str, Any]] = []

    print("=" * 86)
    print(f"Running Benchmark Evaluation on {len(documents)} Excerpts ({corpus['num_gold_records']} Gold, {corpus['num_injected_traps']} Traps)")
    print("=" * 86)

    for mode, label in PIPELINE_SPECS:
        metrics, trace = evaluate_pipeline(mode, label, documents)
        pipeline_results.append(metrics)
        if mode == "full":
            full_trace = trace
        print(
            f"[{mode:10s}] P={metrics['precision']:.3f} | R={metrics['recall']:.3f} | "
            f"F1={metrics['f1']:.3f} | FP_Contam={metrics['false_positive_contamination_rate']:.3f} | "
            f"TrapReject={metrics['hallucination_rejection_rate']:.3f} | "
            f"UnitExpCatch={metrics['unit_and_exponent_error_catch_rate']:.3f} | "
            f"ProvExact={metrics['provenance_exact_match_rate']:.3f}"
        )

    breakdown_rows = build_error_breakdown_rows(documents)

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "benchmark_name": corpus["benchmark_name"],
        "num_documents": corpus["num_documents"],
        "num_gold_records": corpus["num_gold_records"],
        "num_injected_traps": corpus["num_injected_traps"],
        "pipelines": pipeline_results,
        "audit_trace": full_trace,
    }
    json_out = results_dir / "miner_benchmark.json"
    json_out.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_out = results_dir / "error_breakdown.csv"
    fieldnames = [
        "compound_class",
        "failure_mode",
        "total_occurrences",
        "caught_baseline",
        "caught_rdkit_gate",
        "caught_full_miner",
        "full_catch_rate",
    ]
    with csv_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(breakdown_rows)

    print("-" * 86)
    print(f"Saved JSON metrics -> {json_out}")
    print(f"Saved CSV breakdown -> {csv_out} ({len(breakdown_rows)} rows)")


if __name__ == "__main__":
    main()
