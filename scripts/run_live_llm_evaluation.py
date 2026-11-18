"""Evaluate Real Live-LLM Literature Extraction x Domain-Physics Verification on 48 Excerpts.

Runs genuine `/v1/messages` LLM extraction (`qwen3.8-flash` and `deepseek-flash`) across
all 48 environmental oxidation literature excerpts (`data/benchmark_corpus.json`), caches
verbatim raw LLM responses in `results/live_llm_extraction_traces.jsonl` (zero credentials
stored; 100% offline replayable without API key), and compares:
  1. Raw Single-Pass LLM Extractor (unfiltered LLM JSON output)
  2. LLM Extractor + RDKit Cheminformatics Gate (valence/ring/InChIKey sanitization)
  3. ChemLit-Kinetics-Miner Full (LLM x Deterministic Prose/Footnote Cross-Audit + RDKit +
     Unit/Log10 Parity + Smoluchowski Bound + Henderson-Hasselbalch Speciation Repair)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemlit_miner.chem_validator import lookup_reference_compound
from chemlit_miner.cross_verifier import CrossVerificationEngine
from chemlit_miner.parser import ScientificKineticsParser, normalize_oxidant
from chemlit_miner.schema import EvidenceSpan, ReactionKineticsRecord
from scripts.run_evaluation import canonical_name, record_matches_gold, record_matches_trap


EXTRACTION_SYSTEM_PROMPT = (
    "You are an expert environmental chemistry data extraction engine. "
    "Extract all reported bimolecular aqueous oxidation reaction rate constants from the excerpt. "
    "Output ONLY raw JSON matching this schema:\n"
    '{"records": [{"compound_name": str, "raw_smiles": str | null, "oxidant": "O3" | "•OH" | "SO4•-" | "ClO•", '
    '"inequality": "<" | "<=" | ">" | ">=" | null, "k_value": float, "k_units": str, '
    '"standardized_k_M_inv_s_inv": float, "reported_log10_k": float | null, '
    '"ph": float | null, "pka": float | null, "k_ha": float | null, "k_a_minus": float | null, '
    '"temperature_c": float, "verbatim_quote": str}]}'
)


def _extract_json_obj(raw: str) -> Dict[str, Any]:
    txt = raw.strip()
    fenced = re.findall(r"```(?:json)?\s*\n?(.*?)```", txt, flags=re.DOTALL | re.IGNORECASE)
    for cand in fenced[::-1] + [txt]:
        cand = cand.strip()
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        idx_l = cand.find("{")
        idx_r = cand.rfind("}")
        if 0 <= idx_l < idx_r:
            try:
                obj = json.loads(cand[idx_l : idx_r + 1])
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    return {}


class LiveMessagesClient:
    """Zero-dependency `/v1/messages` client with thread-safe JSONL trace caching."""

    def __init__(self, api_key: str, base_url: str, trace_path: Path) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.trace_path = trace_path
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, Dict[str, Any]] = {}
        if self.trace_path.exists():
            for ln in self.trace_path.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    rec = json.loads(ln)
                    self._cache[rec["call_key"]] = rec

    @staticmethod
    def _key(model: str, doc_id: str, text: str) -> str:
        h = hashlib.sha256(f"{model}|{doc_id}|{text}".encode("utf-8")).hexdigest()[:24]
        return h

    def extract_doc(self, model: str, doc_id: str, text: str) -> Dict[str, Any]:
        ck = self._key(model, doc_id, text)
        with self._lock:
            hit = self._cache.get(ck)
        if hit is not None and hit.get("raw_response", "").strip() and int(hit.get("output_tokens", 0)) < 3950:
            return hit

        if not self.api_key:
            raise RuntimeError(
                f"Cache miss for ({model}, {doc_id}). Set CHEMLIT_API_KEY / CROSSVERIFY_API_KEY for live extraction."
            )

        url = (
            self.base_url + "/messages"
            if self.base_url.endswith("/v1")
            else self.base_url + "/v1/messages"
        )
        payload = {
            "model": model,
            "max_tokens": 4096,
            "temperature": 0.0,
            "thinking": {"type": "disabled"},
            "system": EXTRACTION_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Extract all reaction kinetics records from:\n\n{text}"}],
        }
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "Authorization": f"Bearer {self.api_key}",
            "anthropic-version": "2023-06-01",
            "User-Agent": "ChemLitKineticsMiner/0.2",
        }
        last_err: Optional[Exception] = None
        for attempt in range(6):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
                )
                with urllib.request.urlopen(req, timeout=75.0) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                lat_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                blocks = body.get("content", [])
                out_txt = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()
                usage = body.get("usage", {})
                in_tok = int(usage.get("input_tokens", 0))
                out_tok = int(usage.get("output_tokens", 0))
                rec = {
                    "call_key": ck,
                    "doc_id": doc_id,
                    "model": model,
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "total_tokens": in_tok + out_tok,
                    "latency_ms": lat_ms,
                    "raw_response": out_txt,
                }
                with self._lock:
                    self._cache[ck] = rec
                    with self.trace_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return rec
            except Exception as exc:
                last_err = exc
                time.sleep(2.2 * (attempt + 1))
        raise RuntimeError(f"Live LLM call failed for ({model}, {doc_id}): {last_err}")


def _bind_evidence_span(doc_id: str, source_text: str, quote_hint: str) -> EvidenceSpan:
    q = (quote_hint or "").strip()
    if q and q in source_text:
        idx = source_text.index(q)
        return EvidenceSpan(
            doc_id=doc_id,
            page_num=1,
            locator="LLM Verbatim Span",
            char_start=idx,
            char_end=idx + len(q),
            verbatim_quote=q,
        )
    # Empty offsets cannot verify against any source. Keep the proposed quote for audit.
    return EvidenceSpan(
        doc_id=doc_id,
        page_num=1,
        locator="Unverified Span",
        char_start=0,
        char_end=0,
        verbatim_quote=q or "[missing quote]",
    )


def _llm_json_to_candidates(doc_id: str, source_text: str, raw_resp: str) -> List[ReactionKineticsRecord]:
    parsed = _extract_json_obj(raw_resp)
    raw_list = parsed.get("records", [])
    if not isinstance(raw_list, list):
        return []
    out: List[ReactionKineticsRecord] = []
    for idx, item in enumerate(raw_list, start=1):
        if not isinstance(item, dict):
            continue
        cname = str(item.get("compound_name") or "").strip()
        if not cname:
            continue
        ox = normalize_oxidant(str(item.get("oxidant") or "")) or "O3"
        try:
            k_val = float(item.get("k_value", 0.0) or 0.0)
        except Exception:
            continue
        k_units = str(item.get("k_units") or "M^-1 s^-1").strip()
        try:
            std_k = float(item.get("standardized_k_M_inv_s_inv") or k_val)
        except Exception:
            std_k = k_val
        ineq = item.get("inequality")
        if ineq not in ("<", "<=", ">", ">=", "~"):
            ineq = None

        def _opt_float(key: str) -> Optional[float]:
            v = item.get(key)
            if v is None or v == "":
                return None
            try:
                return float(v)
            except Exception:
                return None

        temp_c = _opt_float("temperature_c")
        if temp_c is None:
            temp_c = 20.0
        ref = lookup_reference_compound(cname)
        c_class = ref.compound_class if ref else "Phenols"
        raw_smi = item.get("raw_smiles")
        raw_smi_str = str(raw_smi).strip() if raw_smi not in (None, "", "null", "None") else None
        ev = _bind_evidence_span(doc_id, source_text, str(item.get("verbatim_quote") or ""))
        out.append(
            ReactionKineticsRecord(
                record_id=f"{doc_id}_LLM_{idx:02d}",
                doc_id=doc_id,
                compound_name=cname,
                compound_class=c_class,
                raw_smiles=raw_smi_str,
                oxidant=ox,
                raw_k_string=f"{k_val} {k_units}",
                inequality=ineq,
                k_value=k_val,
                k_units=k_units,
                standardized_k=std_k,
                reported_log10_k=_opt_float("reported_log10_k"),
                ph=_opt_float("ph"),
                pka=_opt_float("pka"),
                k_ha=_opt_float("k_ha"),
                k_a_minus=_opt_float("k_a_minus"),
                temperature_c=temp_c,
                temperature_k=round(temp_c + 273.15, 4),
                evidence=ev,
            )
        )
    return out


def evaluate_llm_model(
    model_name: str,
    documents: List[Dict[str, Any]],
    traces: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluate 3 progressive verification stages on genuine live LLM outputs."""
    parser = ScientificKineticsParser(enable_ocr_repair=True, parse_footnotes=True, extract_prose_standalone=True)
    full_engine = CrossVerificationEngine(mode="full")

    stage_results: Dict[str, Any] = {}
    for stage_key, stage_label in [
        ("raw_llm", f"1. Raw Single-Pass LLM ({model_name})"),
        ("llm_rdkit", f"2. {model_name} + RDKit Structure Gate"),
        ("llm_full_miner", f"3. {model_name} + ChemLit-Kinetics-Miner (Full Physics & Prose Audit)"),
    ]:
        tp = fp = fn = total_gold = total_accepted = 0
        total_traps = traps_rejected = 0

        for doc in documents:
            doc_id = doc["doc_id"]
            text = doc["text"]
            gold_list = doc["gold_records"]
            trap_list = doc["injected_traps"]
            total_gold += len(gold_list)

            raw_resp = traces[doc_id]["raw_response"]
            llm_cands = _llm_json_to_candidates(doc_id, text, raw_resp)

            if stage_key == "raw_llm":
                accepted = llm_cands
            elif stage_key == "llm_rdkit":
                accepted = full_engine._run_rdkit_gate(text, doc_id, llm_cands).accepted_records
            else:
                # Cross-verify LLM candidates against deterministic parser footnotes/prose & full physics auditor!
                det_cands, prose_mentions = parser.parse_document(text=text, doc_id=doc_id, page_num=1)
                det_by_key = {
                    (canonical_name(r.compound_name).lower(), r.oxidant): r for r in det_cands
                }
                merged_cands: List[ReactionKineticsRecord] = []
                seen_keys = set()
                for lc in llm_cands:
                    k_pair = (canonical_name(lc.compound_name).lower(), lc.oxidant)
                    first_for_key = k_pair not in seen_keys
                    seen_keys.add(k_pair)
                    # Reconcile footnote units / exact provenance span if deterministic parser found exact table row
                    if k_pair in det_by_key:
                        dr = det_by_key[k_pair]
                        if not lc.evidence.verify_against_source(text):
                            # Recover a source-derived record, without assigning its quote to LLM values.
                            if first_for_key:
                                merged_cands.append(dr)
                            continue
                        # Prefer footnote-resolved k_units and exact character-offset span from parser
                        if dr.k_units != "M^-1 s^-1" and lc.k_units == "M^-1 s^-1":
                            lc.k_units = dr.k_units
                        if lc.k_ha is None:
                            lc.k_ha = dr.k_ha
                        if lc.k_a_minus is None:
                            lc.k_a_minus = dr.k_a_minus
                        if lc.reported_log10_k is None:
                            lc.reported_log10_k = dr.reported_log10_k
                    merged_cands.append(lc)

                # Recover any prose/table record missed by single-pass LLM
                for k_pair, dr in det_by_key.items():
                    if k_pair not in seen_keys:
                        merged_cands.append(dr)
                        seen_keys.add(k_pair)
                for pm in prose_mentions:
                    if pm.parsed_rate is not None and pm.oxidant is not None:
                        pk = (canonical_name(pm.compound_name).lower(), pm.oxidant)
                        if pk not in seen_keys:
                            defaults = {
                                "temperature_c": merged_cands[0].temperature_c if merged_cands else 20.0,
                                "water_matrix": "phosphate_buffer",
                            }
                            merged_cands.append(
                                parser._prose_mention_to_record(pm, doc_id=doc_id, rec_idx=len(merged_cands) + 1, defaults=defaults)
                            )
                            seen_keys.add(pk)

                accepted = full_engine._run_full_dual_audit(text, doc_id, merged_cands, prose_mentions).accepted_records

            total_accepted += len(accepted)
            matched_g = set()
            for rec in accepted:
                hit_i = None
                for gi, g in enumerate(gold_list):
                    if gi not in matched_g and record_matches_gold(rec, g):
                        hit_i = gi
                        break
                if hit_i is not None:
                    tp += 1
                    matched_g.add(hit_i)
                else:
                    fp += 1
            fn += len(gold_list) - len(matched_g)

            for trap in trap_list:
                total_traps += 1
                if not any(record_matches_trap(r, trap) for r in accepted):
                    traps_rejected += 1

        prec = tp / max(1, tp + fp)
        rec_rate = tp / max(1, total_gold)
        f1 = (2 * prec * rec_rate / (prec + rec_rate)) if (prec + rec_rate) > 0 else 0.0
        stage_results[stage_key] = {
            "label": stage_label,
            "precision": round(prec, 4),
            "recall": round(rec_rate, 4),
            "f1": round(f1, 4),
            "fp_contamination_rate": round(fp / max(1, total_accepted), 4),
            "trap_absence_rate": round(traps_rejected / max(1, total_traps), 4),
            "traps_absent_from_accepted": traps_rejected,
            "total_traps": total_traps,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    mean_tok = sum(t["total_tokens"] for t in traces.values()) / max(1, len(traces))
    parseable = sum(isinstance(_extract_json_obj(t["raw_response"]).get("records"), list) for t in traces.values())
    return {"model": model_name, "responses_with_records_array": parseable,
            "mean_tokens_per_excerpt": round(mean_tok, 1), "stages": stage_results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-only", action="store_true",
                        help="Score only excerpts with saved calls from both models; never contact a provider.")
    args = parser.parse_args()
    api_key = os.environ.get("CHEMLIT_API_KEY") or os.environ.get("CROSSVERIFY_API_KEY") or ""
    base_url = os.environ.get("CHEMLIT_BASE_URL") or os.environ.get("CROSSVERIFY_BASE_URL") or "https://api.anthropic.com"

    corpus_path = ROOT / "data" / "benchmark_corpus.json"
    documents = json.loads(corpus_path.read_text(encoding="utf-8"))["documents"]
    trace_path = ROOT / "results" / "live_llm_extraction_traces.jsonl"
    client = LiveMessagesClient(api_key=api_key, base_url=base_url, trace_path=trace_path)

    models = ["deepseek-flash", "qwen3.8-flash"]
    all_model_traces: Dict[str, Dict[str, Dict[str, Any]]] = {m: {} for m in models}

    if args.cached_only:
        for m in models:
            for doc in documents:
                key = client._key(m, doc["doc_id"], doc["text"])
                if key in client._cache:
                    all_model_traces[m][doc["doc_id"]] = client._cache[key]
    else:
        jobs = [(m, doc["doc_id"], doc["text"]) for m in models for doc in documents]
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(client.extract_doc, m, did, txt): (m, did) for m, did, txt in jobs}
            for future in concurrent.futures.as_completed(futures):
                m, did = futures[future]
                all_model_traces[m][did] = future.result()

    paired_docs = [d for d in documents if all(d["doc_id"] in all_model_traces[m] for m in models)]
    if not paired_docs:
        parser.error("No matching excerpt traces for both models.")
    eval_summaries = [
        evaluate_llm_model(m, paired_docs, {d["doc_id"]: all_model_traces[m][d["doc_id"]] for d in paired_docs})
        for m in models
    ]
    out_json = {
        "benchmark": "ChemLit-Kinetics-Miner Real Live-LLM Extraction & Physics Verification Benchmark",
        "protocol": "Anthropic /v1/messages",
        "num_excerpts": len(documents),
        "num_paired_excerpts": len(paired_docs),
        "paired_doc_ids": [d["doc_id"] for d in paired_docs],
        "missing_doc_ids": {m: [d["doc_id"] for d in documents if d["doc_id"] not in all_model_traces[m]] for m in models},
        "total_cached_llm_calls": len(client._cache),
        "models": eval_summaries,
    }
    out_path = ROOT / "results" / "live_llm_miner_benchmark.json"
    out_path.write_text(json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8")

    for es in eval_summaries:
        print(f"\n=== Model: {es['model']} (mean {es['mean_tokens_per_excerpt']} tok/excerpt) ===")
        for sk, sv in es["stages"].items():
            print(
                f"  {sv['label']:65s} | P={sv['precision']*100:5.1f}% | R={sv['recall']*100:5.1f}% | "
                f"F1={sv['f1']*100:5.1f}% | FP={sv['fp_contamination_rate']*100:5.1f}% | "
                f"Traps absent={sv['traps_absent_from_accepted']}/{sv['total_traps']} ({sv['trap_absence_rate']*100:5.1f}%)"
            )


if __name__ == "__main__":
    main()
