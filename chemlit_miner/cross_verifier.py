"""Dual-Agent Cross-Verification Engine (Extractor x Domain-Physics Auditor).

Implements the OmniHarness-inspired Generator-Auditor architecture for scientific
literature mining:
  - Agent 1 (ExtractorAgent): Parses Markdown/LaTeX tables, footnotes, and prose
    (supports deterministic scientific parser + optional OpenAI-compatible LLM backend).
  - Agent 2 (DomainPhysicsAuditor): Executes RDKit structure sanitization, unit & log10(k)
    parity checks, Smoluchowski diffusion bounds, Henderson-Hasselbalch speciation checks,
    and table-vs-prose conflict reconciliation with exact character-offset provenance.
"""

from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from .chem_validator import (
    ChemValidator,
    lookup_reference_compound,
    normalize_rate_units,
    validate_chemical_structure,
)
from .parser import ProseMention, ScientificKineticsParser
from .schema import (
    ChemAuditReport,
    CrossCheckAuditLog,
    EvidenceSpan,
    ReactionKineticsRecord,
)

PipelineMode = Literal["baseline", "rdkit_gate", "full"]


@dataclass
class LLMBackendConfig:
    """Configuration for optional pluggable OpenAI-compatible LLM extraction backend."""

    base_url: str = "https://api.openai.com/v1"
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    timeout_s: float = 30.0
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> Optional["LLMBackendConfig"]:
        key = os.environ.get("CHEMLIT_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        return cls(
            base_url=os.environ.get("CHEMLIT_LLM_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("CHEMLIT_LLM_MODEL", "gpt-4o-mini"),
            api_key=key,
        )


@dataclass
class ExtractionRunOutput:
    """Complete output of running an extraction + verification pipeline on an excerpt."""

    doc_id: str
    pipeline_mode: PipelineMode
    accepted_records: List[ReactionKineticsRecord]
    rejected_records: List[ReactionKineticsRecord]
    all_candidates: List[ReactionKineticsRecord]
    audit_reports: List[ChemAuditReport]


class OpenAICompatibleExtractorClient:
    """Zero-dependency OpenAI-compatible JSON chat client for optional LLM-assisted extraction."""

    def __init__(self, config: LLMBackendConfig) -> None:
        self.config = config

    def query_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        if not self.config.api_key:
            raise RuntimeError("LLMBackendConfig.api_key is required for LLM extraction mode.")
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            return json.loads(content)
        except (urllib.error.URLError, KeyError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"OpenAI-compatible LLM request failed: {exc}") from exc


class CrossVerificationEngine:
    """Dual-Agent Extractor x Domain-Physics Auditor pipeline."""

    def __init__(
        self,
        mode: PipelineMode = "full",
        validator: Optional[ChemValidator] = None,
        llm_config: Optional[LLMBackendConfig] = None,
    ) -> None:
        self.mode: PipelineMode = mode
        self.validator = validator or ChemValidator()
        self.llm_client = OpenAICompatibleExtractorClient(llm_config) if llm_config else None

        if mode == "baseline":
            # Naive single-pass extractor: no OCR glyph repair, no footnote unit lookup, table-first only
            self.parser = ScientificKineticsParser(
                enable_ocr_repair=False,
                parse_footnotes=False,
                extract_prose_standalone=True,
            )
        elif mode == "rdkit_gate":
            # OCR-aware parser + RDKit structure sanitization gate, without prose/physics cross-audit
            self.parser = ScientificKineticsParser(
                enable_ocr_repair=True,
                parse_footnotes=False,
                extract_prose_standalone=True,
            )
        else:
            # Full parser with OCR repair, footnote unit resolution, and prose cross-verification
            self.parser = ScientificKineticsParser(
                enable_ocr_repair=True,
                parse_footnotes=True,
                extract_prose_standalone=True,
            )

    def run(
        self,
        text: str,
        doc_id: str = "DOC_UNKNOWN",
        page_num: int = 1,
    ) -> ExtractionRunOutput:
        """Execute configured pipeline mode on a scientific document excerpt."""
        candidates, prose_mentions = self.parser.parse_document(
            text=text, doc_id=doc_id, page_num=page_num
        )

        # In full mode, also promote prose-only compounds in hybrid table+prose excerpts!
        if self.mode == "full" and candidates and prose_mentions:
            existing_keys = {
                (self._canonical_comp_key(r.compound_name), r.oxidant) for r in candidates
            }
            next_idx = len(candidates) + 1
            for pm in prose_mentions:
                if pm.parsed_rate is None or pm.oxidant is None:
                    continue
                ckey = (self._canonical_comp_key(pm.compound_name), pm.oxidant)
                if ckey not in existing_keys:
                    defaults = {
                        "temperature_c": candidates[0].temperature_c,
                        "water_matrix": candidates[0].water_matrix,
                    }
                    extra_rec = self.parser._prose_mention_to_record(
                        pm, doc_id=doc_id, rec_idx=next_idx, defaults=defaults
                    )
                    candidates.append(extra_rec)
                    existing_keys.add(ckey)
                    next_idx += 1

        if self.mode == "baseline":
            return self._run_baseline(text, doc_id, candidates)
        if self.mode == "rdkit_gate":
            return self._run_rdkit_gate(text, doc_id, candidates)
        return self._run_full_dual_audit(text, doc_id, candidates, prose_mentions)

    def _canonical_comp_key(self, name: str) -> str:
        ref = lookup_reference_compound(name)
        return ref.name.lower() if ref else name.strip().lower()

    def _run_baseline(
        self,
        text: str,
        doc_id: str,
        candidates: List[ReactionKineticsRecord],
    ) -> ExtractionRunOutput:
        accepted: List[ReactionKineticsRecord] = []
        reports: List[ChemAuditReport] = []
        for rec in candidates:
            # Baseline single-pass does not run RDKit sanitization or physics audit
            rec.canonical_smiles = rec.raw_smiles
            rep = ChemAuditReport(
                record_id=rec.record_id,
                status="VERIFIED",
                extractor_confidence=0.70,
                calibrated_confidence=0.70,
                prose_corroborated=False,
                structure_verified=False,
                unit_and_log_verified=False,
                physics_bounds_verified=False,
                speciation_verified=False,
                diagnostic_messages=["Baseline single-pass extraction (no cheminformatics/physics audit)."],
            )
            rec.audit_report = rep
            rec.audit_log = CrossCheckAuditLog(
                record_id=rec.record_id,
                extractor_confidence=0.70,
                auditor_verdict="VERIFIED",
                calibrated_confidence=0.70,
            )
            accepted.append(rec)
            reports.append(rep)

        return ExtractionRunOutput(
            doc_id=doc_id,
            pipeline_mode="baseline",
            accepted_records=accepted,
            rejected_records=[],
            all_candidates=candidates,
            audit_reports=reports,
        )

    def _run_rdkit_gate(
        self,
        text: str,
        doc_id: str,
        candidates: List[ReactionKineticsRecord],
    ) -> ExtractionRunOutput:
        accepted: List[ReactionKineticsRecord] = []
        rejected: List[ReactionKineticsRecord] = []
        reports: List[ChemAuditReport] = []

        for rec in candidates:
            chem_res, _ = validate_chemical_structure(rec.compound_name, raw_smiles=rec.raw_smiles)
            rec.validation = chem_res
            if not chem_res.is_valid_structure:
                rep = ChemAuditReport(
                    record_id=rec.record_id,
                    status="FLAGGED_ERROR",
                    extractor_confidence=0.75,
                    calibrated_confidence=0.05,
                    structure_verified=False,
                    error_flags=chem_res.error_codes,
                    diagnostic_messages=chem_res.validation_warnings,
                )
                rec.audit_report = rep
                rec.audit_log = CrossCheckAuditLog(
                    record_id=rec.record_id,
                    extractor_confidence=0.75,
                    auditor_verdict="REJECTED_HALLUCINATION",
                    rejection_reasons=chem_res.validation_warnings,
                    calibrated_confidence=0.05,
                )
                rejected.append(rec)
                reports.append(rep)
            else:
                rec.canonical_smiles = chem_res.canonical_smiles
                rec.inchikey = chem_res.inchikey
                rec.molecular_formula = chem_res.molecular_formula
                rec.exact_mw = chem_res.exact_mw
                rec.formal_charge = chem_res.formal_charge
                rep = ChemAuditReport(
                    record_id=rec.record_id,
                    status="VERIFIED",
                    extractor_confidence=0.78,
                    calibrated_confidence=0.82,
                    structure_verified=True,
                    diagnostic_messages=["Passed RDKit valence and canonicalization gate."],
                )
                rec.audit_report = rep
                rec.audit_log = CrossCheckAuditLog(
                    record_id=rec.record_id,
                    extractor_confidence=0.78,
                    auditor_verdict="VERIFIED",
                    calibrated_confidence=0.82,
                )
                accepted.append(rec)
                reports.append(rep)

        return ExtractionRunOutput(
            doc_id=doc_id,
            pipeline_mode="rdkit_gate",
            accepted_records=accepted,
            rejected_records=rejected,
            all_candidates=candidates,
            audit_reports=reports,
        )

    def _find_matching_prose(
        self, rec: ReactionKineticsRecord, prose_mentions: List[ProseMention]
    ) -> Optional[ProseMention]:
        target_key = self._canonical_comp_key(rec.compound_name)
        best: Optional[ProseMention] = None
        for pm in prose_mentions:
            if self._canonical_comp_key(pm.compound_name) != target_key:
                continue
            if pm.oxidant is not None and pm.oxidant != rec.oxidant:
                continue
            best = pm
            if pm.parsed_rate is not None:
                return pm
        return best

    def _resolve_species_form(self, rec: ReactionKineticsRecord) -> str:
        if rec.ph is None and rec.pka is not None:
            return "intrinsic_species"
        if rec.ph is not None and rec.pka is not None:
            if rec.compound_class in ("Amines", "Beta-blockers"):
                if rec.ph <= rec.pka - 2.0:
                    return "cationic_BH+"
                if rec.ph >= rec.pka + 2.0:
                    return "neutral_B"
            else:
                if rec.ph <= rec.pka - 2.0:
                    return "neutral_HA"
                if rec.ph >= rec.pka + 2.0:
                    return "anionic_A-"
        return "apparent_k_app"

    def _run_full_dual_audit(
        self,
        text: str,
        doc_id: str,
        candidates: List[ReactionKineticsRecord],
        prose_mentions: List[ProseMention],
    ) -> ExtractionRunOutput:
        accepted: List[ReactionKineticsRecord] = []
        rejected: List[ReactionKineticsRecord] = []
        reports: List[ChemAuditReport] = []

        for rec in candidates:
            repairs: List[str] = list(rec.metadata.get("ocr_repairs", []))
            error_flags: List[str] = []
            diagnostics: List[str] = []

            # Verify exact provenance offset on primary evidence
            prov_ok = rec.evidence.verify_against_source(text)
            if not prov_ok:
                error_flags.append("PROVENANCE_SPAN_MISMATCH")
                diagnostics.append("Primary evidence span failed exact character-offset verification.")

            # Step 1: Match against prose mentions for cross-verification & enrichment
            pm = self._find_matching_prose(rec, prose_mentions)
            prose_corroborated = False
            prose_std_k: Optional[float] = None

            if pm is not None and pm.evidence.verify_against_source(text):
                rec.prose_evidence = pm.evidence
                if rec.k_ha is None and pm.k_ha is not None:
                    rec.k_ha = pm.k_ha
                if rec.k_a_minus is None and pm.k_a_minus is not None:
                    rec.k_a_minus = pm.k_a_minus
                if rec.reported_log10_k is None and pm.reported_log10_k is not None:
                    rec.reported_log10_k = pm.reported_log10_k
                if rec.temperature_defaulted and pm.temperature_c is not None:
                    rec.temperature_c = pm.temperature_c
                    rec.temperature_k = round(pm.temperature_c + 273.15, 4)
                    rec.temperature_defaulted = False
                if rec.radical_scavenger is None and pm.radical_scavenger is not None:
                    rec.radical_scavenger = pm.radical_scavenger

                if pm.parsed_rate is not None:
                    p_units = pm.parsed_rate.units or "M^-1 s^-1"
                    prose_std_k, _, p_u_ok, _ = normalize_rate_units(pm.parsed_rate.k_value, p_units)
                    if not p_u_ok:
                        prose_std_k = None

            # Track unit normalization repairs (e.g., mM^-1 s^-1 or M^-1 min^-1 -> M^-1 s^-1)
            norm_k_init, factor_init, u_ok_init, _ = normalize_rate_units(rec.k_value, rec.k_units)
            if u_ok_init and abs(factor_init - 1.0) > 1e-9:
                repairs.append(
                    f"Normalized rate units '{rec.k_units}' (factor={factor_init:.4g}) -> M^-1 s^-1"
                )

            # Step 2: Run initial ChemValidator pass
            val = self.validator.validate(
                compound_name=rec.compound_name,
                k_value=rec.k_value,
                k_units=rec.k_units,
                raw_smiles=rec.raw_smiles,
                reported_log10_k=rec.reported_log10_k,
                ph=rec.ph,
                pka=rec.pka,
                temperature_c=rec.temperature_c,
                k_ha=rec.k_ha,
                k_a_minus=rec.k_a_minus,
                pka2=rec.pka2,
                inequality=rec.inequality,
            )

            # Step 3: Attempt deterministic, evidence-backed auto-repairs on recoverable anomalies
            if val.is_valid_structure:
                rec.canonical_smiles = val.canonical_smiles
                rec.inchikey = val.inchikey
                rec.molecular_formula = val.molecular_formula
                rec.exact_mw = val.exact_mw
                rec.formal_charge = val.formal_charge

                needs_revalidate = False

                # Repair A: pH <-> pKa column swap when corroborated by prose or speciation + ref pKa
                if val.ph_pka_swap_suspected and rec.ph is not None and rec.pka is not None:
                    ref = lookup_reference_compound(rec.compound_name)
                    prose_confirms_swap = (
                        pm is not None
                        and pm.ph is not None
                        and abs(pm.ph - rec.pka) <= 0.15
                    )
                    spec_confirms_swap = (
                        val.speciation_check is not None
                        and val.speciation_check.swapped_ph_pka_detected
                    )
                    ref_confirms_swap = (
                        ref is not None
                        and ref.pka is not None
                        and abs(rec.ph - ref.pka) <= 0.20
                        and 1.0 <= rec.pka <= 13.0
                    )
                    if prose_confirms_swap or spec_confirms_swap or ref_confirms_swap:
                        old_ph, old_pka = rec.ph, rec.pka
                        rec.ph, rec.pka = old_pka, old_ph
                        repairs.append(
                            f"Unswapped transposed pH/pKa columns: (pH={old_ph}, pKa={old_pka}) -> "
                            f"(pH={rec.ph}, pKa={rec.pka})"
                        )
                        error_flags.append("PH_PKA_COLUMN_SWAP")
                        needs_revalidate = True

                # Repair B: Millimolar unit trap or integer-decade OCR exponent typo when corroborated!
                if not val.is_unit_consistent and val.standardized_k_M_inv_s_inv and val.standardized_k_M_inv_s_inv > 0:
                    curr_k = val.standardized_k_M_inv_s_inv
                    # Case B1: Both prose and reported_log10_k (or prose alone) corroborate exact value
                    if prose_std_k is not None and prose_std_k > 0:
                        log_diff_prose = abs(math.log10(curr_k) - math.log10(prose_std_k))
                        prose_matches_rep_log = (
                            rec.reported_log10_k is None
                            or abs(math.log10(prose_std_k) - rec.reported_log10_k) <= self.validator.log_k_tolerance
                        )
                        if log_diff_prose > self.validator.log_k_tolerance and prose_matches_rep_log:
                            decade_shift = round(math.log10(prose_std_k) - math.log10(curr_k))
                            if abs(log_diff_prose - abs(decade_shift)) <= 0.08 and abs(decade_shift) in (1, 2, 3, 4):
                                old_std = rec.standardized_k
                                if abs(decade_shift - 3) == 0 and pm and pm.parsed_rate and pm.parsed_rate.units and "mm" in pm.parsed_rate.units.lower():
                                    rec.k_units = pm.parsed_rate.units
                                    repairs.append(
                                        f"Repaired omitted mM^-1 s^-1 unit via prose cross-check: "
                                        f"{old_std:.3e} -> {prose_std_k:.3e} M^-1 s^-1"
                                    )
                                    error_flags.append("UNIT_MILLIMOLAR_TRAP_OR_1000X_SHIFT")
                                else:
                                    rec.k_value = prose_std_k
                                    rec.k_units = "M^-1 s^-1"
                                    repairs.append(
                                        f"Repaired {10**abs(decade_shift)}x OCR exponent misread in table "
                                        f"({old_std:.3e} -> {prose_std_k:.3e} M^-1 s^-1) corroborated by prose"
                                    )
                                    error_flags.append("OCR_EXPONENT_MISMATCH")
                                rec.standardized_k = prose_std_k
                                needs_revalidate = True

                # Also check table vs prose conflict when reported_log10_k wasn't in table!
                if (
                    prose_std_k is not None
                    and prose_std_k > 0
                    and val.standardized_k_M_inv_s_inv
                    and val.standardized_k_M_inv_s_inv > 0
                    and not needs_revalidate
                ):
                    curr_k = val.standardized_k_M_inv_s_inv
                    diff_vs_prose = abs(math.log10(curr_k) - math.log10(prose_std_k))
                    if diff_vs_prose <= 0.08:
                        prose_corroborated = True
                    else:
                        # Check if speciation physics uniquely sides with prose over OCR-corrupted table!
                        if rec.ph is not None and rec.pka is not None and rec.k_ha is not None and rec.k_a_minus is not None:
                            alpha_1 = 1.0 / (1.0 + 10.0 ** (rec.pka - rec.ph))
                            pred_spec = (1.0 - alpha_1) * rec.k_ha + alpha_1 * rec.k_a_minus
                            if pred_spec > 0 and abs(math.log10(prose_std_k) - math.log10(pred_spec)) <= 0.12:
                                old_std = rec.standardized_k
                                rec.k_value = prose_std_k
                                rec.k_units = "M^-1 s^-1"
                                rec.standardized_k = prose_std_k
                                repairs.append(
                                    f"Repaired table rate ({old_std:.3e} -> {prose_std_k:.3e} M^-1 s^-1) "
                                    f"via joint prose + Henderson-Hasselbalch speciation verification"
                                )
                                error_flags.append("OCR_EXPONENT_MISMATCH")
                                needs_revalidate = True
                        elif curr_k > self.validator.diffusion_limit and prose_std_k <= self.validator.diffusion_limit:
                            # Table had OCR diffusion-violating exponent, prose has true sub-diffusion value
                            old_std = rec.standardized_k
                            rec.k_value = prose_std_k
                            rec.k_units = "M^-1 s^-1"
                            rec.standardized_k = prose_std_k
                            repairs.append(
                                f"Repaired diffusion-violating OCR table exponent ({old_std:.3e} -> "
                                f"{prose_std_k:.3e} M^-1 s^-1) using prose value"
                            )
                            error_flags.append("OCR_EXPONENT_MISMATCH")
                            needs_revalidate = True
                        else:
                            error_flags.append("UNRESOLVED_TABLE_PROSE_CONFLICT")
                            diagnostics.append(
                                f"Unresolved conflict between table ({curr_k:.3e}) and prose ({prose_std_k:.3e} M^-1 s^-1)."
                            )

                if needs_revalidate:
                    val = self.validator.validate(
                        compound_name=rec.compound_name,
                        k_value=rec.k_value,
                        k_units=rec.k_units,
                        raw_smiles=rec.raw_smiles,
                        reported_log10_k=rec.reported_log10_k,
                        ph=rec.ph,
                        pka=rec.pka,
                        temperature_c=rec.temperature_c,
                        k_ha=rec.k_ha,
                        k_a_minus=rec.k_a_minus,
                        pka2=rec.pka2,
                        inequality=rec.inequality,
                    )
                    if (
                        prose_std_k is not None
                        and val.standardized_k_M_inv_s_inv
                        and abs(math.log10(val.standardized_k_M_inv_s_inv) - math.log10(prose_std_k)) <= 0.08
                    ):
                        prose_corroborated = True

            if val.standardized_k_M_inv_s_inv is not None:
                rec.standardized_k = val.standardized_k_M_inv_s_inv

            rec.validation = val
            rec.species_form = self._resolve_species_form(rec)

            # Determine final verdict: VERIFIED vs AUTO_REPAIRED vs FLAGGED_ERROR
            unresolved_errors = list(val.error_codes)
            if "UNRESOLVED_TABLE_PROSE_CONFLICT" in error_flags:
                unresolved_errors.append("UNRESOLVED_TABLE_PROSE_CONFLICT")
            if "PROVENANCE_SPAN_MISMATCH" in error_flags:
                unresolved_errors.append("PROVENANCE_SPAN_MISMATCH")

            is_rejected = (
                not val.is_valid_structure
                or not val.is_unit_consistent
                or val.exceeds_diffusion_limit
                or val.invalid_temperature
                or not val.speciation_consistent
                or val.ph_pka_swap_suspected
                or len(unresolved_errors) > 0
            )

            if is_rejected:
                all_flags = list(dict.fromkeys([*error_flags, *unresolved_errors]))
                all_msgs = [*diagnostics, *val.validation_warnings]
                verdict: Literal["REJECTED_HALLUCINATION", "REJECTED_PHYSICS_VIOLATION"] = (
                    "REJECTED_HALLUCINATION" if not val.is_valid_structure else "REJECTED_PHYSICS_VIOLATION"
                )
                report = ChemAuditReport(
                    record_id=rec.record_id,
                    status="FLAGGED_ERROR",
                    extractor_confidence=0.72,
                    calibrated_confidence=0.06,
                    prose_corroborated=prose_corroborated,
                    structure_verified=val.is_valid_structure,
                    unit_and_log_verified=val.is_unit_consistent,
                    physics_bounds_verified=not (val.exceeds_diffusion_limit or val.invalid_temperature),
                    speciation_verified=val.speciation_consistent,
                    repairs_applied=repairs,
                    error_flags=all_flags,
                    diagnostic_messages=all_msgs,
                )
                rec.audit_report = report
                rec.audit_log = CrossCheckAuditLog(
                    record_id=rec.record_id,
                    extractor_confidence=0.72,
                    auditor_verdict=verdict,
                    prose_cross_checked=pm is not None,
                    speciation_resolved=val.speciation_check is not None and val.speciation_consistent,
                    resolved_species_form=rec.species_form,
                    corrections_applied=repairs,
                    rejection_reasons=all_msgs,
                    calibrated_confidence=0.06,
                )
                rejected.append(rec)
                reports.append(report)
            else:
                # Calibrate posterior confidence
                conf = 0.80
                if val.is_unit_consistent:
                    conf += 0.06
                if rec.reported_log10_k is not None:
                    conf += 0.04
                if prose_corroborated:
                    conf += 0.06
                if val.speciation_check is not None and val.speciation_consistent:
                    conf += 0.03
                if repairs:
                    conf -= 0.03
                conf = round(min(0.99, max(0.50, conf)), 3)

                status_val: Literal["VERIFIED", "AUTO_REPAIRED"] = (
                    "AUTO_REPAIRED" if repairs else "VERIFIED"
                )
                auditor_v: Literal["VERIFIED", "CORRECTED"] = (
                    "CORRECTED" if repairs else "VERIFIED"
                )
                msgs = list(diagnostics)
                if val.speciation_note:
                    msgs.append(val.speciation_note)
                if not msgs:
                    msgs.append("All RDKit structure, unit/log, physical bound, and provenance checks passed.")

                report = ChemAuditReport(
                    record_id=rec.record_id,
                    status=status_val,
                    extractor_confidence=0.78,
                    calibrated_confidence=conf,
                    prose_corroborated=prose_corroborated,
                    structure_verified=True,
                    unit_and_log_verified=True,
                    physics_bounds_verified=True,
                    speciation_verified=True,
                    repairs_applied=repairs,
                    error_flags=list(dict.fromkeys(error_flags)),
                    diagnostic_messages=msgs,
                )
                rec.audit_report = report
                rec.audit_log = CrossCheckAuditLog(
                    record_id=rec.record_id,
                    extractor_confidence=0.78,
                    auditor_verdict=auditor_v,
                    prose_cross_checked=pm is not None,
                    speciation_resolved=val.speciation_check is not None,
                    resolved_species_form=rec.species_form,
                    corrections_applied=repairs,
                    calibrated_confidence=conf,
                )
                accepted.append(rec)
                reports.append(report)

        return ExtractionRunOutput(
            doc_id=doc_id,
            pipeline_mode="full",
            accepted_records=accepted,
            rejected_records=rejected,
            all_candidates=candidates,
            audit_reports=reports,
        )
