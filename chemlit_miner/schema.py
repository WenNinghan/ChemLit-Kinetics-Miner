"""Strict Pydantic schemas for Evidence-Traceable Reaction Kinetics Mining.

Connects DECAT (ozone/radical kinetics curation of organic micropollutants),
shiye (local-first scientific OCR & table/formula layout spans), and
OmniHarness (dual-agent Extractor x Domain-Physics Auditor cross-verification).
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


AuditStatus = Literal["VERIFIED", "AUTO_REPAIRED", "FLAGGED_ERROR"]


class EvidenceSpan(BaseModel):
    """Exact document provenance span for an extracted value or record."""

    doc_id: str = Field(..., description="Unique identifier of source document or DOI.")
    page_num: int = Field(default=1, ge=1, description="1-based page number in source PDF/document.")
    locator: str = Field(
        ...,
        description="Table/line structural coordinate, e.g. 'Table 1, Row 2' or 'Line 14 (Prose)'.",
    )
    char_start: int = Field(..., ge=0, description="Exact 0-based start character offset in source text.")
    char_end: int = Field(..., ge=0, description="Exact 0-based end character offset in source text.")
    verbatim_quote: str = Field(
        ...,
        min_length=1,
        description="Verbatim substring extracted from source document [char_start:char_end].",
    )

    def verify_against_source(self, source_text: str) -> bool:
        """Return True iff [char_start:char_end] slices the exact verbatim_quote from source_text."""
        if self.char_start < 0 or self.char_end > len(source_text) or self.char_start >= self.char_end:
            return False
        return source_text[self.char_start : self.char_end] == self.verbatim_quote


class SpeciationCheckResult(BaseModel):
    """Henderson-Hasselbalch acid-base speciation consistency audit outcome."""

    ph: float = Field(..., description="Experimental pH evaluated.")
    pka: float = Field(..., description="Primary acid dissociation constant pKa.")
    pka2: Optional[float] = Field(default=None, description="Optional second pKa for diprotic compounds.")
    alpha_0: float = Field(..., ge=0.0, le=1.0, description="Protonated/acid species fraction alpha_0.")
    alpha_1: float = Field(..., ge=0.0, le=1.0, description="Deprotonated/conjugate-base fraction alpha_1.")
    alpha_2: Optional[float] = Field(
        default=None, ge=0.0, le=1.0, description="Second deprotonated fraction alpha_2 if diprotic."
    )
    k_ha: float = Field(..., ge=0.0, description="Intrinsic rate constant of protonated form (M^-1 s^-1).")
    k_a_minus: float = Field(
        ..., ge=0.0, description="Intrinsic rate constant of conjugate base form (M^-1 s^-1)."
    )
    k_a2_minus: Optional[float] = Field(
        default=None, ge=0.0, description="Intrinsic rate constant of di-deprotonated form (M^-1 s^-1)."
    )
    predicted_k_app: float = Field(
        ..., ge=0.0, description="Theoretical k_app(pH) = alpha_0 * k_HA + alpha_1 * k_A- (M^-1 s^-1)."
    )
    reported_k_app: Optional[float] = Field(
        default=None, description="Reported apparent rate constant at pH (M^-1 s^-1), if provided."
    )
    log10_discrepancy: Optional[float] = Field(
        default=None, description="|log10(reported_k_app) - log10(predicted_k_app)|."
    )
    swapped_ph_pka_detected: bool = Field(
        default=False,
        description="True if swapping pH and pKa resolves a major speciation physics discrepancy.",
    )
    is_consistent: bool = Field(
        default=True, description="True if speciation physics check passes within tolerance."
    )
    note: str = Field(default="", description="Human-readable diagnostic summary.")


class ChemicalValidationResult(BaseModel):
    """Cheminformatics (RDKit) and physical/unit consistency validation outcome."""

    input_identifier: str = Field(..., description="Raw compound name or CAS number from text.")
    input_smiles: Optional[str] = Field(default=None, description="Raw SMILES string before sanitization.")
    canonical_smiles: Optional[str] = Field(default=None, description="RDKit canonical isomeric SMILES.")
    inchikey: Optional[str] = Field(default=None, description="Standard IUPAC InChIKey (27 chars).")
    molecular_formula: Optional[str] = Field(default=None, description="Hill-system molecular formula.")
    exact_mw: Optional[float] = Field(default=None, description="Exact monoisotopic molecular weight (Da).")
    formal_charge: Optional[int] = Field(default=None, description="Net formal charge of molecule.")
    is_valid_structure: bool = Field(
        default=False,
        description="True if RDKit successfully parses and sanitizes valence, aromaticity, and rings.",
    )
    valence_or_sanitization_error: Optional[str] = Field(
        default=None,
        description="Detailed RDKit sanitization or valence violation message if invalid.",
    )
    standardized_k_M_inv_s_inv: Optional[float] = Field(
        default=None,
        description="Second-order rate constant converted to canonical M^-1 s^-1 (L mol^-1 s^-1).",
    )
    unit_conversion_factor: float = Field(
        default=1.0,
        description="Multiplicative factor applied to convert reported k_value into M^-1 s^-1.",
    )
    recomputed_log10_k: Optional[float] = Field(
        default=None,
        description="Recomputed log10(k / M^-1 s^-1) from standardized rate constant.",
    )
    reported_log10_k: Optional[float] = Field(
        default=None,
        description="Explicitly reported log10(k) from table or text, if available.",
    )
    log_k_discrepancy: Optional[float] = Field(
        default=None,
        description="Absolute difference |reported_log10_k - recomputed_log10_k|.",
    )
    is_unit_consistent: bool = Field(
        default=True,
        description="True if unit conversion succeeds and log10(k) matches standardized k within tolerance.",
    )
    exceeds_diffusion_limit: bool = Field(
        default=False,
        description="True if k > 2.5e10 M^-1 s^-1 (exceeds Smoluchowski aqueous diffusion limit).",
    )
    invalid_temperature: bool = Field(
        default=False,
        description="True if temperature violates absolute physical bounds (T <= 0 K) or liquid water range.",
    )
    ph_pka_swap_suspected: bool = Field(
        default=False,
        description="True if pH and pKa values appear transposed relative to chemical structure/buffer.",
    )
    speciation_consistent: bool = Field(
        default=True,
        description="True if pH, pKa, and assigned species reactivity satisfy Henderson-Hasselbalch physics.",
    )
    speciation_check: Optional[SpeciationCheckResult] = Field(
        default=None,
        description="Detailed speciation calculation result when species constants are available.",
    )
    speciation_note: Optional[str] = Field(
        default=None,
        description="Explanation of Henderson-Hasselbalch speciation check (e.g. HA vs A-).",
    )
    error_codes: List[str] = Field(
        default_factory=list,
        description="Machine-readable failure mode tags (e.g. 'INVALID_VALENCE', 'LOG_K_MISMATCH', 'DIFFUSION_LIMIT').",
    )
    validation_warnings: List[str] = Field(
        default_factory=list,
        description="List of specific cheminformatics or physical consistency warnings.",
    )


class ChemAuditReport(BaseModel):
    """Unified cheminformatics & domain-physics audit report for a mined kinetics record."""

    record_id: str = Field(..., description="Identifier of the audited ReactionKineticsRecord.")
    status: AuditStatus = Field(
        ...,
        description="Audit verdict: 'VERIFIED', 'AUTO_REPAIRED', or 'FLAGGED_ERROR'.",
    )
    extractor_confidence: float = Field(
        default=0.75, ge=0.0, le=1.0, description="Initial single-pass extractor confidence."
    )
    calibrated_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Posterior confidence integrating provenance, RDKit validity, and physical audits.",
    )
    prose_corroborated: bool = Field(
        default=False,
        description="True if table value was corroborated or repaired via body prose / footnotes.",
    )
    structure_verified: bool = Field(
        default=False,
        description="True if canonical RDKit molecular structure and InChIKey passed sanitization.",
    )
    unit_and_log_verified: bool = Field(
        default=False,
        description="True if unit conversion and log10(k) parity check passed.",
    )
    physics_bounds_verified: bool = Field(
        default=False,
        description="True if Smoluchowski diffusion limit and thermodynamic temperature bounds passed.",
    )
    speciation_verified: bool = Field(
        default=True,
        description="True if Henderson-Hasselbalch acid-base speciation check passed (or N/A).",
    )
    repairs_applied: List[str] = Field(
        default_factory=list,
        description="Audit trail of deterministic auto-repairs applied (e.g., OCR exponent fix, pH/pKa unswap).",
    )
    error_flags: List[str] = Field(
        default_factory=list,
        description="Machine-readable error taxonomy codes when status is FLAGGED_ERROR or AUTO_REPAIRED.",
    )
    diagnostic_messages: List[str] = Field(
        default_factory=list,
        description="Detailed human-readable audit notes.",
    )


class CrossCheckAuditLog(BaseModel):
    """Detailed dual-agent Generator-Auditor trace (compatible with OmniHarness audit pipelines)."""

    record_id: str = Field(..., description="Identifier of the audited ReactionKineticsRecord.")
    extractor_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    auditor_verdict: Literal[
        "VERIFIED",
        "CORRECTED",
        "REJECTED_HALLUCINATION",
        "REJECTED_PHYSICS_VIOLATION",
    ] = Field(..., description="Decision of the Domain-Physics Auditor Agent.")
    prose_cross_checked: bool = Field(default=False)
    speciation_resolved: bool = Field(default=False)
    resolved_species_form: str = Field(default="apparent_k_app")
    corrections_applied: List[str] = Field(default_factory=list)
    rejection_reasons: List[str] = Field(default_factory=list)
    calibrated_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class ReactionKineticsRecord(BaseModel):
    """Canonical schema for an extracted aqueous oxidation reaction rate constant record."""

    record_id: str = Field(..., description="Unique record ID (e.g., 'DOC001_R01').")
    doc_id: str = Field(..., description="Source literature identifier.")
    compound_name: str = Field(..., description="Name of organic micropollutant.")
    cas_number: Optional[str] = Field(default=None, description="CAS Registry Number if present.")
    compound_class: str = Field(
        default="Phenols",
        description="Chemical family: 'Phenols', 'Sulfonamides', 'Beta-blockers', 'Triazines', 'Olefins', 'Amines'.",
    )
    raw_smiles: Optional[str] = Field(default=None, description="Extracted or looked-up raw SMILES.")
    canonical_smiles: Optional[str] = Field(default=None, description="RDKit-validated canonical SMILES.")
    inchikey: Optional[str] = Field(default=None, description="RDKit-generated IUPAC InChIKey.")
    molecular_formula: Optional[str] = Field(default=None, description="RDKit molecular formula.")
    exact_mw: Optional[float] = Field(default=None, description="Exact monoisotopic mass (Da).")
    formal_charge: Optional[int] = Field(default=None, description="Formal molecular charge.")
    oxidant: Literal["O3", "•OH", "SO4•-", "ClO•"] = Field(
        ..., description="Oxidant species ('O3', '•OH', 'SO4•-', or 'ClO•')."
    )
    raw_k_string: str = Field(..., description="Verbatim rate constant string before normalization.")
    inequality: Optional[Literal["<", "<=", ">", ">=", "~"]] = Field(
        default=None,
        description="Inequality or approximation qualifier if rate constant is an upper/lower bound (e.g. k_O3 < 0.1).",
    )
    k_value: float = Field(..., description="Parsed numeric value of rate constant in reported units.")
    k_uncertainty: Optional[float] = Field(
        default=None, description="Reported +/- uncertainty in the same units as k_value."
    )
    k_units: str = Field(
        default="M^-1 s^-1",
        description="Reported units (e.g., 'M^-1 s^-1', 'L mol^-1 s^-1', 'mM^-1 s^-1', 'M^-1 min^-1').",
    )
    standardized_k: float = Field(
        ..., description="Standardized second-order rate constant in M^-1 s^-1."
    )
    reported_log10_k: Optional[float] = Field(
        default=None, description="Explicitly reported log10(k) value if tabulated or stated."
    )
    ph: Optional[float] = Field(default=None, description="Experimental pH (None if intrinsic species k).")
    pka: Optional[float] = Field(default=None, description="Acid dissociation constant pKa of compound.")
    pka2: Optional[float] = Field(default=None, description="Second pKa if diprotic.")
    k_ha: Optional[float] = Field(
        default=None, description="Intrinsic neutral/acid species rate constant k_HA (M^-1 s^-1) if reported."
    )
    k_a_minus: Optional[float] = Field(
        default=None, description="Intrinsic conjugate-base rate constant k_A- (M^-1 s^-1) if reported."
    )
    temperature_c: float = Field(
        default=20.0, description="Experimental temperature in degrees Celsius (°C)."
    )
    temperature_k: float = Field(
        default=293.15, description="Experimental absolute temperature in Kelvin (K)."
    )
    temperature_defaulted: bool = Field(
        default=False,
        description="True if temperature was inherited from general methods or defaulted to 20 °C.",
    )
    water_matrix: str = Field(
        default="phosphate_buffer",
        description="Aqueous matrix (e.g., 'phosphate_buffer', 'ultrapure_water', 'bicarbonate_free_buffer').",
    )
    radical_scavenger: Optional[str] = Field(
        default=None,
        description="Radical scavenger used during ozonation (e.g., 'tert-butanol', 'DMSO').",
    )
    species_form: str = Field(
        default="apparent_k_app",
        description="Reactive form ('neutral_HA', 'anionic_A-', 'cationic_BH+', 'zwitterion', 'apparent_k_app').",
    )
    evidence: EvidenceSpan = Field(..., description="Primary table or text evidence span.")
    prose_evidence: Optional[EvidenceSpan] = Field(
        default=None, description="Secondary corroborating prose evidence span, if found."
    )
    validation: Optional[ChemicalValidationResult] = Field(
        default=None, description="RDKit & physical unit consistency validation result."
    )
    audit_report: Optional[ChemAuditReport] = Field(
        default=None, description="Unified ChemAuditReport with status VERIFIED / AUTO_REPAIRED / FLAGGED_ERROR."
    )
    audit_log: Optional[CrossCheckAuditLog] = Field(
        default=None, description="Dual-Agent Generator-Auditor trace."
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional parser metadata.")
