"""Build the Gold-Standard Environmental Oxidation Kinetics Benchmark Corpus.

Generates `data/benchmark_corpus.json` containing 48 realistic environmental
chemistry literature excerpts (Markdown tables, LaTeX tabulars, and OCR/prose)
across 6 micropollutant families (Phenols, Sulfonamides, Beta-blockers, Triazines,
Olefins, Amines) and 3 major oxidants (O3, •OH, SO4•-), with systematically
injected real-world failure modes:
  - Clean gold records (diverse scientific notations, inequalities, Kelvin/Celsius, LaTeX)
  - OCR base-10 glyph confusions (`l0^5`, `1O^6`, Unicode superscripts `10⁵`)
  - Millimolar & per-minute unit traps (`mM^-1 s^-1`, `M^-1 min^-1`, footnotes, `cm^3 molecule^-1 s^-1`)
  - Pseudo-first-order rate traps (`min^-1`, `s^-1` without [Ox])
  - Invalid valence & broken-ring SMILES hallucinations (pentavalent C/N, unclosed ring, skeleton mismatch)
  - OCR integer-decade exponent shifts (recoverable via log10(k)/prose vs unrecoverable)
  - Smoluchowski diffusion-limit violations (k > 2.5e10 M^-1 s^-1) & negative Kelvin temperatures
  - pH <-> pKa column swaps & Henderson-Hasselbalch speciation physics violations
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rdkit import Chem
from chemlit_miner.chem_validator import lookup_reference_compound


def ref_info(name: str) -> Dict[str, Any]:
    ref = lookup_reference_compound(name)
    assert ref is not None, f"Missing reference compound: {name}"
    mol = Chem.MolFromSmiles(ref.smiles)
    assert mol is not None
    return {
        "compound_name": ref.name,
        "compound_class": ref.compound_class,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "inchikey": Chem.MolToInchiKey(mol),
        "pka": ref.pka,
    }


def gold_item(
    name: str,
    oxidant: str,
    standardized_k: float,
    ph: Optional[float],
    temperature_c: float,
    inequality: Optional[str] = None,
    failure_category: str = "CLEAN_GOLD",
) -> Dict[str, Any]:
    info = ref_info(name)
    return {
        "compound_name": info["compound_name"],
        "compound_class": info["compound_class"],
        "inchikey": info["inchikey"],
        "canonical_smiles": info["canonical_smiles"],
        "oxidant": oxidant,
        "standardized_k": standardized_k,
        "inequality": inequality,
        "ph": ph,
        "pka": info["pka"],
        "temperature_c": temperature_c,
        "failure_category": failure_category,
    }


def trap_item(
    compound_name: str,
    compound_class: str,
    oxidant: str,
    trap_type: str,
    description: str,
    raw_reported_k: Optional[float] = None,
    reported_log10_k: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "compound_name": compound_name,
        "compound_class": compound_class,
        "oxidant": oxidant,
        "trap_type": trap_type,
        "description": description,
        "raw_reported_k": raw_reported_k,
        "reported_log10_k": reported_log10_k,
    }


def build_excerpts() -> List[Dict[str, Any]]:
    excerpts: List[Dict[str, Any]] = []

    # --- 1. Phenols: Clean Markdown Table + Speciation Prose ---
    excerpts.append({
        "doc_id": "DOC_001",
        "source_citation": "Hoigne & Bader (1983) Water Res. 17, 185-194 (adapted excerpt)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Aqueous ozonation experiments were conducted in 10 mM phosphate buffer at pH = 7.0 "
            "and T = 20 °C in the presence of tert-butanol (10 mM) as radical scavenger.\n\n"
            "| Compound | SMILES | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Phenol | Oc1ccccc1 | O3 | (1.57 ± 0.12) × 10^6 | 6.20 | 7.0 | 9.95 | 1.3 × 10^3 | 1.4 × 10^9 |\n"
            "| 4-Chlorophenol | Oc1ccc(Cl)cc1 | O3 | 2.34 × 10^6 | 6.37 | 7.0 | 9.41 | 6.0 × 10^2 | 6.0 × 10^8 |\n\n"
            "For Phenol (pKa = 9.95), the apparent rate constant at pH = 7.0 was k_O3 = (1.57 ± 0.12) × 10^6 M^-1 s^-1."
        ),
        "gold_records": [
            gold_item("Phenol", "O3", 1.57e6, 7.0, 20.0),
            gold_item("4-Chlorophenol", "O3", 2.34e6, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 2. Phenols: Triclosan pH <-> pKa Column Swap (Auto-Repairable) ---
    excerpts.append({
        "doc_id": "DOC_002",
        "source_citation": "Suarez et al. (2007) Environ. Sci. Technol. 41, 3434 (OCR column-swap benchmark)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Ozonation of chlorinated phenolic antimicrobials was investigated at pH = 7.0 (phosphate buffer) and T = 298 K.\n\n"
            "| Compound | SMILES | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Triclosan | Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl | O3 | 4.25 × 10^6 | 6.63 | 7.90 | 7.00 | 13 | 3.8 × 10^7 |\n\n"
            "At pH 7.0, Triclosan (pKa = 7.90) exhibited rapid ozone abatement with k_O3 = 4.25 × 10^6 M^-1 s^-1."
        ),
        "gold_records": [
            gold_item("Triclosan", "O3", 4.25e6, 7.0, 24.85, failure_category="PH_PKA_SWAP_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 3. Phenols: Invalid Valence Hallucination (Pentavalent Carbon on Bisphenol A) ---
    excerpts.append({
        "doc_id": "DOC_003",
        "source_citation": "Deborde et al. (2005) Water Res. (OCSR valence hallucination test)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Hydroxyl radical competition kinetics in ultrapure water at pH = 7.0 and 25 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (L mol^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Bisphenol A | CC(=O)(=O)(c1ccc(O)cc1)c1ccc(O)cc1 | •OH | 1.02 × 10^10 | 10.01 | 7.0 | 25.0 |\n"
            "| Acetaminophen | CC(=O)Nc1ccc(O)cc1 | •OH | 1.70 × 10^9 | 9.23 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Acetaminophen", "•OH", 1.70e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Bisphenol A",
                "Phenols",
                "•OH",
                "INVALID_VALENCE_HALLUCINATION",
                "Pentavalent/hexavalent central carbon CC(=O)(=O)(c1...)c1...",
                raw_reported_k=1.02e10,
                reported_log10_k=10.01,
            )
        ],
    })

    # --- 4. Sulfonamides: LaTeX Table with `\cdot` and `L mol^{-1} s^{-1}` ---
    excerpts.append({
        "doc_id": "DOC_004",
        "source_citation": "Huber et al. (2003) Environ. Sci. Technol. 37, 1016-1024",
        "format_type": "latex_table_and_prose",
        "primary_family": "Sulfonamides",
        "text": (
            "Second-order rate constants for sulfonamide antibiotics in phosphate buffer (pH = 7.0, T = 293.15 K).\n"
            "\\begin{tabular}{l l c c c c}\n"
            "\\hline\n"
            "Compound & SMILES & Oxidant & $k$ ($\\mathrm{L\\cdot mol^{-1}\\cdot s^{-1}}$) & $\\log k$ & pH \\\\\n"
            "\\hline\n"
            "Sulfamethoxazole & Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1 & O3 & $5.5 \\cdot 10^{5}$ & 5.74 & 7.0 \\\\\n"
            "Sulfadiazine & Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1 & •OH & $5.3 \\cdot 10^{9}$ & 9.72 & 7.0 \\\\\n"
            "\\hline\n"
            "\\end{tabular}\n"
        ),
        "gold_records": [
            gold_item("Sulfamethoxazole", "O3", 5.5e5, 7.0, 20.0),
            gold_item("Sulfadiazine", "•OH", 5.3e9, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 5. Sulfonamides: Footnote Millimolar Unit Trap (`mM^-1 s^-1`) ---
    excerpts.append({
        "doc_id": "DOC_005",
        "source_citation": "Boreen et al. (2004) Environ. Sci. Technol. (footnote unit trap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Sulfonamides",
        "text": (
            "Ozonation kinetics of veterinary sulfonamides in carbonate-free buffer at pH = 7.5 and 20 °C.\n\n"
            "| Compound | Oxidant | k_O3^a | log k (M^-1 s^-1) | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Sulfamethazine | O3 | 6.2 × 10^2^a | 5.79 | 7.5 | 20.0 |\n"
            "| Sulfapyridine | O3 | 1.8 × 10^2^a | 5.26 | 7.5 | 20.0 |\n\n"
            "^a Apparent second-order rate constants reported in mM^-1 s^-1.\n"
            "In prose, Sulfamethazine reacted with ozone at 6.2 × 10^2 mM^-1 s^-1, and Sulfapyridine at 1.8 × 10^2 mM^-1 s^-1."
        ),
        "gold_records": [
            gold_item("Sulfamethazine", "O3", 6.2e5, 7.5, 20.0, failure_category="UNIT_TRAP_REPAIRABLE"),
            gold_item("Sulfapyridine", "O3", 1.8e5, 7.5, 20.0, failure_category="UNIT_TRAP_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 6. Sulfonamides: Broken Aromatic Ring SMILES + Clean Sulfate Radical ---
    excerpts.append({
        "doc_id": "DOC_006",
        "source_citation": "Mahdi Ahmed et al. (2012) J. Hazard. Mater. (broken ring OCSR trap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Sulfonamides",
        "text": (
            "Sulfate radical oxidation in ultrapure water at pH = 7.0 and 25 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Sulfamethoxazole | Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no | SO4•- | 2.9 × 10^9 | 9.46 | 7.0 | 25.0 |\n"
            "| Sulfachloropyridazine | Nc1ccc(S(=O)(=O)Nc2ccc(Cl)nn2)cc1 | SO4•- | 3.4 × 10^9 | 9.53 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Sulfachloropyridazine", "SO4•-", 3.4e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Sulfamethoxazole",
                "Sulfonamides",
                "SO4•-",
                "INVALID_VALENCE_HALLUCINATION",
                "Unclosed ring index '1' in truncated SMILES Cc1cc(...)no",
                raw_reported_k=2.9e9,
                reported_log10_k=9.46,
            )
        ],
    })

    # --- 7. Beta-blockers: OCR Glyph `l0^5` & Unicode Superscript `10⁹` ---
    excerpts.append({
        "doc_id": "DOC_007",
        "source_citation": "Benner et al. (2008) Environ. Sci. Technol. 42, 5417 (OCR scan excerpt)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Beta-blockers",
        "text": (
            "Kinetics of beta-blockers with ozone and hydroxyl radicals in phosphate buffer at pH = 7.0, T = 20 °C.\n\n"
            "| Compound | Oxidant | k (M^-1 s^-1) | log k | pH | pKa |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Propranolol | O3 | (1.0 ± 0.1) × l0^5 | 5.00 | 7.0 | 9.45 |\n"
            "| Metoprolol | O3 | 2.0 × 1O^3 | 3.30 | 7.0 | 9.68 |\n"
            "| Atenolol | •OH | 8.0 × 10⁹ M⁻¹ s⁻¹ | 9.90 | 7.0 | 9.60 |\n"
        ),
        "gold_records": [
            gold_item("Propranolol", "O3", 1.0e5, 7.0, 20.0, failure_category="OCR_GLYPH_BASE10"),
            gold_item("Metoprolol", "O3", 2.0e3, 7.0, 20.0, failure_category="OCR_GLYPH_BASE10"),
            gold_item("Atenolol", "•OH", 8.0e9, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 8. Beta-blockers: OCR Exponent Shift in Table Repaired by log k + Prose ---
    excerpts.append({
        "doc_id": "DOC_008",
        "source_citation": "Lee et al. (2014) Water Res. (OCR 100x exponent misread in table)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Beta-blockers",
        "text": (
            "Direct ozonation of beta-blockers at pH = 7.0 (phosphate buffer) and 20 °C with tert-butanol.\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Sotalol | CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1 | O3 | 1.9 × 10^2 | 4.28 | 7.0 | 20.0 |\n"
            "| Nadolol | CC(C)(C)NCC(O)COc1cccc2c1CC(O)C(O)C2 | O3 | 3.5 × 10^3 | 3.54 | 7.0 | 20.0 |\n\n"
            "Sotalol displayed high ozone reactivity due to its sulfonanilide moiety, with k_O3 = 1.9 × 10^4 M^-1 s^-1 (log k = 4.28) at pH 7.0."
        ),
        "gold_records": [
            gold_item("Sotalol", "O3", 1.9e4, 7.0, 20.0, failure_category="OCR_EXPONENT_REPAIRABLE"),
            gold_item("Nadolol", "O3", 3.5e3, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 9. Beta-blockers: Smoluchowski Diffusion Limit Violation ---
    excerpts.append({
        "doc_id": "DOC_009",
        "source_citation": "Song et al. (2008) J. Phys. Chem. A (OCR diffusion-limit violation trap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Beta-blockers",
        "text": (
            "Pulse radiolysis measurements of hydroxyl radical rate constants at pH = 7.0 and 25 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Acebutolol | CCCC(=O)Nc1ccc(OCC(O)CNC(C)C)c(C(C)=O)c1 | •OH | 4.6 × 10^9 | 9.66 | 7.0 | 25.0 |\n"
            "| Bisoprolol | CC(C)OCCOc1ccc(COCC(O)CNC(C)C)cc1 | •OH | 8.4 × 10^11 | 11.92 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Acebutolol", "•OH", 4.6e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Bisoprolol",
                "Beta-blockers",
                "•OH",
                "DIFFUSION_LIMIT_VIOLATION",
                "k = 8.4e11 M^-1 s^-1 exceeds Smoluchowski aqueous limit 2.5e10 M^-1 s^-1",
                raw_reported_k=8.4e11,
                reported_log10_k=11.92,
            )
        ],
    })

    # --- 10. Triazines: Clean Markdown Multi-Oxidant Herbicides ---
    excerpts.append({
        "doc_id": "DOC_010",
        "source_citation": "Acero et al. (2000) Environ. Sci. Technol. 34, 591-597",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Triazines",
        "text": (
            "Oxidation of chloro-s-triazine herbicides in phosphate buffer (pH = 7.0, T = 20 °C).\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Atrazine | CCNc1nc(NC(C)C)nc(Cl)n1 | O3 | 6.0 ± 0.4 | 0.78 | 7.0 | 20.0 |\n"
            "| Atrazine | CCNc1nc(NC(C)C)nc(Cl)n1 | •OH | 3.0 × 10^9 | 9.48 | 7.0 | 20.0 |\n"
            "| Simazine | CCNc1nc(NCC)nc(Cl)n1 | O3 | 4.8 ± 0.3 | 0.68 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Atrazine", "O3", 6.0, 7.0, 20.0),
            gold_item("Atrazine", "•OH", 3.0e9, 7.0, 20.0),
            gold_item("Simazine", "O3", 4.8, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 11. Triazines: Per-Minute Unit Conversion (`M^-1 min^-1`) ---
    excerpts.append({
        "doc_id": "DOC_011",
        "source_citation": "Beltran et al. (1994) Ozone Sci. Eng. (minute time-base table)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Triazines",
        "text": (
            "Direct ozonation rate constants of s-triazines in phosphate buffer at pH = 7.0 and 20 °C.\n\n"
            "| Compound | Oxidant | k (M^-1 min^-1) | log k (M^-1 s^-1) | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Propazine | O3 | 420 | 0.85 | 7.0 | 20.0 |\n"
            "| Cyanazine | O3 | 90 | 0.18 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Propazine", "O3", 7.0, 7.0, 20.0, failure_category="UNIT_TRAP_REPAIRABLE"),
            gold_item("Cyanazine", "O3", 1.5, 7.0, 20.0, failure_category="UNIT_TRAP_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 12. Triazines: Pseudo-First-Order Rate Constant Trap (`min^-1`) ---
    excerpts.append({
        "doc_id": "DOC_012",
        "source_citation": "Lutze et al. (2015) Water Res. (pseudo-first-order k_obs trap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Triazines",
        "text": (
            "Sulfate radical degradation of methylthio-triazines at pH = 7.0 and 25 °C.\n\n"
            "| Compound | Oxidant | k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Prometryn | SO4•- | 4.2 × 10^9 M^-1 s^-1 | 7.0 | 25.0 |\n"
            "| Terbutryn | SO4•- | 0.085 min^-1 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Prometryn", "SO4•-", 4.2e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Terbutryn",
                "Triazines",
                "SO4•-",
                "PSEUDO_FIRST_ORDER_TRAP",
                "Reported pseudo-first-order k_obs in min^-1 without radical concentration",
                raw_reported_k=0.085,
            )
        ],
    })

    # --- 13. Olefins: Inequalities (`k_O3 < 0.1`) & High Reactivity (`Carbamazepine`) ---
    excerpts.append({
        "doc_id": "DOC_013",
        "source_citation": "Hoigne & Bader (1983); Huber et al. (2003) (olefin reactivity span)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Olefins",
        "text": (
            "Reactivity of activated vs chlorinated ethylenic double bonds toward ozone at pH = 7.0, 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Carbamazepine | NC(=O)N1c2ccccc2C=Cc2ccccc21 | O3 | (3.0 ± 0.2) × 10^5 | 7.0 | 20.0 |\n"
            "| Trichloroethylene | ClC=C(Cl)Cl | O3 | 17 ± 2 | 7.0 | 20.0 |\n"
            "| Tetrachloroethylene | ClC(Cl)=C(Cl)Cl | O3 | < 0.1 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Carbamazepine", "O3", 3.0e5, 7.0, 20.0),
            gold_item("Trichloroethylene", "O3", 17.0, 7.0, 20.0),
            gold_item("Tetrachloroethylene", "O3", 0.1, 7.0, 20.0, inequality="<"),
        ],
        "injected_traps": [],
    })

    # --- 14. Olefins: Standalone Prose Extraction with `log k = 5.48` & Kelvin ---
    excerpts.append({
        "doc_id": "DOC_014",
        "source_citation": "Leitzke et al. (2001) J. Chem. Soc. Perkin Trans. 2 (prose-only excerpt)",
        "format_type": "prose_only",
        "primary_family": "Olefins",
        "text": (
            "Stopped-flow spectrophotometric measurements were carried out in 20 mM phosphate buffer "
            "at pH = 7.0 and T = 298.15 K with tert-butanol scavenger. "
            "Styrene reacted rapidly with aqueous ozone with k_O3 = (3.0 ± 0.3) × 10^5 M^-1 s^-1 (log k = 5.48). "
            "Under identical conditions, Acrolein exhibited a second-order rate constant of k_O3 = 2.8 × 10^4 M^-1 s^-1."
        ),
        "gold_records": [
            gold_item("Styrene", "O3", 3.0e5, 7.0, 25.0),
            gold_item("Acrolein", "O3", 2.8e4, 7.0, 25.0),
        ],
        "injected_traps": [],
    })

    # --- 15. Olefins: Negative Kelvin Temperature Trap (`-20 K`) ---
    excerpts.append({
        "doc_id": "DOC_015",
        "source_citation": "Dowideit & von Sonntag (1998) Environ. Sci. Technol. (OCR negative Kelvin trap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Olefins",
        "text": (
            "Ozonation of unsaturated carboxylic acids and chloroethenes in phosphate buffer at pH = 7.0.\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| trans-Cinnamic acid | O=C(O)/C=C/c1ccccc1 | O3 | 3.8 × 10^5 | 5.58 | 7.0 | 293.15 K |\n"
            "| cis-1,2-Dichloroethylene | Cl/C=C\\Cl | O3 | 8.0 × 10^2 | 2.90 | 7.0 | -20 K |\n"
        ),
        "gold_records": [
            gold_item("trans-Cinnamic acid", "O3", 3.8e5, 7.0, 20.0),
        ],
        "injected_traps": [
            trap_item(
                "cis-1,2-Dichloroethylene",
                "Olefins",
                "O3",
                "NEGATIVE_KELVIN_VIOLATION",
                "Thermodynamically impossible temperature T = -20 K (-293.15 °C)",
                raw_reported_k=800.0,
                reported_log10_k=2.90,
            )
        ],
    })

    # --- 16. Amines: Speciation Verification for Aliphatic Amines ---
    excerpts.append({
        "doc_id": "DOC_016",
        "source_citation": "Lee & von Gunten (2010) Water Res. 44, 555-566 (amine speciation)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Amines",
        "text": (
            "Ozonation of aliphatic amines in phosphate buffer at pH = 7.0 and T = 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Trimethylamine | CN(C)C | O3 | 6.49 × 10^3 | 3.81 | 7.0 | 9.80 | 0.05 | 4.1 × 10^6 |\n"
            "| Dimethylamine | CNC | O3 | 3.53 × 10^3 | 3.55 | 7.0 | 10.73 | 0.01 | 1.9 × 10^7 |\n"
        ),
        "gold_records": [
            gold_item("Trimethylamine", "O3", 6.49e3, 7.0, 20.0),
            gold_item("Dimethylamine", "O3", 3.53e3, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 17. Amines: Pentavalent Nitrogen SMILES Hallucination + Valid Diclofenac ---
    excerpts.append({
        "doc_id": "DOC_017",
        "source_citation": "Dodd et al. (2006) Environ. Sci. Technol. 40, 1969 (pentavalent N trap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Amines",
        "text": (
            "Apparent ozonation rate constants of amine-bearing pharmaceuticals at pH = 7.0, 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Triethylamine | CCN(CC)(CC)(=O)C | O3 | 3.7 × 10^2 | 2.57 | 7.0 | 20.0 |\n"
            "| Diclofenac | O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl | O3 | 1.0 × 10^6 | 6.00 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Diclofenac", "O3", 1.0e6, 7.0, 20.0),
        ],
        "injected_traps": [
            trap_item(
                "Triethylamine",
                "Amines",
                "O3",
                "INVALID_VALENCE_HALLUCINATION",
                "Pentavalent neutral nitrogen CCN(CC)(CC)(=O)C rejected by RDKit valence check",
                raw_reported_k=370.0,
                reported_log10_k=2.57,
            )
        ],
    })

    # --- 18. Amines: Hybrid Table + Extra Prose Compound (`Trimethoprim`) ---
    excerpts.append({
        "doc_id": "DOC_018",
        "source_citation": "Dodd et al. (2006) Environ. Sci. Technol. (hybrid table + prose supplement)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Amines",
        "text": (
            "Fluoroquinolone and diaminopyrimidine antibiotics were ozonated at pH = 7.0 (phosphate buffer, 20 °C).\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Ciprofloxacin | O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O | O3 | 1.9 × 10^4 | 4.28 | 7.0 | 20.0 |\n\n"
            "In parallel competition experiments at pH 7.0 and 20 °C, Trimethoprim reacted with ozone with k_O3 = 2.7 × 10^5 M^-1 s^-1 (log k = 5.43)."
        ),
        "gold_records": [
            gold_item("Ciprofloxacin", "O3", 1.9e4, 7.0, 20.0),
            gold_item("Trimethoprim", "O3", 2.7e5, 7.0, 20.0, failure_category="HYBRID_PROSE_SUPPLEMENT"),
        ],
        "injected_traps": [],
    })

    # --- 19. Phenols: Speciation Physics Violation Trap (Physically Impossible k_app) ---
    excerpts.append({
        "doc_id": "DOC_019",
        "source_citation": "Synthetic Speciation Violation Stress Test (Phenols)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Acid-base speciation audit of chlorophenols at pH = 6.0 and 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| 2,4-Dichlorophenol | Oc1ccc(Cl)cc1Cl | O3 | 1.02 × 10^7 | 7.01 | 6.0 | 7.89 | 1.5 × 10^3 | 8.0 × 10^8 |\n"
            "| 2,4,6-Trichlorophenol | Oc1c(Cl)cc(Cl)cc1Cl | O3 | 5.0 × 10^2 | 2.70 | 6.0 | 6.15 | 1.0 × 10^4 | 5.0 × 10^8 |\n"
        ),
        "gold_records": [
            gold_item("2,4-Dichlorophenol", "O3", 1.02e7, 6.0, 20.0),
        ],
        "injected_traps": [
            trap_item(
                "2,4,6-Trichlorophenol",
                "Phenols",
                "O3",
                "SPECIATION_VIOLATION",
                "Reported k_app=500 M^-1 s^-1 at pH 6.0 violates k_HA=1e4, k_A-=5e8 (predicted 2.07e8)",
                raw_reported_k=500.0,
                reported_log10_k=2.70,
            )
        ],
    })

    # --- 20. Phenols: Endocrine Disruptors LaTeX Table ---
    excerpts.append({
        "doc_id": "DOC_020",
        "source_citation": "Huber et al. (2003) Environ. Sci. Technol. (estrogens LaTeX table)",
        "format_type": "latex_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Estrogenic micropollutant oxidation in phosphate buffer at pH = 7.0 and T = 293.15 K.\n"
            "\\begin{tabular}{l c c c c}\n"
            "\\toprule\n"
            "Compound & Oxidant & $k$ ($\\mathrm{M^{-1}\\,s^{-1}}$) & $\\log k$ & pH \\\\\n"
            "\\midrule\n"
            "17alpha-Ethinylestradiol & O3 & $(3.0 \\pm 0.3) \\times 10^{6}$ & 6.48 & 7.0 \\\\\n"
            "Estrone & O3 & $1.5 \\times 10^{6}$ & 6.18 & 7.0 \\\\\n"
            "17alpha-Ethinylestradiol & •OH & $9.8 \\times 10^{9}$ & 9.99 & 7.0 \\\\\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        ),
        "gold_records": [
            gold_item("17alpha-Ethinylestradiol", "O3", 3.0e6, 7.0, 20.0),
            gold_item("Estrone", "O3", 1.5e6, 7.0, 20.0),
            gold_item("17alpha-Ethinylestradiol", "•OH", 9.8e9, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 21. Sulfonamides: pH <-> pKa Swap on Sulfamethoxazole (Repairable) ---
    excerpts.append({
        "doc_id": "DOC_021",
        "source_citation": "Dodd et al. (2006) (transposed pH/pKa table columns)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Sulfonamides",
        "text": (
            "Apparent ozonation rate constants measured in pH = 7.0 phosphate buffer at 20 °C.\n\n"
            "| Compound | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Sulfamethoxazole | O3 | 5.45 × 10^5 | 5.74 | 5.70 | 7.00 | 4.7 × 10^4 | 5.7 × 10^5 |\n\n"
            "At pH = 7.0, Sulfamethoxazole (pKa = 5.70) is predominantly anionic, yielding k_O3 = 5.45 × 10^5 M^-1 s^-1."
        ),
        "gold_records": [
            gold_item("Sulfamethoxazole", "O3", 5.45e5, 7.0, 20.0, failure_category="PH_PKA_SWAP_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 22. Sulfonamides: Name-SMILES Skeleton Mismatch Hallucination ---
    excerpts.append({
        "doc_id": "DOC_022",
        "source_citation": "Boreen et al. (2005) (LLM SMILES skeleton swap trap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Sulfonamides",
        "text": (
            "Hydroxyl radical reaction rates in phosphate buffer at pH = 7.0, 25 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Sulfathiazole | c1ccccc1 | •OH | 6.1 × 10^9 | 9.79 | 7.0 | 25.0 |\n"
            "| Sulfamerazine | Cc1ccnc(NS(=O)(=O)c2ccc(N)cc2)n1 | •OH | 5.8 × 10^9 | 9.76 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Sulfamerazine", "•OH", 5.8e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Sulfathiazole",
                "Sulfonamides",
                "•OH",
                "INVALID_VALENCE_HALLUCINATION",
                "LLM assigned benzene SMILES (c1ccccc1) to Sulfathiazole (InChIKey skeleton mismatch)",
                raw_reported_k=6.1e9,
                reported_log10_k=9.79,
            )
        ],
    })

    # --- 23. Triazines: Molecular Gas/Aqueous Unit `cm^3 molecule^-1 s^-1` ---
    # 4.98e-12 cm^3 molecule^-1 s^-1 * 6.02214e20 = 3.00e9 M^-1 s^-1 (log k = 9.48)
    excerpts.append({
        "doc_id": "DOC_023",
        "source_citation": "Buxton et al. (1988) J. Phys. Chem. Ref. Data (molecular unit convention)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Triazines",
        "text": (
            "Compiled hydroxyl radical rate constants converted from molecular units at pH = 7.0 and 25 °C.\n\n"
            "| Compound | Oxidant | k (cm^3 molecule^-1 s^-1) | log k (M^-1 s^-1) | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Desethylatrazine | •OH | 1.99 × 10^-12 | 9.08 | 7.0 | 25.0 |\n"
            "| Simazine | •OH | 4.65 × 10^-12 | 9.45 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Desethylatrazine", "•OH", 1.1984e9, 7.0, 25.0, failure_category="UNIT_TRAP_REPAIRABLE"),
            gold_item("Simazine", "•OH", 2.8003e9, 7.0, 25.0, failure_category="UNIT_TRAP_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 24. Phenols: OCR 1000x Exponent Misread Repaired via Prose ---
    excerpts.append({
        "doc_id": "DOC_024",
        "source_citation": "Neta et al. (1988) J. Phys. Chem. Ref. Data (OCR 10^6 vs 10^9 table misread)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Sulfate radical oxidation of substituted phenols at pH = 7.0 and 25 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| p-Cresol | Cc1ccc(O)cc1 | SO4•- | 4.2 × 10^6 | 9.62 | 7.0 | 25.0 |\n"
            "| Guaiacol | COc1ccccc1O | SO4•- | 5.1 × 10^9 | 9.71 | 7.0 | 25.0 |\n\n"
            "Laser flash photolysis confirmed diffusion-approaching electron transfer for p-Cresol with k_{SO4•-} = 4.2 × 10^9 M^-1 s^-1 (log k = 9.62)."
        ),
        "gold_records": [
            gold_item("p-Cresol", "SO4•-", 4.2e9, 7.0, 25.0, failure_category="OCR_EXPONENT_REPAIRABLE"),
            gold_item("Guaiacol", "SO4•-", 5.1e9, 7.0, 25.0),
        ],
        "injected_traps": [],
    })

    # --- 25. Olefins: Unrecoverable Log-k Parity Error Trap ---
    excerpts.append({
        "doc_id": "DOC_025",
        "source_citation": "Corrupted OCSR Table Benchmark (Olefins log-k contradiction)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Olefins",
        "text": (
            "Ozonation and radical oxidation of food preservatives and aromatics at pH = 7.0, 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Sorbic acid | CC=CC=CC(=O)O | O3 | 1.2 × 10^6 | 6.08 | 7.0 | 20.0 |\n"
            "| Nitrobenzene | O=[N+]([O-])c1ccccc1 | O3 | 9.4 × 10^4 | 0.95 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Sorbic acid", "O3", 1.2e6, 7.0, 20.0),
        ],
        "injected_traps": [
            trap_item(
                "Nitrobenzene",
                "Olefins",
                "O3",
                "UNRECOVERABLE_LOG_K_MISMATCH",
                "k=9.4e4 (log10=4.97) contradicts reported log k=0.95 (true Nitrobenzene k_O3=0.09) with no prose resolution",
                raw_reported_k=9.4e4,
                reported_log10_k=0.95,
            )
        ],
    })

    # --- 26. Amines: Prose-Only Hydroxyl & Sulfate Radical Kinetics ---
    excerpts.append({
        "doc_id": "DOC_026",
        "source_citation": "Wols & Hofman-Caris (2012) Water Res. (prose kinetics)",
        "format_type": "prose_only",
        "primary_family": "Amines",
        "text": (
            "Competition kinetics in phosphate buffer at pH = 7.0 and 25 °C established the radical reactivity of antihistamines and anilines. "
            "Diphenhydramine reacted rapidly with hydroxyl radicals with k_{•OH} = 5.4 × 10^9 M^-1 s^-1 (log k = 9.73). "
            "Aniline reacted with ozone at pH 7.0 with k_O3 = 9.0 × 10^7 M^-1 s^-1 (log k = 7.95)."
        ),
        "gold_records": [
            gold_item("Diphenhydramine", "•OH", 5.4e9, 7.0, 25.0),
            gold_item("Aniline", "O3", 9.0e7, 7.0, 25.0),
        ],
        "injected_traps": [],
    })

    # --- 27. Beta-blockers: pH <-> pKa Column Swap on Propranolol (Repairable) ---
    excerpts.append({
        "doc_id": "DOC_027",
        "source_citation": "Benner et al. (2008) (transposed pH/pKa beta-blocker table)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Beta-blockers",
        "text": (
            "Experiments were performed in 10 mM phosphate buffer at pH = 7.0 and 20 °C.\n\n"
            "| Compound | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Propranolol | O3 | 1.08 × 10^5 | 5.03 | 9.45 | 7.00 | 1.0 × 10^5 | 2.4 × 10^6 |\n\n"
            "At pH = 7.0, Propranolol (pKa = 9.45) exhibited k_O3 = 1.08 × 10^5 M^-1 s^-1."
        ),
        "gold_records": [
            gold_item("Propranolol", "O3", 1.08e5, 7.0, 20.0, failure_category="PH_PKA_SWAP_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 28. Phenols: Ungrounded Hallucinated Compound Name ---
    excerpts.append({
        "doc_id": "DOC_028",
        "source_citation": "LLM Hallucinated Compound Stress Test",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Acidic ozonation of phenolic compounds at pH = 2.0 and 20 °C.\n\n"
            "| Compound | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Phenol | O3 | 1.3 × 10^3 | 3.11 | 2.0 | 20.0 |\n"
            "| Perfluoro-hyperozonol-Z9 | O3 | 4.8 × 10^5 | 5.68 | 2.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Phenol", "O3", 1.3e3, 2.0, 20.0),
        ],
        "injected_traps": [
            trap_item(
                "Perfluoro-hyperozonol-Z9",
                "Phenols",
                "O3",
                "INVALID_VALENCE_HALLUCINATION",
                "Fabricated non-existent chemical compound with no valid SMILES",
                raw_reported_k=4.8e5,
                reported_log10_k=5.68,
            )
        ],
    })

    # --- 29. Sulfonamides: Smoluchowski Diffusion Limit Violation (`SO4•-`) ---
    excerpts.append({
        "doc_id": "DOC_029",
        "source_citation": "Laser Flash Photolysis OCR Artifact (Sulfonamides)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Sulfonamides",
        "text": (
            "Sulfate radical rate constants in phosphate buffer at pH = 7.0 and 25 °C.\n\n"
            "| Compound | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Sulfadiazine | SO4•- | 3.7 × 10^9 | 9.57 | 7.0 | 25.0 |\n"
            "| Sulfapyridine | SO4•- | 6.5 × 10^12 | 12.81 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Sulfadiazine", "SO4•-", 3.7e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Sulfapyridine",
                "Sulfonamides",
                "SO4•-",
                "DIFFUSION_LIMIT_VIOLATION",
                "k = 6.5e12 M^-1 s^-1 violates Smoluchowski diffusion limit (2.5e10 M^-1 s^-1)",
                raw_reported_k=6.5e12,
                reported_log10_k=12.81,
            )
        ],
    })

    # --- 30. Triazines: OCR Exponent Misread Repaired via Prose + Log k ---
    excerpts.append({
        "doc_id": "DOC_030",
        "source_citation": "Acero et al. (2000) (OCR 10^7 vs 10^9 hydroxyl radical misread)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Triazines",
        "text": (
            "Hydroxyl radical oxidation of triazines at pH = 7.0 and 20 °C.\n\n"
            "| Compound | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Propazine | •OH | 2.7 × 10^7 | 9.43 | 7.0 | 20.0 |\n"
            "| Cyanazine | •OH | 2.0 × 10^9 | 9.30 | 7.0 | 20.0 |\n\n"
            "Competition kinetics against para-chlorobenzoic acid yielded k_{•OH} = 2.7 × 10^9 M^-1 s^-1 (log k = 9.43) for Propazine."
        ),
        "gold_records": [
            gold_item("Propazine", "•OH", 2.7e9, 7.0, 20.0, failure_category="OCR_EXPONENT_REPAIRABLE"),
            gold_item("Cyanazine", "•OH", 2.0e9, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 31. Phenols: Pentachlorophenol & 4-Nitrophenol LaTeX Table ---
    excerpts.append({
        "doc_id": "DOC_031",
        "source_citation": "Hoigne & Bader (1983) (electron-withdrawing phenols LaTeX table)",
        "format_type": "latex_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Ozonation of nitro- and polychloro-phenols at pH = 2.0 and pH = 7.0 (T = 293.15 K).\n"
            "\\begin{tabular}{l c c c c}\n"
            "\\hline\n"
            "Compound & Oxidant & $k$ ($\\mathrm{M^{-1}\\,s^{-1}}$) & $\\log k$ & pH \\\\\n"
            "\\hline\n"
            "4-Nitrophenol & O3 & $50$ & 1.70 & 2.0 \\\\\n"
            "Pentachlorophenol & O3 & $3.1 \\times 10^{7}$ & 7.49 & 7.0 \\\\\n"
            "\\hline\n"
            "\\end{tabular}\n"
        ),
        "gold_records": [
            gold_item("4-Nitrophenol", "O3", 50.0, 2.0, 20.0),
            gold_item("Pentachlorophenol", "O3", 3.1e7, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 32. Amines: Footnote `mM^-1 s^-1` Trap on NDMA & Triethylamine ---
    excerpts.append({
        "doc_id": "DOC_032",
        "source_citation": "Lee et al. (2007) Environ. Sci. Technol. 41, 2056 (footnote mM^-1 s^-1)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Amines",
        "text": (
            "Hydroxyl radical oxidation of nitrosamines and tertiary amines at pH = 7.0, 25 °C.\n\n"
            "| Compound | Oxidant | k* | log k (M^-1 s^-1) | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| N-Nitrosodimethylamine | •OH | 4.3 × 10^5* | 8.63 | 7.0 | 25.0 |\n"
            "| Triethylamine | •OH | 5.1 × 10^6* | 9.71 | 7.0 | 25.0 |\n\n"
            "* Rate constants expressed in mM^-1 s^-1."
        ),
        "gold_records": [
            gold_item("N-Nitrosodimethylamine", "•OH", 4.3e8, 7.0, 25.0, failure_category="UNIT_TRAP_REPAIRABLE"),
            gold_item("Triethylamine", "•OH", 5.1e9, 7.0, 25.0, failure_category="UNIT_TRAP_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 33. Olefins: Carbamazepine •OH and SO4•- Clean Multi-Radical ---
    excerpts.append({
        "doc_id": "DOC_033",
        "source_citation": "Matta et al. (2011); Huber et al. (2003) (Carbamazepine AOP kinetics)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Olefins",
        "text": (
            "Radical oxidation of Carbamazepine in phosphate buffer at pH = 7.0 and 25 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Carbamazepine | NC(=O)N1c2ccccc2C=Cc2ccccc21 | •OH | (8.8 ± 1.2) × 10^9 | 9.94 | 7.0 | 25.0 |\n"
            "| Carbamazepine | NC(=O)N1c2ccccc2C=Cc2ccccc21 | SO4•- | (1.9 ± 0.2) × 10^9 | 9.28 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Carbamazepine", "•OH", 8.8e9, 7.0, 25.0),
            gold_item("Carbamazepine", "SO4•-", 1.9e9, 7.0, 25.0),
        ],
        "injected_traps": [],
    })

    # --- 34. Beta-blockers: Speciation Physics Violation Trap on Atenolol ---
    excerpts.append({
        "doc_id": "DOC_034",
        "source_citation": "Beta-blocker Speciation Audit Stress Test",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Beta-blockers",
        "text": (
            "Apparent ozonation rate constants of alkanolamine beta-blockers at pH = 7.0 and 20 °C.\n\n"
            "| Compound | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Atenolol | O3 | 1.7 × 10^3 | 3.23 | 7.0 | 9.60 | 1.1 × 10^2 | 6.3 × 10^5 |\n"
            "| Metoprolol | O3 | 9.5 × 10^6 | 6.98 | 7.0 | 9.68 | 3.3 × 10^2 | 8.1 × 10^5 |\n"
        ),
        "gold_records": [
            gold_item("Atenolol", "O3", 1.7e3, 7.0, 20.0),
        ],
        "injected_traps": [
            trap_item(
                "Metoprolol",
                "Beta-blockers",
                "O3",
                "SPECIATION_VIOLATION",
                "Reported k_app=9.5e6 M^-1 s^-1 exceeds even intrinsic k_A-=8.1e5 (predicted 2.02e3 at pH 7.0)",
                raw_reported_k=9.5e6,
                reported_log10_k=6.98,
            )
        ],
    })

    # --- 35. Phenols: NSAIDs & Benzoic Acid Reference Probes + OCR Glyph ---
    excerpts.append({
        "doc_id": "DOC_035",
        "source_citation": "Packer et al. (2003); Huber et al. (2003) (Ibuprofen & Benzoic acid)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Ozonation and hydroxyl radical probe kinetics at pH = 7.0 and 20 °C.\n\n"
            "| Compound | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Ibuprofen | O3 | 9.6 ± 1.0 | 0.98 | 7.0 | 20.0 |\n"
            "| Ibuprofen | •OH | 7.4 × l0^9 | 9.87 | 7.0 | 20.0 |\n"
            "| Benzoic acid | O3 | 1.2 ± 0.2 | 0.08 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Ibuprofen", "O3", 9.6, 7.0, 20.0),
            gold_item("Ibuprofen", "•OH", 7.4e9, 7.0, 20.0, failure_category="OCR_GLYPH_BASE10"),
            gold_item("Benzoic acid", "O3", 1.2, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 36. Triazines: Invalid Valence SMILES Trap on Atrazine + Clean Prometryn O3 ---
    excerpts.append({
        "doc_id": "DOC_036",
        "source_citation": "Triazine OCSR Valence Stress Test",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Triazines",
        "text": (
            "Ozonation of s-triazines in phosphate buffer at pH = 7.0, 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Atrazine | CCNc1nc(NC(C)(C)(C)C)nc(Cl)(Cl)n1 | O3 | 6.0 | 0.78 | 7.0 | 20.0 |\n"
            "| Prometryn | CSc1nc(NC(C)C)nc(NC(C)C)n1 | O3 | 5.2 × 10^2 | 2.72 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Prometryn", "O3", 520.0, 7.0, 20.0),
        ],
        "injected_traps": [
            trap_item(
                "Atrazine",
                "Triazines",
                "O3",
                "INVALID_VALENCE_HALLUCINATION",
                "Pentavalent aromatic carbon nc(Cl)(Cl)n1 in corrupted Atrazine SMILES",
                raw_reported_k=6.0,
                reported_log10_k=0.78,
            )
        ],
    })

    # --- 37. Amines: OCR Exponent Shift on Diclofenac •OH Repaired via Prose ---
    excerpts.append({
        "doc_id": "DOC_037",
        "source_citation": "Huber et al. (2003) (Diclofenac & Trimethoprim •OH)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Amines",
        "text": (
            "Hydroxyl radical competition kinetics in phosphate buffer at pH = 7.0 and 25 °C.\n\n"
            "| Compound | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- |\n"
            "| Diclofenac | •OH | 7.5 × 10^7 | 9.88 | 7.0 | 25.0 |\n"
            "| Trimethoprim | •OH | 6.9 × 10^9 | 9.84 | 7.0 | 25.0 |\n\n"
            "Diclofenac reacted near the diffusion-controlled regime with hydroxyl radicals, yielding k_{•OH} = 7.5 × 10^9 M^-1 s^-1 (log k = 9.88)."
        ),
        "gold_records": [
            gold_item("Diclofenac", "•OH", 7.5e9, 7.0, 25.0, failure_category="OCR_EXPONENT_REPAIRABLE"),
            gold_item("Trimethoprim", "•OH", 6.9e9, 7.0, 25.0),
        ],
        "injected_traps": [],
    })

    # --- 38. Olefins: Pseudo-First-Order `s^-1` Trap + Valid TCE •OH ---
    excerpts.append({
        "doc_id": "DOC_038",
        "source_citation": "Buxton et al. (1988) (chloroethene •OH + s^-1 trap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Olefins",
        "text": (
            "Hydroxyl radical oxidation of chlorinated ethenes at pH = 7.0 and 25 °C.\n\n"
            "| Compound | Oxidant | k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Trichloroethylene | •OH | 4.2 × 10^9 M^-1 s^-1 | 7.0 | 25.0 |\n"
            "| Tetrachloroethylene | •OH | 2.6 × 10^9 M^-1 s^-1 | 7.0 | 25.0 |\n"
            "| Styrene | •OH | 0.042 s^-1 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Trichloroethylene", "•OH", 4.2e9, 7.0, 25.0),
            gold_item("Tetrachloroethylene", "•OH", 2.6e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Styrene",
                "Olefins",
                "•OH",
                "PSEUDO_FIRST_ORDER_TRAP",
                "Pseudo-first-order decay rate 0.042 s^-1 reported without oxidant concentration",
                raw_reported_k=0.042,
            )
        ],
    })

    # --- 39. Phenols: 4-Nonylphenol & Bisphenol A Clean Ozonation ---
    excerpts.append({
        "doc_id": "DOC_039",
        "source_citation": "Deborde et al. (2005) Environ. Sci. Technol. 39, 6086",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Apparent ozonation rate constants of alkylphenols and bisphenols at pH = 7.0 (phosphate buffer, 20 °C).\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Bisphenol A | CC(C)(c1ccc(O)cc1)c1ccc(O)cc1 | O3 | 1.68 × 10^6 | 6.23 | 7.0 | 20.0 |\n"
            "| 4-Nonylphenol | CCCCCCCCCc1ccc(O)cc1 | O3 | 3.9 × 10^5 | 5.59 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Bisphenol A", "O3", 1.68e6, 7.0, 20.0),
            gold_item("4-Nonylphenol", "O3", 3.9e5, 7.0, 20.0),
        ],
        "injected_traps": [],
    })

    # --- 40. Sulfonamides: OCR Diffusion-Exceeding Table Typo Repaired via Prose ---
    excerpts.append({
        "doc_id": "DOC_040",
        "source_citation": "Mezyk et al. (2007) J. Phys. Chem. A (OCR 10^11 -> 10^9 repair)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Sulfonamides",
        "text": (
            "Pulse radiolysis determination of hydroxyl radical rate constants at pH = 7.0, 25 °C.\n\n"
            "| Compound | Oxidant | k (M^-1 s^-1) | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Sulfamethoxazole | •OH | 5.5 × 10^11 | 7.0 | 25.0 |\n"
            "| Sulfamethazine | •OH | 5.0 × 10^9 | 7.0 | 25.0 |\n\n"
            "Transient absorption growth kinetics yielded k_{•OH} = 5.5 × 10^9 M^-1 s^-1 for Sulfamethoxazole at pH 7.0."
        ),
        "gold_records": [
            gold_item("Sulfamethoxazole", "•OH", 5.5e9, 7.0, 25.0, failure_category="OCR_EXPONENT_REPAIRABLE"),
            gold_item("Sulfamethazine", "•OH", 5.0e9, 7.0, 25.0),
        ],
        "injected_traps": [],
    })

    # --- 41. Amines: pH <-> pKa Swap on Trimethylamine (Repairable) ---
    excerpts.append({
        "doc_id": "DOC_041",
        "source_citation": "Hoigne & Bader (1983) (Trimethylamine pH 8.0 vs pKa 9.80 swap)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Amines",
        "text": (
            "Amine ozonation in borate-phosphate buffer at pH = 8.0 and 20 °C.\n\n"
            "| Compound | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Trimethylamine | O3 | 6.40 × 10^4 | 4.81 | 9.80 | 8.00 | 0.05 | 4.1 × 10^6 |\n\n"
            "At pH = 8.0, Trimethylamine (pKa = 9.80) had an apparent rate constant k_O3 = 6.40 × 10^4 M^-1 s^-1."
        ),
        "gold_records": [
            gold_item("Trimethylamine", "O3", 6.40e4, 8.0, 20.0, failure_category="PH_PKA_SWAP_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 42. Beta-blockers: LaTeX Sulfate Radical Table + OCR Exponent Repair ---
    excerpts.append({
        "doc_id": "DOC_042",
        "source_citation": "Yang et al. (2017) Water Res. (beta-blocker SO4•- LaTeX table)",
        "format_type": "latex_table_and_prose",
        "primary_family": "Beta-blockers",
        "text": (
            "Sulfate radical oxidation of beta-blockers at pH = 7.0 and T = 298.15 K.\n"
            "\\begin{tabular}{l c c c c}\n"
            "\\hline\n"
            "Compound & Oxidant & $k$ ($\\mathrm{M^{-1}\\,s^{-1}}$) & $\\log k$ & pH \\\\\n"
            "\\hline\n"
            "Propranolol & SO4•- & $4.8 \\times 10^{7}$ & 9.68 & 7.0 \\\\\n"
            "Metoprolol & SO4•- & $3.1 \\times 10^{9}$ & 9.49 & 7.0 \\\\\n"
            "\\hline\n"
            "\\end{tabular}\n"
            "Propranolol reacted rapidly with sulfate radicals with k_{SO4•-} = 4.8 × 10^9 M^-1 s^-1 (log k = 9.68) at pH 7.0."
        ),
        "gold_records": [
            gold_item("Propranolol", "SO4•-", 4.8e9, 7.0, 25.0, failure_category="OCR_EXPONENT_REPAIRABLE"),
            gold_item("Metoprolol", "SO4•-", 3.1e9, 7.0, 25.0),
        ],
        "injected_traps": [],
    })

    # --- 43. Triazines: Sulfate Radical Atrazine + Unclosed Parenthesis SMILES Trap ---
    excerpts.append({
        "doc_id": "DOC_043",
        "source_citation": "Manoj et al. (2007) Chemosphere (triazine SO4•- + broken SMILES)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Triazines",
        "text": (
            "Sulfate radical rate constants of triazine herbicides at pH = 7.0, 25 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Atrazine | CCNc1nc(NC(C)C)nc(Cl)n1 | SO4•- | 2.6 × 10^9 | 9.41 | 7.0 | 25.0 |\n"
            "| Terbutryn | CCNc1nc(SC)nc(NC(C)(C)C | O3 | 1.1 × 10^3 | 3.04 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Atrazine", "SO4•-", 2.6e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Terbutryn",
                "Triazines",
                "O3",
                "INVALID_VALENCE_HALLUCINATION",
                "Truncated SMILES with unbalanced parentheses and unclosed ring",
                raw_reported_k=1100.0,
                reported_log10_k=3.04,
            )
        ],
    })

    # --- 44. Amines: Speciation Violation Trap on Aniline + Valid Ciprofloxacin •OH ---
    excerpts.append({
        "doc_id": "DOC_044",
        "source_citation": "Amine Speciation Physics Audit Excerpt",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Amines",
        "text": (
            "Aqueous oxidation of aromatic and piperazinyl amines at pH = 7.0, 25 °C.\n\n"
            "| Compound | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Aniline | O3 | 1.2 × 10^3 | 3.08 | 7.0 | 4.60 | 60 | 9.0 × 10^7 |\n"
            "| Ciprofloxacin | •OH | 4.1 × 10^9 | 9.61 | 7.0 | 6.09 | - | - |\n"
        ),
        "gold_records": [
            gold_item("Ciprofloxacin", "•OH", 4.1e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Aniline",
                "Amines",
                "O3",
                "SPECIATION_VIOLATION",
                "At pH 7.0 >> pKa 4.60, Aniline is 99.6% neutral B (k_B=9.0e7), so k_app=1.2e3 is physically impossible",
                raw_reported_k=1200.0,
                reported_log10_k=3.08,
            )
        ],
    })

    # --- 45. Olefins: Diffusion Limit Trap on Styrene + Valid Sorbic Acid •OH ---
    excerpts.append({
        "doc_id": "DOC_045",
        "source_citation": "Olefin Radical Diffusion Limit Benchmark",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Olefins",
        "text": (
            "Hydroxyl radical addition to conjugated double bonds at pH = 7.0, 25 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Sorbic acid | CC=CC=CC(=O)O | •OH | 6.8 × 10^9 | 9.83 | 7.0 | 25.0 |\n"
            "| Styrene | C=Cc1ccccc1 | •OH | 5.9 × 10^11 | 11.77 | 7.0 | 25.0 |\n"
        ),
        "gold_records": [
            gold_item("Sorbic acid", "•OH", 6.8e9, 7.0, 25.0),
        ],
        "injected_traps": [
            trap_item(
                "Styrene",
                "Olefins",
                "•OH",
                "DIFFUSION_LIMIT_VIOLATION",
                "k = 5.9e11 M^-1 s^-1 exceeds Smoluchowski diffusion limit 2.5e10 M^-1 s^-1",
                raw_reported_k=5.9e11,
                reported_log10_k=11.77,
            )
        ],
    })

    # --- 46. Phenols: Salicylic Acid & Acetaminophen O3 Prose-Supplemented Hybrid ---
    excerpts.append({
        "doc_id": "DOC_046",
        "source_citation": "Real et al. (2009); Hoigne & Bader (1983)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Phenols",
        "text": (
            "Ozonation of analgesic phenols in phosphate buffer at pH = 7.0 and 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Acetaminophen | CC(=O)Nc1ccc(O)cc1 | O3 | 1.4 × 10^6 | 6.15 | 7.0 | 20.0 |\n\n"
            "Under the same pH 7.0 buffer conditions at 20 °C, Salicylic acid reacted more slowly with ozone with k_O3 = 2.8 × 10^4 M^-1 s^-1 (log k = 4.45)."
        ),
        "gold_records": [
            gold_item("Acetaminophen", "O3", 1.4e6, 7.0, 20.0),
            gold_item("Salicylic acid", "O3", 2.8e4, 7.0, 20.0, failure_category="HYBRID_PROSE_SUPPLEMENT"),
        ],
        "injected_traps": [],
    })

    # --- 47. Sulfonamides:OCR Exponent Shift on Sulfathiazole O3 Repaired via Prose ---
    excerpts.append({
        "doc_id": "DOC_047",
        "source_citation": "Garoma et al. (2010) Ozone Sci. Eng. (Sulfathiazole OCR repair)",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Sulfonamides",
        "text": (
            "Ozonation of thiazole-ring sulfonamides at pH = 7.0 (phosphate buffer) and 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k_O3 (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Sulfathiazole | Nc1ccc(S(=O)(=O)Nc2nccs2)cc1 | O3 | 1.2 × 10^4 | 6.08 | 7.0 | 20.0 |\n\n"
            "Sulfathiazole underwent rapid direct ozonation at pH 7.0 with k_O3 = 1.2 × 10^6 M^-1 s^-1 (log k = 6.08)."
        ),
        "gold_records": [
            gold_item("Sulfathiazole", "O3", 1.2e6, 7.0, 20.0, failure_category="OCR_EXPONENT_REPAIRABLE"),
        ],
        "injected_traps": [],
    })

    # --- 48. Multi-Family Ambiguous Multi-pH Prose Benchmark (Realistic Hard Edge Case) ---
    # Contains both clean Bisoprolol O3 in table and a complex comparative prose clause
    # mentioning Nitrobenzene •OH alongside a second unlabelled pH reference to test realistic edge behavior.
    excerpts.append({
        "doc_id": "DOC_048",
        "source_citation": "Lee & von Gunten (2012) Water Res. 46, 6177-6195",
        "format_type": "markdown_table_and_prose",
        "primary_family": "Beta-blockers",
        "text": (
            "Comparative abatement of trace organic contaminants during ozonation at pH = 7.0 and 20 °C.\n\n"
            "| Compound | SMILES | Oxidant | k (M^-1 s^-1) | log k | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- | --- | --- |\n"
            "| Bisoprolol | CC(C)OCCOc1ccc(COCC(O)CNC(C)C)cc1 | O3 | 2.2 × 10^3 | 3.34 | 7.0 | 20.0 |\n"
            "| Nitrobenzene | O=[N+]([O-])c1ccccc1 | •OH | 3.9 × 10^9 | 9.59 | 7.0 | 20.0 |\n"
        ),
        "gold_records": [
            gold_item("Bisoprolol", "O3", 2.2e3, 7.0, 20.0),
            gold_item("Nitrobenzene", "•OH", 3.9e9, 7.0, 20.0),
        ],
        "injected_traps": [
            # Plus an extra subtle uncorroborated table-prose conflict in an appendix row?
            # Let's include a realistic ambiguous conflict row in DOC_048 so we also test UNRESOLVED_TABLE_PROSE_CONFLICT!
        ],
    })

    # Add an unresolved table-prose contradiction row to DOC_048 to test conflict rejection too!
    excerpts[-1]["text"] += (
        "| Acrolein | C=CC=O | •OH | 7.0 × 10^9 | 9.85 | 7.0 | 20.0 |\n\n"
        "However, re-evaluation of Acrolein with hydroxyl radical gave k_{•OH} = 1.9 × 10^8 M^-1 s^-1 under identical pH 7.0 conditions."
    )
    excerpts[-1]["injected_traps"].append(
        trap_item(
            "Acrolein",
            "Olefins",
            "•OH",
            "UNRESOLVED_TABLE_PROSE_CONFLICT",
            "Table states 7.0e9 M^-1 s^-1 while prose contradicts with 1.9e8 M^-1 s^-1",
            raw_reported_k=7.0e9,
            reported_log10_k=9.85,
        )
    )

    return excerpts


def main() -> None:
    excerpts = build_excerpts()
    total_gold = sum(len(d["gold_records"]) for d in excerpts)
    total_traps = sum(len(d["injected_traps"]) for d in excerpts)

    corpus = {
        "benchmark_name": "ChemLit-Kinetics-Miner Gold Benchmark v1.0",
        "description": (
            "48 environmental oxidation literature excerpts (Markdown tables, LaTeX tabulars, "
            "and scientific OCR/prose) across 6 micropollutant families (Phenols, Sulfonamides, "
            "Beta-blockers, Triazines, Olefins, Amines) with gold canonical RDKit structures, "
            "standardized rate constants (M^-1 s^-1), and injected failure modes."
        ),
        "num_documents": len(excerpts),
        "num_gold_records": total_gold,
        "num_injected_traps": total_traps,
        "documents": excerpts,
    }

    out_path = ROOT / "data" / "benchmark_corpus.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} ({len(excerpts)} docs, {total_gold} gold records, {total_traps} injected traps).")


if __name__ == "__main__":
    main()
