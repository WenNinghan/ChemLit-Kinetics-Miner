"""Hard Cheminformatics (RDKit) and Physical Chemistry Validator.

Provides deterministic scientific guardrails for aqueous oxidation kinetics mining:
1. RDKit structure sanitization, valence/ring hallucination rejection, canonical SMILES & InChIKey.
2. Second-order rate unit normalization (M^-1 s^-1, L mol^-1 s^-1, mM^-1 s^-1, M^-1 min^-1, etc.)
   and logarithmic parity verification |log10(k_norm) - reported log10(k)| <= tau.
3. Smoluchowski aqueous diffusion-controlled physical bound (k <= 2.5e10 M^-1 s^-1) and T > 0 K.
4. Henderson-Hasselbalch acid-base speciation consistency (k_app(pH) = alpha_0*k_HA + alpha_1*k_A-)
   and pH <-> pKa column-swap detection.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors

from .schema import ChemicalValidationResult, SpeciationCheckResult

# Silence RDKit C++ console spam so sanitization diagnostics are captured cleanly
RDLogger.DisableLog("rdApp.*")

AVOGADRO_CONSTANT = 6.02214076e23
SMOLUCHOWSKI_AQUEOUS_LIMIT = 2.5e10  # M^-1 s^-1 at ambient aqueous conditions


@dataclass(frozen=True)
class ReferenceCompound:
    """Reference environmental micropollutant metadata curated from DECAT / literature."""

    name: str
    compound_class: str
    smiles: str
    cas: Optional[str] = None
    pka: Optional[float] = None
    pka2: Optional[float] = None
    aliases: Tuple[str, ...] = ()


# Curated environmental micropollutant registry spanning all 6 benchmark chemical families
REFERENCE_COMPOUNDS: Tuple[ReferenceCompound, ...] = (
    # --- Phenols ---
    ReferenceCompound("Phenol", "Phenols", "Oc1ccccc1", "108-95-2", 9.95, None, ("hydroxybenzene",)),
    ReferenceCompound("4-Chlorophenol", "Phenols", "Oc1ccc(Cl)cc1", "106-48-9", 9.41, None, ("p-chlorophenol",)),
    ReferenceCompound("2,4-Dichlorophenol", "Phenols", "Oc1ccc(Cl)cc1Cl", "120-83-2", 7.89, None, ("2,4-dcp",)),
    ReferenceCompound("2,4,6-Trichlorophenol", "Phenols", "Oc1c(Cl)cc(Cl)cc1Cl", "88-06-2", 6.15, None, ("2,4,6-tcp",)),
    ReferenceCompound(
        "Pentachlorophenol", "Phenols", "Oc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl", "87-86-5", 4.70, None, ("pcp",)
    ),
    ReferenceCompound(
        "Bisphenol A", "Phenols", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", "80-05-7", 9.60, 10.20, ("bpa",)
    ),
    ReferenceCompound(
        "Triclosan", "Phenols", "Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl", "3380-34-5", 7.90, None, ("tcs",)
    ),
    ReferenceCompound(
        "4-Nitrophenol", "Phenols", "O=[N+]([O-])c1ccc(O)cc1", "100-02-7", 7.15, None, ("p-nitrophenol",)
    ),
    ReferenceCompound("p-Cresol", "Phenols", "Cc1ccc(O)cc1", "106-44-5", 10.26, None, ("4-methylphenol",)),
    ReferenceCompound(
        "Acetaminophen", "Phenols", "CC(=O)Nc1ccc(O)cc1", "103-90-2", 9.38, None, ("paracetamol",)
    ),
    ReferenceCompound(
        "17alpha-Ethinylestradiol",
        "Phenols",
        "C#C[C@]1(O)CC[C@H]2[C@@H]3CCc4cc(O)ccc4[C@H]3CC[C@@]21C",
        "57-63-6",
        10.40,
        None,
        ("ee2", "ethinylestradiol", "17α-ethinylestradiol"),
    ),
    ReferenceCompound(
        "Estrone",
        "Phenols",
        "C[C@]12CC[C@@H]3c4ccc(O)cc4CC[C@H]3[C@@H]1CCC2=O",
        "53-16-7",
        10.34,
        None,
        ("e1",),
    ),
    ReferenceCompound("Guaiacol", "Phenols", "COc1ccccc1O", "90-05-1", 9.98, None, ("2-methoxyphenol",)),
    ReferenceCompound("4-Nonylphenol", "Phenols", "CCCCCCCCCc1ccc(O)cc1", "104-40-5", 10.28, None, ("nonylphenol",)),
    ReferenceCompound("Salicylic acid", "Phenols", "O=C(O)c1ccccc1O", "69-72-7", 2.97, 13.40, ()),
    # --- Sulfonamides ---
    ReferenceCompound(
        "Sulfamethoxazole",
        "Sulfonamides",
        "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1",
        "723-46-6",
        5.70,
        None,
        ("smx",),
    ),
    ReferenceCompound(
        "Sulfadiazine",
        "Sulfonamides",
        "Nc1ccc(S(=O)(=O)Nc2ncccn2)cc1",
        "68-35-9",
        6.48,
        None,
        ("sdz",),
    ),
    ReferenceCompound(
        "Sulfamethazine",
        "Sulfonamides",
        "Cc1cc(C)nc(NS(=O)(=O)c2ccc(N)cc2)n1",
        "57-68-1",
        7.40,
        None,
        ("smz", "sulfadimidine"),
    ),
    ReferenceCompound(
        "Sulfapyridine",
        "Sulfonamides",
        "Nc1ccc(S(=O)(=O)Nc2ccccn2)cc1",
        "144-83-2",
        8.43,
        None,
        ("spy",),
    ),
    ReferenceCompound(
        "Sulfachloropyridazine",
        "Sulfonamides",
        "Nc1ccc(S(=O)(=O)Nc2ccc(Cl)nn2)cc1",
        "80-32-0",
        5.90,
        None,
        ("scp",),
    ),
    ReferenceCompound(
        "Sulfathiazole",
        "Sulfonamides",
        "Nc1ccc(S(=O)(=O)Nc2nccs2)cc1",
        "72-14-0",
        7.20,
        None,
        ("stz",),
    ),
    ReferenceCompound(
        "Sulfamerazine",
        "Sulfonamides",
        "Cc1ccnc(NS(=O)(=O)c2ccc(N)cc2)n1",
        "127-79-7",
        6.98,
        None,
        ("smr",),
    ),
    # --- Beta-blockers ---
    ReferenceCompound(
        "Atenolol",
        "Beta-blockers",
        "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",
        "29122-68-7",
        9.60,
        None,
        ("atl",),
    ),
    ReferenceCompound(
        "Metoprolol",
        "Beta-blockers",
        "COCCc1ccc(OCC(O)CNC(C)C)cc1",
        "51384-51-1",
        9.68,
        None,
        ("mtp",),
    ),
    ReferenceCompound(
        "Propranolol",
        "Beta-blockers",
        "CC(C)NCC(O)COc1cccc2ccccc12",
        "525-66-6",
        9.45,
        None,
        ("prp",),
    ),
    ReferenceCompound(
        "Nadolol",
        "Beta-blockers",
        "CC(C)(C)NCC(O)COc1cccc2c1CC(O)C(O)C2",
        "42200-33-9",
        9.67,
        None,
        (),
    ),
    ReferenceCompound(
        "Sotalol",
        "Beta-blockers",
        "CC(C)NCC(O)c1ccc(NS(C)(=O)=O)cc1",
        "3930-20-9",
        8.30,
        9.80,
        (),
    ),
    ReferenceCompound(
        "Acebutolol",
        "Beta-blockers",
        "CCCC(=O)Nc1ccc(OCC(O)CNC(C)C)c(C(C)=O)c1",
        "37517-30-9",
        9.20,
        None,
        (),
    ),
    ReferenceCompound(
        "Bisoprolol",
        "Beta-blockers",
        "CC(C)OCCOc1ccc(COCC(O)CNC(C)C)cc1",
        "66722-44-9",
        9.50,
        None,
        (),
    ),
    # --- Triazines ---
    ReferenceCompound(
        "Atrazine",
        "Triazines",
        "CCNc1nc(NC(C)C)nc(Cl)n1",
        "1912-24-9",
        1.70,
        None,
        ("atz",),
    ),
    ReferenceCompound(
        "Simazine",
        "Triazines",
        "CCNc1nc(NCC)nc(Cl)n1",
        "122-34-9",
        1.62,
        None,
        ("smz-triazine",),
    ),
    ReferenceCompound(
        "Cyanazine",
        "Triazines",
        "CCNc1nc(NC(C)(C)C#N)nc(Cl)n1",
        "21725-46-2",
        1.10,
        None,
        (),
    ),
    ReferenceCompound(
        "Propazine",
        "Triazines",
        "CC(C)Nc1nc(NC(C)C)nc(Cl)n1",
        "139-40-2",
        1.70,
        None,
        (),
    ),
    ReferenceCompound(
        "Prometryn",
        "Triazines",
        "CSc1nc(NC(C)C)nc(NC(C)C)n1",
        "7287-19-6",
        4.05,
        None,
        (),
    ),
    ReferenceCompound(
        "Terbutryn",
        "Triazines",
        "CCNc1nc(SC)nc(NC(C)(C)C)n1",
        "886-50-0",
        4.30,
        None,
        (),
    ),
    ReferenceCompound(
        "Desethylatrazine",
        "Triazines",
        "CC(C)Nc1nc(N)nc(Cl)n1",
        "6190-65-4",
        1.50,
        None,
        ("dea",),
    ),
    # --- Olefins ---
    ReferenceCompound(
        "Carbamazepine",
        "Olefins",
        "NC(=O)N1c2ccccc2C=Cc2ccccc21",
        "298-46-4",
        None,
        None,
        ("cbz",),
    ),
    ReferenceCompound(
        "Trichloroethylene",
        "Olefins",
        "ClC=C(Cl)Cl",
        "79-01-6",
        None,
        None,
        ("tce", "trichloroethene"),
    ),
    ReferenceCompound(
        "Tetrachloroethylene",
        "Olefins",
        "ClC(Cl)=C(Cl)Cl",
        "127-18-4",
        None,
        None,
        ("pce", "perchloroethylene", "tetrachloroethene"),
    ),
    ReferenceCompound(
        "trans-Cinnamic acid",
        "Olefins",
        "O=C(O)/C=C/c1ccccc1",
        "140-10-3",
        4.44,
        None,
        ("cinnamic acid",),
    ),
    ReferenceCompound(
        "Sorbic acid",
        "Olefins",
        "CC=CC=CC(=O)O",
        "110-44-1",
        4.76,
        None,
        ("2,4-hexadienoic acid",),
    ),
    ReferenceCompound("Styrene", "Olefins", "C=Cc1ccccc1", "100-42-5", None, None, ("vinylbenzene",)),
    ReferenceCompound(
        "cis-1,2-Dichloroethylene",
        "Olefins",
        r"Cl/C=C\Cl",
        "156-59-2",
        None,
        None,
        ("cis-dce", "cis-1,2-dichloroethene"),
    ),
    ReferenceCompound("Acrolein", "Olefins", "C=CC=O", "107-02-8", None, None, ("2-propenal",)),
    # --- Amines ---
    ReferenceCompound("Trimethylamine", "Amines", "CN(C)C", "75-50-3", 9.80, None, ("tma",)),
    ReferenceCompound("Triethylamine", "Amines", "CCN(CC)CC", "121-44-8", 10.75, None, ("tea",)),
    ReferenceCompound("Dimethylamine", "Amines", "CNC", "124-40-3", 10.73, None, ("dma",)),
    ReferenceCompound("Aniline", "Amines", "Nc1ccccc1", "62-53-3", 4.60, None, ("phenylamine",)),
    ReferenceCompound(
        "Ciprofloxacin",
        "Amines",
        "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O",
        "85721-33-1",
        6.09,
        8.74,
        ("cip",),
    ),
    ReferenceCompound(
        "Diclofenac",
        "Amines",
        "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl",
        "15307-86-5",
        4.15,
        None,
        ("dcf",),
    ),
    ReferenceCompound(
        "Trimethoprim",
        "Amines",
        "COc1cc(Cc2cnc(N)nc2N)cc(OC)c1OC",
        "738-70-5",
        7.12,
        None,
        ("tmp",),
    ),
    ReferenceCompound(
        "Diphenhydramine",
        "Amines",
        "CN(C)CCOC(c1ccccc1)c1ccccc1",
        "58-73-1",
        8.98,
        None,
        ("dph",),
    ),
    ReferenceCompound(
        "N-Nitrosodimethylamine",
        "Amines",
        "CN(C)N=O",
        "62-75-9",
        None,
        None,
        ("ndma",),
    ),
    # Additional benchmark reference probes
    ReferenceCompound("Nitrobenzene", "Olefins", "O=[N+]([O-])c1ccccc1", "98-95-3", None, None, ("nb",)),
    ReferenceCompound("Benzoic acid", "Phenols", "O=C(O)c1ccccc1", "65-85-0", 4.20, None, ("benzoate",)),
    ReferenceCompound("Ibuprofen", "Phenols", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "15687-27-1", 4.91, None, ("ibp",)),
)


def _normalize_key(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name.strip().lower())
    cleaned = re.sub(r"^[\(\[\s]+|[\)\]\s]+$", "", cleaned)
    return cleaned


_COMPOUND_LOOKUP: Dict[str, ReferenceCompound] = {}
for _comp in REFERENCE_COMPOUNDS:
    _COMPOUND_LOOKUP[_normalize_key(_comp.name)] = _comp
    if _comp.cas:
        _COMPOUND_LOOKUP[_normalize_key(_comp.cas)] = _comp
    for _alias in _comp.aliases:
        _COMPOUND_LOOKUP[_normalize_key(_alias)] = _comp


def lookup_reference_compound(identifier: Optional[str]) -> Optional[ReferenceCompound]:
    """Look up reference compound metadata by common name, alias, or CAS number."""
    if not identifier:
        return None
    key = _normalize_key(identifier)
    if key in _COMPOUND_LOOKUP:
        return _COMPOUND_LOOKUP[key]
    # Strip trailing parenthetical qualifiers e.g. "Triclosan (TCS)" or "Phenol [HA]"
    stripped = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]\s*$", "", key).strip()
    if stripped in _COMPOUND_LOOKUP:
        return _COMPOUND_LOOKUP[stripped]
    return None


def _diagnose_smiles_syntax_failure(raw_smiles: str) -> Tuple[str, str]:
    """Classify why Chem.MolFromSmiles(..., sanitize=False) returned None."""
    if raw_smiles.count("(") != raw_smiles.count(")"):
        return ("UNBALANCED_PARENTHESES", f"Unbalanced branch parentheses in SMILES '{raw_smiles}'.")
    if raw_smiles.count("[") != raw_smiles.count("]"):
        return ("UNBALANCED_BRACKETS", f"Unbalanced atom brackets in SMILES '{raw_smiles}'.")
    # Count single-digit ring closure tokens outside %nn
    cleaned = re.sub(r"\[[^\]]*\]", "", raw_smiles)
    ring_digits = re.findall(r"(?<!%)\d", cleaned)
    for d in set(ring_digits):
        if ring_digits.count(d) % 2 != 0:
            return (
                "UNCLOSED_RING",
                f"Broken/unclosed ring closure index '{d}' in SMILES '{raw_smiles}'.",
            )
    return ("INVALID_SMILES_SYNTAX", f"RDKit SMILES lexer failed to parse '{raw_smiles}'.")


def validate_chemical_structure(
    compound_name: str,
    raw_smiles: Optional[str] = None,
    allow_lexicon_fallback_on_missing: bool = True,
) -> Tuple[ChemicalValidationResult, Optional[ReferenceCompound]]:
    """Validate raw SMILES or resolve compound name via RDKit, catching valence & ring errors."""
    ref = lookup_reference_compound(compound_name)
    candidate_smiles = raw_smiles.strip() if raw_smiles and raw_smiles.strip() else None

    if candidate_smiles is None and allow_lexicon_fallback_on_missing and ref is not None:
        candidate_smiles = ref.smiles

    if candidate_smiles is None:
        return (
            ChemicalValidationResult(
                input_identifier=compound_name,
                input_smiles=raw_smiles,
                is_valid_structure=False,
                valence_or_sanitization_error=(
                    f"No SMILES provided and compound '{compound_name}' is not in verified lexicon."
                ),
                error_codes=["UNKNOWN_UNGROUNDED_COMPOUND"],
                validation_warnings=[f"Ungrounded compound identifier: {compound_name}"],
            ),
            ref,
        )

    mol = Chem.MolFromSmiles(candidate_smiles, sanitize=False)
    if mol is None:
        err_code, err_msg = _diagnose_smiles_syntax_failure(candidate_smiles)
        return (
            ChemicalValidationResult(
                input_identifier=compound_name,
                input_smiles=candidate_smiles,
                is_valid_structure=False,
                valence_or_sanitization_error=err_msg,
                error_codes=[err_code],
                validation_warnings=[err_msg],
            ),
            ref,
        )

    problems = Chem.DetectChemistryProblems(mol)
    if problems:
        messages = [f"{p.GetType()}: {p.Message()}" for p in problems]
        joined = "; ".join(messages)
        codes: List[str] = []
        for p in problems:
            ptype = p.GetType()
            if "Valence" in ptype:
                codes.append("INVALID_VALENCE")
            elif "Kekulize" in ptype or "Aromatic" in ptype:
                codes.append("AROMATICITY_ERROR")
            else:
                codes.append("RDKIT_SANITIZATION_ERROR")
        return (
            ChemicalValidationResult(
                input_identifier=compound_name,
                input_smiles=candidate_smiles,
                is_valid_structure=False,
                valence_or_sanitization_error=joined,
                error_codes=sorted(set(codes)),
                validation_warnings=[joined],
            ),
            ref,
        )

    sanitize_flag = Chem.SanitizeMol(mol, catchErrors=True)
    if sanitize_flag != Chem.SanitizeFlags.SANITIZE_NONE:
        msg = f"RDKit sanitization failed with bitmask {sanitize_flag} for '{candidate_smiles}'."
        return (
            ChemicalValidationResult(
                input_identifier=compound_name,
                input_smiles=candidate_smiles,
                is_valid_structure=False,
                valence_or_sanitization_error=msg,
                error_codes=["RDKIT_SANITIZATION_ERROR"],
                validation_warnings=[msg],
            ),
            ref,
        )

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    inchikey = Chem.MolToInchiKey(mol)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    exact_mw = round(float(Descriptors.ExactMolWt(mol)), 6)
    formal_charge = int(Chem.GetFormalCharge(mol))

    warnings: List[str] = []
    codes = []

    # Cross-check connectivity layer if both explicit SMILES and known compound name exist
    if raw_smiles and ref is not None:
        ref_mol = Chem.MolFromSmiles(ref.smiles)
        if ref_mol is not None:
            ref_ik = Chem.MolToInchiKey(ref_mol)
            if inchikey[:14] != ref_ik[:14]:
                # Allow benign protonation/deprotonation ONLY if connectivity skeleton matches;
                # differing first 14 chars of InChIKey means a completely different molecular skeleton!
                msg = (
                    f"Name-SMILES skeleton mismatch: '{compound_name}' expects InChIKey connectivity "
                    f"{ref_ik[:14]}, but SMILES '{raw_smiles}' gives {inchikey[:14]}."
                )
                return (
                    ChemicalValidationResult(
                        input_identifier=compound_name,
                        input_smiles=candidate_smiles,
                        canonical_smiles=canonical_smiles,
                        inchikey=inchikey,
                        molecular_formula=formula,
                        exact_mw=exact_mw,
                        formal_charge=formal_charge,
                        is_valid_structure=False,
                        valence_or_sanitization_error=msg,
                        error_codes=["NAME_STRUCTURE_MISMATCH"],
                        validation_warnings=[msg],
                    ),
                    ref,
                )

    return (
        ChemicalValidationResult(
            input_identifier=compound_name,
            input_smiles=candidate_smiles,
            canonical_smiles=canonical_smiles,
            inchikey=inchikey,
            molecular_formula=formula,
            exact_mw=exact_mw,
            formal_charge=formal_charge,
            is_valid_structure=True,
            error_codes=codes,
            validation_warnings=warnings,
        ),
        ref,
    )


def normalize_rate_units(raw_k: float, raw_units: str) -> Tuple[float, float, bool, Optional[str]]:
    """Normalize rate constant units to canonical M^-1 s^-1 (L mol^-1 s^-1).

    Returns:
        (standardized_k, conversion_factor, is_valid_second_order_unit, warning_or_error_code)
    """
    u = raw_units.strip()
    # Normalize Unicode superscripts, dots, spaces, and LaTeX formatting
    sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
    u_norm = u.translate(sup_map)
    u_norm = (
        u_norm.replace("\\cdot", " ")
        .replace("\\,", " ")
        .replace("\\;", " ")
        .replace("\\!", "")
        .replace("·", " ")
        .replace("*", " ")
    )
    u_norm = re.sub(r"\\text\{([^}]*)\}", r"\1", u_norm)
    u_norm = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", u_norm)
    u_norm = u_norm.replace("{", "").replace("}", "").replace("$", "")
    u_norm = re.sub(r"\s+", " ", u_norm).strip()

    compact = u_norm.replace(" ", "")
    compact_lower = compact.lower()

    # Canonical M^-1 s^-1 equivalents
    if compact_lower in {
        "m^-1s^-1",
        "m-1s-1",
        "1/(ms)",
        "lmol^-1s^-1",
        "lmol-1s-1",
        "mol^-1ls^-1",
        "mol-1ls-1",
        "dm^3mol^-1s^-1",
        "dm3mol-1s-1",
        "l/(mols)",
    }:
        return raw_k * 1.0, 1.0, True, None

    # Millimolar inverse: 1 mM^-1 s^-1 = (10^-3 M)^-1 s^-1 = 10^3 M^-1 s^-1
    if compact_lower in {
        "mm^-1s^-1",
        "mm-1s-1",
        "lmmol^-1s^-1",
        "lmmol-1s-1",
        "mmol^-1ls^-1",
        "mmol-1ls-1",
    }:
        return raw_k * 1.0e3, 1.0e3, True, None

    # Micromolar inverse: 1 uM^-1 s^-1 = 10^6 M^-1 s^-1
    if compact_lower in {
        "um^-1s^-1",
        "um-1s-1",
        "μm^-1s^-1",
        "μm-1s-1",
        "lumol^-1s^-1",
        "lμmol^-1s^-1",
    }:
        return raw_k * 1.0e6, 1.0e6, True, None

    # Per minute second-order: 1 M^-1 min^-1 = (1/60) M^-1 s^-1
    if compact_lower in {
        "m^-1min^-1",
        "m-1min-1",
        "lmol^-1min^-1",
        "lmol-1min-1",
        "mol^-1lmin^-1",
    }:
        factor = 1.0 / 60.0
        return raw_k * factor, factor, True, None

    # mM^-1 min^-1: 1000 / 60 M^-1 s^-1
    if compact_lower in {
        "mm^-1min^-1",
        "mm-1min-1",
        "lmmol^-1min^-1",
        "lmmol-1min-1",
    }:
        factor = 1000.0 / 60.0
        return raw_k * factor, factor, True, None

    # Per hour second-order: 1 M^-1 h^-1 = (1/3600) M^-1 s^-1
    if compact_lower in {"m^-1h^-1", "m-1h-1", "lmol^-1h^-1", "m^-1hr^-1"}:
        factor = 1.0 / 3600.0
        return raw_k * factor, factor, True, None

    # Molecular gas/aqueous convention: cm^3 molecule^-1 s^-1 -> M^-1 s^-1
    if compact_lower in {
        "cm^3molecule^-1s^-1",
        "cm3molecule-1s-1",
        "cm^3molc^-1s^-1",
    }:
        factor = 1.0e-3 * AVOGADRO_CONSTANT
        return raw_k * factor, factor, True, None

    # Pseudo-first-order rate constant trap: min^-1 or s^-1 without concentration dimension!
    if compact_lower in {"min^-1", "min-1", "1/min"}:
        factor = 1.0 / 60.0
        return raw_k * factor, factor, False, "PSEUDO_FIRST_ORDER_UNIT"

    if compact_lower in {"s^-1", "s-1", "1/s", "sec^-1"}:
        return raw_k * 1.0, 1.0, False, "PSEUDO_FIRST_ORDER_UNIT"

    if compact_lower in {"h^-1", "h-1", "hr^-1"}:
        factor = 1.0 / 3600.0
        return raw_k * factor, factor, False, "PSEUDO_FIRST_ORDER_UNIT"

    return raw_k, 1.0, False, "UNRECOGNIZED_RATE_UNIT"


def verify_speciation_physics(
    ph: float,
    pka: float,
    k_ha: float,
    k_a_minus: float,
    reported_k_app: Optional[float] = None,
    pka2: Optional[float] = None,
    k_a2_minus: Optional[float] = None,
    tolerance_log: float = 0.15,
) -> SpeciationCheckResult:
    """Verify Henderson-Hasselbalch speciation kinetics consistency:
    k_app(pH) = alpha_0(pH)*k_HA + alpha_1(pH)*k_A- (+ alpha_2(pH)*k_A2- if diprotic).
    """
    if pka2 is not None and k_a2_minus is not None:
        p1, p2 = (pka, pka2) if pka <= pka2 else (pka2, pka)
        h = 10.0 ** (-ph)
        ka1 = 10.0 ** (-p1)
        ka2 = 10.0 ** (-p2)
        denom = h * h + ka1 * h + ka1 * ka2
        alpha_0 = (h * h) / denom
        alpha_1 = (ka1 * h) / denom
        alpha_2: Optional[float] = (ka1 * ka2) / denom
        predicted = alpha_0 * k_ha + alpha_1 * k_a_minus + alpha_2 * k_a2_minus
    else:
        alpha_1 = 1.0 / (1.0 + 10.0 ** (pka - ph))
        alpha_0 = 1.0 - alpha_1
        alpha_2 = None
        predicted = alpha_0 * k_ha + alpha_1 * k_a_minus

    discrepancy: Optional[float] = None
    swapped_detected = False
    is_consistent = True
    notes: List[str] = []

    if reported_k_app is not None and reported_k_app > 0 and predicted > 0:
        discrepancy = abs(math.log10(reported_k_app) - math.log10(predicted))
        if discrepancy > tolerance_log:
            is_consistent = False
            # Test whether pH <-> pKa column swap explains the discrepancy
            if pka2 is None and 0.0 <= pka <= 14.0:
                alpha_1_swap = 1.0 / (1.0 + 10.0 ** (ph - pka))
                alpha_0_swap = 1.0 - alpha_1_swap
                pred_swap = alpha_0_swap * k_ha + alpha_1_swap * k_a_minus
                if pred_swap > 0:
                    swap_disc = abs(math.log10(reported_k_app) - math.log10(pred_swap))
                    if swap_disc <= tolerance_log:
                        swapped_detected = True
                        notes.append(
                            f"Swapping pH ({ph}) <-> pKa ({pka}) reduces log10 discrepancy "
                            f"from {discrepancy:.2f} to {swap_disc:.2f} (pH/pKa column swap!)."
                        )
            if not swapped_detected:
                notes.append(
                    f"Speciation mismatch at pH={ph:.2f}, pKa={pka:.2f}: "
                    f"predicted k_app={predicted:.3e} vs reported {reported_k_app:.3e} M^-1 s^-1 "
                    f"(|Δlog10 k|={discrepancy:.2f} > {tolerance_log:.2f})."
                )
        else:
            notes.append(
                f"Speciation verified: k_app(pH={ph:.2f}) = {reported_k_app:.3e} matches "
                f"alpha_0*k_HA + alpha_1*k_A- = {predicted:.3e} M^-1 s^-1 (|Δlog10 k|={discrepancy:.3f})."
            )
    else:
        notes.append(
            f"Computed theoretical k_app(pH={ph:.2f}, pKa={pka:.2f}) = {predicted:.3e} M^-1 s^-1 "
            f"(alpha_0={alpha_0:.4f}, alpha_1={alpha_1:.4f})."
        )

    return SpeciationCheckResult(
        ph=ph,
        pka=pka,
        pka2=pka2,
        alpha_0=round(alpha_0, 6),
        alpha_1=round(alpha_1, 6),
        alpha_2=round(alpha_2, 6) if alpha_2 is not None else None,
        k_ha=k_ha,
        k_a_minus=k_a_minus,
        k_a2_minus=k_a2_minus,
        predicted_k_app=predicted,
        reported_k_app=reported_k_app,
        log10_discrepancy=round(discrepancy, 4) if discrepancy is not None else None,
        swapped_ph_pka_detected=swapped_detected,
        is_consistent=is_consistent,
        note=" ".join(notes),
    )


class ChemValidator:
    """Comprehensive RDKit + Unit/Log + Diffusion-Bound + Speciation Physics Validator."""

    def __init__(
        self,
        log_k_tolerance: float = 0.08,
        speciation_log_tolerance: float = 0.15,
        diffusion_limit: float = SMOLUCHOWSKI_AQUEOUS_LIMIT,
    ) -> None:
        self.log_k_tolerance = log_k_tolerance
        self.speciation_log_tolerance = speciation_log_tolerance
        self.diffusion_limit = diffusion_limit

    def validate(
        self,
        compound_name: str,
        k_value: float,
        k_units: str = "M^-1 s^-1",
        raw_smiles: Optional[str] = None,
        reported_log10_k: Optional[float] = None,
        ph: Optional[float] = None,
        pka: Optional[float] = None,
        temperature_c: float = 20.0,
        k_ha: Optional[float] = None,
        k_a_minus: Optional[float] = None,
        pka2: Optional[float] = None,
        k_a2_minus: Optional[float] = None,
        inequality: Optional[str] = None,
    ) -> ChemicalValidationResult:
        """Run full cheminformatics and physical chemistry audit on a candidate kinetics entry."""
        chem_res, ref = validate_chemical_structure(compound_name, raw_smiles=raw_smiles)
        errors = list(chem_res.error_codes)
        warnings = list(chem_res.validation_warnings)

        # 1. Normalize units & check rate positivity
        norm_k, factor, unit_ok, unit_err = normalize_rate_units(k_value, k_units)
        chem_res.unit_conversion_factor = factor
        chem_res.standardized_k_M_inv_s_inv = norm_k
        chem_res.reported_log10_k = reported_log10_k

        if not unit_ok and unit_err:
            chem_res.is_unit_consistent = False
            errors.append(unit_err)
            warnings.append(
                f"Rate unit '{k_units}' is not a valid bimolecular second-order unit ({unit_err})."
            )

        if norm_k <= 0.0 or math.isnan(norm_k) or math.isinf(norm_k):
            chem_res.is_unit_consistent = False
            errors.append("NON_POSITIVE_RATE_CONSTANT")
            warnings.append(f"Rate constant must be strictly positive; got {norm_k}.")
        else:
            recomputed_log = math.log10(norm_k)
            chem_res.recomputed_log10_k = round(recomputed_log, 4)

            # 2. Logarithmic parity cross-check: |log10(k_norm) - reported_log10_k| <= tau
            if reported_log10_k is not None:
                disc = abs(recomputed_log - reported_log10_k)
                chem_res.log_k_discrepancy = round(disc, 4)
                if disc > self.log_k_tolerance:
                    chem_res.is_unit_consistent = False
                    # Identify common integer-decade OCR exponent shifts or mM^-1 unit traps
                    nearest_int = round(abs(recomputed_log - reported_log10_k))
                    if abs(disc - 3.0) <= 0.08:
                        errors.append("UNIT_MILLIMOLAR_TRAP_OR_1000X_SHIFT")
                    elif nearest_int >= 1 and abs(disc - nearest_int) <= 0.08:
                        errors.append("OCR_EXPONENT_MISMATCH")
                    else:
                        errors.append("LOG_K_PARITY_MISMATCH")
                    warnings.append(
                        f"Logarithmic parity check failed: log10(k_norm)={recomputed_log:.2f} vs "
                        f"reported log10(k)={reported_log10_k:.2f} (|Δ|={disc:.2f} > {self.log_k_tolerance})."
                    )

            # 3. Smoluchowski aqueous diffusion-controlled upper bound
            if inequality not in ("<", "<=") and norm_k > self.diffusion_limit:
                chem_res.exceeds_diffusion_limit = True
                errors.append("EXCEEDS_SMOLUCHOWSKI_DIFFUSION_LIMIT")
                warnings.append(
                    f"Standardized k = {norm_k:.2e} M^-1 s^-1 exceeds aqueous diffusion-controlled "
                    f"Smoluchowski limit ({self.diffusion_limit:.2e} M^-1 s^-1)."
                )

        # 4. Thermodynamic & liquid-phase aqueous temperature bounds
        temp_k = temperature_c + 273.15
        if temp_k <= 0.0:
            chem_res.invalid_temperature = True
            errors.append("NEGATIVE_ABSOLUTE_TEMPERATURE")
            warnings.append(
                f"Unphysical negative absolute temperature: T = {temp_k:.2f} K ({temperature_c:.2f} °C <= -273.15 °C)."
            )
        elif temperature_c < -5.0 or temperature_c > 105.0:
            chem_res.invalid_temperature = True
            errors.append("OUT_OF_AQUEOUS_LIQUID_TEMPERATURE_RANGE")
            warnings.append(
                f"Temperature {temperature_c:.1f} °C ({temp_k:.2f} K) outside ambient liquid aqueous kinetic range."
            )

        # 5. pH bounds & pH <-> pKa column swap check
        if ph is not None and (ph < -1.0 or ph > 15.0):
            errors.append("INVALID_PH_RANGE")
            warnings.append(f"Reported pH={ph} outside physically meaningful aqueous pH range [-1, 15].")

        if ph is not None and pka is not None and ref is not None and ref.pka is not None:
            # Check if extracted (pH, pKa) swapped the compound's known pKa into the pH slot!
            dist_normal = abs(pka - ref.pka)
            dist_swapped = abs(ph - ref.pka)
            if dist_normal > 0.65 and dist_swapped <= 0.25 and abs(ph - pka) >= 0.7:
                chem_res.ph_pka_swap_suspected = True
                errors.append("PH_PKA_COLUMN_SWAP")
                warnings.append(
                    f"Suspected pH <-> pKa column swap for {compound_name}: "
                    f"reported pH={ph}, pKa={pka}, whereas reference pKa is {ref.pka:.2f}."
                )

        # 6. Henderson-Hasselbalch speciation physics check
        if ph is not None and pka is not None and k_ha is not None and k_a_minus is not None:
            spec_res = verify_speciation_physics(
                ph=ph,
                pka=pka,
                k_ha=k_ha,
                k_a_minus=k_a_minus,
                reported_k_app=norm_k if norm_k > 0 else None,
                pka2=pka2,
                k_a2_minus=k_a2_minus,
                tolerance_log=self.speciation_log_tolerance,
            )
            chem_res.speciation_check = spec_res
            chem_res.speciation_note = spec_res.note
            if not spec_res.is_consistent:
                chem_res.speciation_consistent = False
                if spec_res.swapped_ph_pka_detected:
                    chem_res.ph_pka_swap_suspected = True
                    if "PH_PKA_COLUMN_SWAP" not in errors:
                        errors.append("PH_PKA_COLUMN_SWAP")
                else:
                    errors.append("SPECIATION_PHYSICS_VIOLATION")
                warnings.append(spec_res.note)

        chem_res.error_codes = list(dict.fromkeys(errors))
        chem_res.validation_warnings = warnings
        return chem_res
