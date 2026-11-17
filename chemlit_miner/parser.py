"""Scientific Markdown, LaTeX, and OCR Literature Parser for Reaction Kinetics.

Handles noisy OCR scientific notations, LaTeX equations, Markdown/LaTeX tables,
table footnotes, and prose kinetics statements while tracking exact character-level
provenance spans (source_text[char_start:char_end] == verbatim_quote).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from .chem_validator import lookup_reference_compound, normalize_rate_units
from .schema import EvidenceSpan, ReactionKineticsRecord

UNICODE_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
OxidantType = Literal["O3", "•OH", "SO4•-", "ClO•"]


@dataclass
class ParsedRateValue:
    """Structured result of parsing a raw scientific rate constant expression."""

    raw_string: str
    k_value: float
    uncertainty: Optional[float] = None
    inequality: Optional[Literal["<", "<=", ">", ">=", "~"]] = None
    units: Optional[str] = None
    derived_from_log_k: bool = False
    reported_log10_k: Optional[float] = None
    ocr_repairs: List[str] = field(default_factory=list)


@dataclass
class ProseMention:
    """A reaction kinetics observation extracted from a prose sentence."""

    compound_name: str
    oxidant: Optional[OxidantType]
    parsed_rate: Optional[ParsedRateValue]
    ph: Optional[float]
    pka: Optional[float]
    temperature_c: Optional[float]
    water_matrix: Optional[str]
    radical_scavenger: Optional[str]
    k_ha: Optional[float]
    k_a_minus: Optional[float]
    reported_log10_k: Optional[float]
    evidence: EvidenceSpan


def normalize_oxidant(raw_ox: str) -> Optional[OxidantType]:
    """Normalize diverse chemical/LaTeX/OCR strings for aqueous oxidants."""
    if not raw_ox:
        return None
    s = raw_ox.strip()
    s_clean = (
        s.replace("$", "")
        .replace("\\text", "")
        .replace("\\mathrm", "")
        .replace("\\bullet", "•")
        .replace("\\cdot", "•")
        .replace("{", "")
        .replace("}", "")
        .strip()
    )
    low = s_clean.lower()
    if any(tok in low for tok in ("so4", "sulfate radical", "sulphate radical")):
        return "SO4•-"
    if re.search(r"(?<![a-z])clo\s*•|(?<![a-z])k_?clo\b|hypochlorite\s+radical|chlorine\s+oxide", low):
        return "ClO•"
    if any(
        tok in low
        for tok in ("•oh", ".oh", "ho•", "oh•", "*oh", "hydroxyl radical", "k_oh", "koh")
    ) or s_clean in ("OH", "HO"):
        return "•OH"
    if any(tok in low for tok in ("o3", "ozone", "ozonation", "ko3", "k_o3")):
        return "O3"
    return None


def _clean_latex_text(text: str) -> str:
    """Strip LaTeX wrappers while keeping chemical names and numeric tokens intact."""
    s = text.strip()
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace("$", "").strip()
    return s


def extract_unit_string(text: str) -> Optional[str]:
    """Extract canonicalized unit substring from a rate expression or column header."""
    s = text.translate(UNICODE_SUPERSCRIPTS)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = (
        s.replace("$", "")
        .replace("\\,", " ")
        .replace("\\;", " ")
        .replace("\\!", "")
        .replace("{", "")
        .replace("}", "")
    )
    s = s.replace("\\cdot", " ").replace("·", " ")

    unit_patterns = [
        r"(cm\^?3\s*molecule\^?-1\s*s\^?-1)",
        r"(dm\^?3\s*mol\^?-1\s*s\^?-1)",
        r"(L\s*(?:mmol|μmol|umol|mol)\^?-1\s*(?:min|s|h|hr)\^?-1)",
        r"((?:mmol|μmol|umol|mol)\^?-1\s*L\s*(?:min|s|h|hr)\^?-1)",
        r"((?:mM|μM|uM|M)\^?-1\s*(?:min|s|h|hr)\^?-1)",
        r"(1\s*/\s*\(\s*M\s*s\s*\))",
        r"(?<![A-Za-z0-9\^])((?:min|s|sec|h|hr)\^?-1)(?![A-Za-z0-9])",
    ]
    for pat in unit_patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            raw_u = re.sub(r"\s+", " ", m.group(1)).strip()
            # Preserve case of M vs mM!
            if re.match(r"^mm", raw_u):
                raw_u = "mM" + raw_u[2:]
            elif re.match(r"^m\b|^m\^|-", raw_u) and raw_u[0] == "m" and not raw_u.lower().startswith("min") and not raw_u.lower().startswith("mol") and not raw_u.lower().startswith("mmol"):
                raw_u = "M" + raw_u[1:]
            return raw_u
    return None


def parse_scientific_rate_string(
    raw_text: str,
    default_units: Optional[str] = None,
    enable_ocr_repair: bool = True,
) -> Optional[ParsedRateValue]:
    """Parse messy scientific rate constant strings from Markdown, LaTeX, or OCR output.

    Supports examples:
      - '(3.2 ± 0.4) × 10^5 M^-1 s^-1'
      - '1.8 \\cdot 10^{-2} L mol^{-1} s^{-1}'
      - 'k_O3 < 0.1 M^-1 s^-1'
      - 'log k = 5.51'
      - '4.5 × 10⁵' / '2.1 x l0^6' / '6.3 × 1O^-2'
      - '3.2e5'
    """
    if not raw_text or not raw_text.strip():
        return None

    orig = raw_text.strip()
    ocr_repairs: List[str] = []

    work = orig
    # Convert Unicode superscripts (e.g., 10⁵, 10⁻², M⁻¹ s⁻¹) into caret notation
    def _sup_repl(match: re.Match[str]) -> str:
        sup_chars = match.group(1).translate(UNICODE_SUPERSCRIPTS)
        return f"^{sup_chars}"

    if re.search(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+", work):
        # Differentiate base 10⁵ vs unit M⁻¹
        work_converted = re.sub(r"([⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻]+)", _sup_repl, work)
        if work_converted != work:
            ocr_repairs.append("Normalized Unicode superscripts to caret notation")
            work = work_converted

    # Repair OCR glyph confusions in 10^exp (e.g., l0^5, I0^5, 1O^5, lO^-3)
    if enable_ocr_repair:
        ocr_base10_pat = re.compile(r"(?<![A-Za-z0-9])([lI1][0O])\s*\^\s*\{?\s*([+-]?\d+)\s*\}?")

        def _fix_ocr_10(m: re.Match[str]) -> str:
            token = m.group(1)
            exp = m.group(2)
            if token != "10":
                ocr_repairs.append(f"Repaired OCR base-10 token '{token}^{exp}' -> '10^{exp}'")
            return f"10^{exp}"

        work = ocr_base10_pat.sub(_fix_ocr_10, work)

    # Extract unit before stripping text
    detected_units = extract_unit_string(work) or default_units

    # Clean LaTeX multiplication & plus-minus symbols
    s = work.replace("$", "")
    s = s.replace("\\pm", "±").replace("+/-", "±").replace("+-", "±")
    s = s.replace("\\times", "×").replace("\\cdot", "×").replace("·", "×")
    s = s.replace("\\leq", "<=").replace("\\le", "<=").replace("\\geq", ">=").replace("\\ge", ">=")
    s = s.replace("≤", "<=").replace("≥", ">=").replace("≈", "~").replace("\\approx", "~")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)

    # Check if explicit log k is reported in the same string, e.g. "(log k = 5.51)"
    reported_log_k: Optional[float] = None
    log_m = re.search(
        r"log(?:_?\{?10\}?)?\s*\(?k[^=:\)]*\)?\s*(?:=|:)\s*([+-]?\d+(?:\.\d+)?)",
        s,
        flags=re.IGNORECASE,
    )
    if log_m:
        reported_log_k = float(log_m.group(1))
        # Remove the log k clause from s so we can test if another primary k value exists
        s_without_log = (s[: log_m.start()] + " " + s[log_m.end() :]).strip()
    else:
        s_without_log = s

    # Strip leading variable assignment like "k_O3 =" or "k_{app} =" or "k(•OH) ="
    s_num = re.sub(
        r"^\s*k(?:\s*_?\s*\{?[^=<>~:\(]*\}?|\s*\([^\)]*\))?\s*(?:=|:)?\s*",
        "",
        s_without_log,
        flags=re.IGNORECASE,
    ).strip()

    # Strip trailing footnote markers like ^a, *, (a)
    s_num = re.sub(r"(\^[a-z]|\*+|\([a-z]\))\s*$", "", s_num).strip()

    # Detect inequality prefix (<, <=, >, >=, ~)
    ineq: Optional[Literal["<", "<=", ">", ">=", "~"]] = None
    ineq_m = re.match(r"^(<=|>=|<|>|~)\s*", s_num)
    if ineq_m:
        ineq = ineq_m.group(1)  # type: ignore[assignment]
        s_num = s_num[ineq_m.end() :].strip()

    # Case 1: Parenthesized mantissa with uncertainty and power of 10: (3.2 ± 0.4) × 10^{5}
    pat_paren_sci = re.compile(
        r"^\(\s*([+-]?\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)\s*\)\s*(?:×|x|\*)\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",
        flags=re.IGNORECASE,
    )
    m = pat_paren_sci.search(s_num)
    if m:
        mantissa = float(m.group(1))
        unc_m = float(m.group(2))
        exp = int(m.group(3))
        scale = 10.0**exp
        return ParsedRateValue(
            raw_string=orig,
            k_value=mantissa * scale,
            uncertainty=unc_m * scale,
            inequality=ineq,
            units=detected_units,
            reported_log10_k=reported_log_k,
            ocr_repairs=ocr_repairs,
        )

    # Case 2: Unparenthesized mantissa ± uncertainty × 10^exp: 3.2 ± 0.4 × 10^5
    pat_unc_sci = re.compile(
        r"^([+-]?\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)\s*(?:×|x|\*)\s*10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",
        flags=re.IGNORECASE,
    )
    m = pat_unc_sci.search(s_num)
    if m:
        mantissa = float(m.group(1))
        unc_m = float(m.group(2))
        exp = int(m.group(3))
        scale = 10.0**exp
        return ParsedRateValue(
            raw_string=orig,
            k_value=mantissa * scale,
            uncertainty=unc_m * scale,
            inequality=ineq,
            units=detected_units,
            reported_log10_k=reported_log_k,
            ocr_repairs=ocr_repairs,
        )

    # Case 3: Standard scientific notation mantissa × 10^{exp} or 10^{exp} alone
    pat_sci = re.compile(
        r"^(?:([+-]?\d+(?:\.\d+)?)\s*(?:×|x|\*)\s*)?10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?",
        flags=re.IGNORECASE,
    )
    m = pat_sci.search(s_num)
    if m:
        mantissa = float(m.group(1)) if m.group(1) is not None else 1.0
        exp = int(m.group(2))
        return ParsedRateValue(
            raw_string=orig,
            k_value=mantissa * (10.0**exp),
            uncertainty=None,
            inequality=ineq,
            units=detected_units,
            reported_log10_k=reported_log_k,
            ocr_repairs=ocr_repairs,
        )

    # Case 4: E-notation, e.g. 3.2e5 or (3.2 ± 0.4)e5
    pat_enot = re.compile(r"^([+-]?\d+(?:\.\d+)?)[eE]([+-]?\d+)(?![A-Za-z])")
    m = pat_enot.search(s_num)
    if m:
        val = float(f"{m.group(1)}e{m.group(2)}")
        return ParsedRateValue(
            raw_string=orig,
            k_value=val,
            inequality=ineq,
            units=detected_units,
            reported_log10_k=reported_log_k,
            ocr_repairs=ocr_repairs,
        )

    # Case 5: Plain decimal / integer with optional ± uncertainty (e.g., '6.0 ± 0.5 M^-1 s^-1', '< 0.1')
    # Guard against matching unparsed OCR token like '3.2 × l0^5' when enable_ocr_repair=False!
    if not enable_ocr_repair and re.search(r"(?:×|x|\*)\s*[lI1][0O]\s*\^", s_num):
        return None

    pat_plain = re.compile(r"^\(?\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:±\s*(\d+(?:\.\d+)?))?\s*\)?")
    m = pat_plain.search(s_num)
    if m:
        val_str = m.group(1).replace(",", "")
        val = float(val_str)
        unc = float(m.group(2)) if m.group(2) else None
        return ParsedRateValue(
            raw_string=orig,
            k_value=val,
            uncertainty=unc,
            inequality=ineq,
            units=detected_units,
            reported_log10_k=reported_log_k,
            ocr_repairs=ocr_repairs,
        )

    # Case 6: Only `log k = 5.51` was provided
    if reported_log_k is not None:
        return ParsedRateValue(
            raw_string=orig,
            k_value=10.0**reported_log_k,
            uncertainty=None,
            inequality=ineq,
            units=detected_units or "M^-1 s^-1",
            derived_from_log_k=True,
            reported_log10_k=reported_log_k,
            ocr_repairs=ocr_repairs,
        )

    return None


def parse_temperature(raw_temp: str, header_unit_hint: Optional[str] = None) -> Optional[Tuple[float, float]]:
    """Parse temperature string into (temperature_c, temperature_k).

    Handles '298 K', '293.15 K', '25 °C', '20 ± 1 °C', '25 C', and unphysical '-15 K' (for physics gate).
    """
    if not raw_temp or not raw_temp.strip():
        return None
    s = raw_temp.strip().replace("$", "").replace("\\circ", "°").replace("\\pm", "±")
    m = re.search(
        r"([+-]?\d+(?:\.\d+)?)\s*(?:±\s*\d+(?:\.\d+)?)?\s*(°\s*C|deg\s*C|℃|K|C)\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        val = float(m.group(1))
        unit = m.group(2).upper().replace(" ", "")
        if unit == "K":
            return (round(val - 273.15, 4), round(val, 4))
        return (round(val, 4), round(val + 273.15, 4))

    # Numeric cell with unit in column header
    m_num = re.search(r"^([+-]?\d+(?:\.\d+)?)", s)
    if m_num:
        val = float(m_num.group(1))
        if header_unit_hint and "k" in header_unit_hint.lower() and "c" not in header_unit_hint.lower():
            return (round(val - 273.15, 4), round(val, 4))
        if val > 150.0:  # Clearly Kelvin (e.g., 293.15 or 298)
            return (round(val - 273.15, 4), round(val, 4))
        return (round(val, 4), round(val + 273.15, 4))
    return None


def parse_ph_and_matrix(raw_ph: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse pH value and optional buffer/matrix annotation, e.g. 'pH = 7.0 (phosphate buffer)'."""
    if not raw_ph or not raw_ph.strip():
        return None, None
    s = raw_ph.strip()
    matrix: Optional[str] = None
    paren_m = re.search(r"\(([^\)]+)\)", s)
    if paren_m:
        inside = paren_m.group(1).strip()
        if any(w in inside.lower() for w in ("buffer", "water", "wastewater", "effluent", "matrix", "borate", "phosphate", "carbonate")):
            matrix = inside.lower().replace(" ", "_")

    m = re.search(r"(?:pH\s*(?:=|:)?\s*)?([+-]?\d+(?:\.\d+)?)", s, flags=re.IGNORECASE)
    if m:
        return float(m.group(1)), matrix
    return None, matrix


def _extract_document_defaults(text: str, parse_footnotes: bool = True) -> Dict[str, Any]:
    """Extract document-level experimental defaults (pH, T, matrix, scavenger, footnote units)."""
    defaults: Dict[str, Any] = {
        "ph": None,
        "temperature_c": 20.0,
        "temperature_k": 293.15,
        "temperature_found": False,
        "water_matrix": "phosphate_buffer",
        "radical_scavenger": None,
        "oxidant": None,
        "footnote_units": {},
    }

    # Matrix detection
    matrix_patterns = [
        (r"phosphate[\-\s]+buffered|phosphate\s+buffer", "phosphate_buffer"),
        (r"borate\s+buffer", "borate_buffer"),
        (r"carbonate[\-\s]+free\s+buffer", "carbonate_free_buffer"),
        (r"ultrapure\s+water|milli[\-\s]?q\s+water", "ultrapure_water"),
        (r"surface\s+water|lake\s+water", "surface_water"),
        (r"wastewater\s+effluent", "wastewater_effluent"),
    ]
    for pat, label in matrix_patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            defaults["water_matrix"] = label
            break

    # Scavenger detection
    scav_patterns = [
        (r"tert[\-\s]?butanol|t[\-\s]?buoh|tba\b", "tert-butanol"),
        (r"\bdmso\b|dimethyl\s+sulfoxide", "DMSO"),
        (r"methanol\s+scavenger", "methanol"),
    ]
    for pat, label in scav_patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            defaults["radical_scavenger"] = label
            break

    # Temperature in experimental prose (only scan non-table prose lines)
    prose_only_lines = [
        ln for ln in text.splitlines() if not ln.strip().startswith("|") and "&" not in ln
    ]
    prose_header_blob = "\n".join(prose_only_lines)
    temp_m = re.search(
        r"(?:T\s*=\s*|temperature\s+(?:of|at|was)\s+|at\s+|,\s*|\(\s*)([+-]?\d+(?:\.\d+)?\s*(?:±\s*\d+(?:\.\d+)?)?\s*(?:°\s*C|℃|K))\b",
        prose_header_blob,
    )
    if temp_m:
        parsed_t = parse_temperature(temp_m.group(1))
        if parsed_t:
            defaults["temperature_c"], defaults["temperature_k"] = parsed_t
            defaults["temperature_found"] = True

    # Default pH in experimental prose
    ph_m = re.search(r"\bpH\s*(?:=|of|at)?\s*(\d+(?:\.\d+)?)\b", text)
    if ph_m:
        defaults["ph"] = float(ph_m.group(1))

    # Dominant oxidant in title/header if single-oxidant study
    for ox_candidate in ("SO4•-", "•OH", "O3"):
        pass
    if re.search(r"\bozonation\b|\bozone\b|k_?\{?O_?3\}?", text, flags=re.IGNORECASE) and not re.search(
        r"hydroxyl\s+radical|•OH|\\bullet\s*OH|SO_?4", text, flags=re.IGNORECASE
    ):
        defaults["oxidant"] = "O3"

    # Parse footnote unit qualifiers e.g. "^a Rate constants reported in mM^-1 s^-1." or "* Units: M^-1 min^-1"
    if parse_footnotes:
        for line in text.splitlines():
            ls = line.strip()
            fn_m = re.match(r"^(?:\^?([a-d])|\*(\*?)|Note\s*:)\s*(.+)$", ls, flags=re.IGNORECASE)
            if fn_m:
                marker = fn_m.group(1) or ("**" if fn_m.group(2) else "*")
                body = fn_m.group(3)
                u = extract_unit_string(body)
                if u:
                    defaults["footnote_units"][marker] = u
                    defaults["footnote_units"]["DEFAULT_FOOTNOTE"] = u

    return defaults


def _classify_column(col_header: str) -> Tuple[str, Dict[str, Any]]:
    """Map messy Markdown/LaTeX column headers into canonical semantic slots."""
    clean = _clean_latex_text(col_header)
    low = clean.lower().strip()
    meta: Dict[str, Any] = {"raw_header": col_header}

    u = extract_unit_string(col_header)
    if u:
        meta["units"] = u

    ox = normalize_oxidant(col_header)
    if ox:
        meta["oxidant"] = ox

    if any(k in low for k in ("smiles", "structure")):
        return ("smiles", meta)
    if "cas" in low:
        return ("cas", meta)
    if any(k in low for k in ("compound", "chemical", "micropollutant", "pollutant", "substrate", "analyte", "solute")):
        return ("compound", meta)
    if "class" in low or "family" in low:
        return ("compound_class", meta)
    if low in ("oxidant", "radical", "reactive species"):
        return ("oxidant", meta)
    if "log" in low and "k" in low and "pka" not in low:
        return ("log_k", meta)
    if re.search(r"(?<![a-z])k_?\{?(?:ha|neutral|acid|bh\+|bh|h2a)(?![a-z])", low):
        return ("k_ha", meta)
    if (
        any(tok in low for tok in ("k_a-", "k_a^-", "ka-", "k_anion", "k_base", "k_deprot"))
        or re.search(r"(?<![a-z])k_?\{?a\s*\^?\s*\-", low)
        or "k_{a^-}" in col_header.lower()
    ):
        return ("k_a_minus", meta)
    if "pka2" in low:
        return ("pka2", meta)
    if "pka" in low or "pk_a" in low:
        return ("pka", meta)
    if re.search(r"\bph\b", low):
        return ("ph", meta)
    if any(k in low for k in ("temp", "t (°c)", "t (k)", "t / k", "t / °c")) or re.match(r"^t\b", low):
        meta["temp_hint"] = col_header
        return ("temperature", meta)
    if any(k in low for k in ("matrix", "buffer", "water")):
        return ("matrix", meta)
    if "scavenger" in low:
        return ("scavenger", meta)
    if "k" in low or "rate" in low:
        return ("k", meta)
    return ("other", meta)


class ScientificKineticsParser:
    """Extracts candidate ReactionKineticsRecords and ProseMentions with exact provenance spans."""

    def __init__(
        self,
        enable_ocr_repair: bool = True,
        parse_footnotes: bool = True,
        extract_prose_standalone: bool = True,
    ) -> None:
        self.enable_ocr_repair = enable_ocr_repair
        self.parse_footnotes = parse_footnotes
        self.extract_prose_standalone = extract_prose_standalone

    def parse_document(
        self,
        text: str,
        doc_id: str = "DOC_UNKNOWN",
        page_num: int = 1,
    ) -> Tuple[List[ReactionKineticsRecord], List[ProseMention]]:
        """Parse tables (Markdown & LaTeX) and prose mentions from a literature excerpt."""
        defaults = _extract_document_defaults(text, parse_footnotes=self.parse_footnotes)
        records: List[ReactionKineticsRecord] = []

        # 1. Parse Markdown Pipe Tables
        md_records = self._parse_markdown_tables(text, doc_id, page_num, defaults, start_idx=1)
        records.extend(md_records)

        # 2. Parse LaTeX Tabular Environments
        latex_records = self._parse_latex_tables(
            text, doc_id, page_num, defaults, start_idx=len(records) + 1
        )
        records.extend(latex_records)

        # 3. Parse Prose Mentions
        prose_mentions = self.parse_prose_mentions(text, doc_id, page_num, defaults)

        # 4. If no tables exist in the excerpt (pure prose excerpt) and standalone prose enabled,
        # promote prose mentions to primary candidate ReactionKineticsRecords
        if not records and self.extract_prose_standalone:
            for idx, pm in enumerate(prose_mentions, start=1):
                if pm.parsed_rate is None or pm.oxidant is None:
                    continue
                rec = self._prose_mention_to_record(
                    pm, doc_id=doc_id, rec_idx=idx, defaults=defaults
                )
                records.append(rec)

        return records, prose_mentions

    def _split_md_row(self, line: str) -> List[str]:
        s = line.strip()
        if s.startswith("|"):
            s = s[1:]
        if s.endswith("|"):
            s = s[:-1]
        return [cell.strip() for cell in s.split("|")]

    def _parse_markdown_tables(
        self,
        text: str,
        doc_id: str,
        page_num: int,
        defaults: Dict[str, Any],
        start_idx: int = 1,
    ) -> List[ReactionKineticsRecord]:
        records: List[ReactionKineticsRecord] = []
        lines_with_spans: List[Tuple[int, int, int, str]] = []
        pos = 0
        for line_no, raw_line in enumerate(text.splitlines(keepends=True), start=1):
            stripped_nl = raw_line.rstrip("\r\n")
            c_start = pos
            c_end = pos + len(stripped_nl)
            lines_with_spans.append((line_no, c_start, c_end, stripped_nl))
            pos += len(raw_line)

        i = 0
        table_num = 0
        rec_counter = start_idx
        n = len(lines_with_spans)
        while i < n - 1:
            _, _, _, line_str = lines_with_spans[i]
            _, _, _, next_str = lines_with_spans[i + 1]
            if (
                "|" in line_str
                and line_str.strip().startswith("|")
                and re.match(r"^\s*\|[\s:\-\|]+\|\s*$", next_str)
            ):
                table_num += 1
                headers = self._split_md_row(line_str)
                col_specs = [_classify_column(h) for h in headers]
                i += 2
                row_num = 0
                while i < n:
                    line_no, c_start, c_end, row_str = lines_with_spans[i]
                    if not row_str.strip().startswith("|") or "|" not in row_str:
                        break
                    if re.match(r"^\s*\|[\s:\-\|]+\|\s*$", row_str):
                        i += 1
                        continue
                    row_num += 1
                    cells = self._split_md_row(row_str)
                    row_recs = self._build_records_from_cells(
                        cells=cells,
                        col_specs=col_specs,
                        doc_id=doc_id,
                        page_num=page_num,
                        locator=f"Table {table_num}, Row {row_num} (Line {line_no})",
                        char_start=c_start,
                        char_end=c_end,
                        verbatim_quote=text[c_start:c_end],
                        defaults=defaults,
                        rec_counter_start=rec_counter,
                    )
                    records.extend(row_recs)
                    rec_counter += len(row_recs)
                    i += 1
            else:
                i += 1
        return records

    def _parse_latex_tables(
        self,
        text: str,
        doc_id: str,
        page_num: int,
        defaults: Dict[str, Any],
        start_idx: int = 1,
    ) -> List[ReactionKineticsRecord]:
        records: List[ReactionKineticsRecord] = []
        tab_pat = re.compile(r"\\begin\{tabular\}\{[^\}]*\}(.*?)\\end\{tabular\}", flags=re.DOTALL)
        rec_counter = start_idx

        for tab_idx, tab_match in enumerate(tab_pat.finditer(text), start=1):
            body = tab_match.group(1)
            body_offset = tab_match.start(1)
            # Split rows by \\ while tracking exact character offsets in `text`
            row_spans: List[Tuple[int, int, str]] = []
            cursor = 0
            for sep_m in re.finditer(r"\\\\", body):
                r_start = body_offset + cursor
                r_end = body_offset + sep_m.start()
                row_spans.append((r_start, r_end, text[r_start:r_end]))
                cursor = sep_m.end()
            if cursor < len(body):
                r_start = body_offset + cursor
                r_end = body_offset + len(body)
                tail = text[r_start:r_end]
                if "&" in tail:
                    row_spans.append((r_start, r_end, tail))

            cleaned_rows: List[Tuple[int, int, List[str]]] = []
            for r_start, r_end, raw_r in row_spans:
                # Trim leading whitespace/rule commands while shifting r_start accurately!
                s_local = raw_r
                rule_pat = re.compile(
                    r"^(\s|\\hline|\\toprule|\\midrule|\\bottomrule|\\cline\{[^\}]*\})+"
                )
                lead_m = rule_pat.match(s_local)
                if lead_m:
                    r_start += lead_m.end()
                    s_local = s_local[lead_m.end() :]
                trail_m = re.search(r"(\s|\\hline|\\bottomrule)+$", s_local)
                if trail_m:
                    r_end -= len(trail_m.group(0))
                    s_local = s_local[: trail_m.start()]
                if "&" not in s_local or r_start >= r_end:
                    continue
                cells = [c.strip() for c in s_local.split("&")]
                cleaned_rows.append((r_start, r_end, cells))

            if len(cleaned_rows) < 2:
                continue

            _, _, header_cells = cleaned_rows[0]
            col_specs = [_classify_column(h) for h in header_cells]

            for row_num, (c_start, c_end, cells) in enumerate(cleaned_rows[1:], start=1):
                row_recs = self._build_records_from_cells(
                    cells=cells,
                    col_specs=col_specs,
                    doc_id=doc_id,
                    page_num=page_num,
                    locator=f"LaTeX Table {tab_idx}, Row {row_num}",
                    char_start=c_start,
                    char_end=c_end,
                    verbatim_quote=text[c_start:c_end],
                    defaults=defaults,
                    rec_counter_start=rec_counter,
                )
                records.extend(row_recs)
                rec_counter += len(row_recs)

        return records

    def _build_records_from_cells(
        self,
        cells: List[str],
        col_specs: List[Tuple[str, Dict[str, Any]]],
        doc_id: str,
        page_num: int,
        locator: str,
        char_start: int,
        char_end: int,
        verbatim_quote: str,
        defaults: Dict[str, Any],
        rec_counter_start: int,
    ) -> List[ReactionKineticsRecord]:
        compound_name: Optional[str] = None
        cas_number: Optional[str] = None
        compound_class: Optional[str] = None
        raw_smiles: Optional[str] = None
        row_oxidant: Optional[OxidantType] = None
        reported_log10_k: Optional[float] = None
        ph: Optional[float] = defaults.get("ph")
        pka: Optional[float] = None
        pka2: Optional[float] = None
        k_ha: Optional[float] = None
        k_a_minus: Optional[float] = None
        temperature_c: float = defaults.get("temperature_c", 20.0)
        temperature_k: float = defaults.get("temperature_k", 293.15)
        temp_defaulted: bool = True
        water_matrix: str = defaults.get("water_matrix", "phosphate_buffer")
        scavenger: Optional[str] = defaults.get("radical_scavenger")

        k_columns: List[Tuple[str, Dict[str, Any]]] = []

        for idx, cell in enumerate(cells):
            if idx >= len(col_specs):
                break
            role, meta = col_specs[idx]
            val_clean = _clean_latex_text(cell)
            if not val_clean or val_clean in ("-", "—", "N/A", "n.d.", "ND"):
                continue

            if role == "compound":
                compound_name = re.sub(r"(\^[a-d]|\*+)\s*$", "", val_clean).strip()
            elif role == "cas":
                cas_number = val_clean
            elif role == "compound_class":
                compound_class = val_clean
            elif role == "smiles":
                raw_smiles = val_clean.strip("` ")
            elif role == "oxidant":
                row_oxidant = normalize_oxidant(cell)
            elif role == "log_k":
                m_lk = re.search(r"([+-]?\d+(?:\.\d+)?)", val_clean)
                if m_lk:
                    reported_log10_k = float(m_lk.group(1))
            elif role == "ph":
                ph_val, mat = parse_ph_and_matrix(val_clean)
                if ph_val is not None:
                    ph = ph_val
                if mat:
                    water_matrix = mat
            elif role == "pka":
                m_pka = re.search(r"([+-]?\d+(?:\.\d+)?)", val_clean)
                if m_pka:
                    pka = float(m_pka.group(1))
            elif role == "pka2":
                m_pka2 = re.search(r"([+-]?\d+(?:\.\d+)?)", val_clean)
                if m_pka2:
                    pka2 = float(m_pka2.group(1))
            elif role == "k_ha":
                p_ha = parse_scientific_rate_string(
                    cell, default_units=meta.get("units", "M^-1 s^-1"), enable_ocr_repair=self.enable_ocr_repair
                )
                if p_ha:
                    norm_ha, _, _, _ = normalize_rate_units(p_ha.k_value, p_ha.units or "M^-1 s^-1")
                    k_ha = norm_ha
            elif role == "k_a_minus":
                p_am = parse_scientific_rate_string(
                    cell, default_units=meta.get("units", "M^-1 s^-1"), enable_ocr_repair=self.enable_ocr_repair
                )
                if p_am:
                    norm_am, _, _, _ = normalize_rate_units(p_am.k_value, p_am.units or "M^-1 s^-1")
                    k_a_minus = norm_am
            elif role == "temperature":
                t_parsed = parse_temperature(val_clean, header_unit_hint=meta.get("temp_hint"))
                if t_parsed:
                    temperature_c, temperature_k = t_parsed
                    temp_defaulted = False
            elif role == "matrix":
                water_matrix = val_clean.lower().replace(" ", "_")
            elif role == "scavenger":
                scavenger = val_clean
            elif role == "k":
                k_columns.append((cell, meta))

        if not compound_name or not k_columns:
            return []

        ref = lookup_reference_compound(compound_name)
        resolved_class = compound_class or (ref.compound_class if ref else "Phenols")
        if cas_number is None and ref and ref.cas:
            cas_number = ref.cas

        built: List[ReactionKineticsRecord] = []
        for offset, (raw_k_cell, k_meta) in enumerate(k_columns):
            # Check for footnote unit marker in cell or header
            fn_units: Optional[str] = None
            if self.parse_footnotes and defaults.get("footnote_units"):
                fn_map: Dict[str, str] = defaults["footnote_units"]
                fn_m = re.search(r"\^([a-d])|(\*\*|\*)\s*$", raw_k_cell.strip())
                if fn_m:
                    key = fn_m.group(1) or fn_m.group(2)
                    fn_units = fn_map.get(key) or fn_map.get("DEFAULT_FOOTNOTE")
                elif re.search(r"\^([a-d])|(\*\*|\*)", k_meta.get("raw_header", "")):
                    fn_units = fn_map.get("DEFAULT_FOOTNOTE")

            default_u = fn_units or k_meta.get("units") or "M^-1 s^-1"
            parsed_k = parse_scientific_rate_string(
                raw_k_cell,
                default_units=default_u,
                enable_ocr_repair=self.enable_ocr_repair,
            )
            if parsed_k is None:
                continue

            ox: OxidantType = (
                row_oxidant
                or k_meta.get("oxidant")
                or normalize_oxidant(raw_k_cell)
                or defaults.get("oxidant")
                or "O3"
            )
            units_str = parsed_k.units or default_u
            if self.enable_ocr_repair:
                std_k, _, _, _ = normalize_rate_units(parsed_k.k_value, units_str)
            else:
                # Naive single-pass mode assumes raw k_value is already M^-1 s^-1 unless trivial M^-1 s^-1
                std_k = parsed_k.k_value

            rep_log = reported_log10_k if reported_log10_k is not None else parsed_k.reported_log10_k
            rec_id = f"{doc_id}_R{rec_counter_start + offset:02d}"

            ev = EvidenceSpan(
                doc_id=doc_id,
                page_num=page_num,
                locator=locator,
                char_start=char_start,
                char_end=char_end,
                verbatim_quote=verbatim_quote,
            )

            built.append(
                ReactionKineticsRecord(
                    record_id=rec_id,
                    doc_id=doc_id,
                    compound_name=compound_name,
                    cas_number=cas_number,
                    compound_class=resolved_class,
                    raw_smiles=raw_smiles,
                    oxidant=ox,
                    raw_k_string=raw_k_cell.strip(),
                    inequality=parsed_k.inequality,
                    k_value=parsed_k.k_value,
                    k_uncertainty=parsed_k.uncertainty,
                    k_units=units_str,
                    standardized_k=std_k,
                    reported_log10_k=rep_log,
                    ph=ph,
                    pka=pka if pka is not None else (ref.pka if ref else None),
                    pka2=pka2 if pka2 is not None else (ref.pka2 if ref else None),
                    k_ha=k_ha,
                    k_a_minus=k_a_minus,
                    temperature_c=temperature_c,
                    temperature_k=temperature_k,
                    temperature_defaulted=temp_defaulted,
                    water_matrix=water_matrix,
                    radical_scavenger=scavenger,
                    evidence=ev,
                    metadata={
                        "ocr_repairs": parsed_k.ocr_repairs,
                        "derived_from_log_k": parsed_k.derived_from_log_k,
                        "pka_explicit_in_table": pka is not None,
                    },
                )
            )
        return built

    def parse_prose_mentions(
        self,
        text: str,
        doc_id: str,
        page_num: int = 1,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> List[ProseMention]:
        """Extract compound-specific rate constant statements from prose lines."""
        if defaults is None:
            defaults = _extract_document_defaults(text, parse_footnotes=self.parse_footnotes)

        mentions: List[ProseMention] = []
        in_latex_table = False
        pos = 0

        for line_no, raw_line in enumerate(text.splitlines(keepends=True), start=1):
            stripped_nl = raw_line.rstrip("\r\n")
            line_start = pos
            pos += len(raw_line)

            ls = stripped_nl.strip()
            if "\\begin{tabular}" in ls:
                in_latex_table = True
                continue
            if "\\end{tabular}" in ls:
                in_latex_table = False
                continue
            if in_latex_table or not ls or ls.startswith("|") or ls.startswith("#"):
                continue
            # Skip pure footnote legend lines without compound names
            if re.match(r"^(\^[a-d]|\*+)\s+", ls):
                continue

            # Split line into sentences/clauses without breaking inside decimal numbers (e.g. 7.0, 2.8)
            boundary_pat = re.compile(r"(?:(?<!\b[A-Za-z])[.!?](?=\s+[A-Z0-9\(])|;\s+|,\s+and\s+(?=[A-Z0-9]))")
            seg_starts = [0]
            seg_ends: List[int] = []
            for bm in boundary_pat.finditer(stripped_nl):
                # Keep punctuation mark inside preceding sentence if '.', '!', '?'
                if stripped_nl[bm.start()] in ".!?":
                    seg_ends.append(bm.start() + 1)
                    next_pos = bm.end()
                    while next_pos < len(stripped_nl) and stripped_nl[next_pos].isspace():
                        next_pos += 1
                    seg_starts.append(next_pos)
                else:
                    seg_ends.append(bm.start())
                    seg_starts.append(bm.end())
            seg_ends.append(len(stripped_nl))

            for raw_s_start, raw_s_end in zip(seg_starts, seg_ends):
                sent = stripped_nl[raw_s_start:raw_s_end]
                if not sent.strip():
                    continue
                lead_ws = len(sent) - len(sent.lstrip())
                trail_ws = len(sent) - len(sent.rstrip())
                s_start = line_start + raw_s_start + lead_ws
                s_end = line_start + raw_s_end - trail_ws
                if s_start >= s_end:
                    continue
                sent_clean = text[s_start:s_end]

                found_comp = self._find_compound_in_sentence(sent_clean)
                if not found_comp:
                    continue

                ox = normalize_oxidant(sent_clean) or defaults.get("oxidant")
                parsed_rate = self._extract_rate_from_sentence(sent_clean)
                ph_m = re.search(r"\bpH\s*(?:=|of|at)?\s*([+-]?\d+(?:\.\d+)?)", sent_clean)
                ph_val = float(ph_m.group(1)) if ph_m else defaults.get("ph")

                pka_m = re.search(r"\bpK_?a\s*(?:=|:)?\s*(\d+(?:\.\d+)?)", sent_clean, flags=re.IGNORECASE)
                pka_val = float(pka_m.group(1)) if pka_m else None

                t_m = re.search(
                    r"([+-]?\d+(?:\.\d+)?\s*(?:±\s*\d+(?:\.\d+)?)?\s*(?:°\s*C|℃|K))\b",
                    sent_clean,
                )
                temp_c: Optional[float] = None
                if t_m:
                    pt = parse_temperature(t_m.group(1))
                    if pt:
                        temp_c = pt[0]

                k_ha_val = self._extract_named_sub_k(sent_clean, r"k_?\{?(?:HA|neutral|BH\+)\}?")
                k_am_val = self._extract_named_sub_k(sent_clean, r"k_?\{?(?:A\-|A\^\-|anion|B)\}?")

                logk_m = re.search(
                    r"log(?:_?\{?10\}?)?\s*\(?k[^=:\)]*\)?\s*(?:=|:)\s*([+-]?\d+(?:\.\d+)?)",
                    sent_clean,
                    flags=re.IGNORECASE,
                )
                rep_log_k = float(logk_m.group(1)) if logk_m else (
                    parsed_rate.reported_log10_k if parsed_rate else None
                )

                if parsed_rate is None and pka_val is None and ph_val is None:
                    continue

                ev = EvidenceSpan(
                    doc_id=doc_id,
                    page_num=page_num,
                    locator=f"Line {line_no} (Prose)",
                    char_start=s_start,
                    char_end=s_end,
                    verbatim_quote=sent_clean,
                )
                mentions.append(
                    ProseMention(
                        compound_name=found_comp,
                        oxidant=ox,
                        parsed_rate=parsed_rate,
                        ph=ph_val,
                        pka=pka_val,
                        temperature_c=temp_c,
                        water_matrix=defaults.get("water_matrix"),
                        radical_scavenger=defaults.get("radical_scavenger"),
                        k_ha=k_ha_val,
                        k_a_minus=k_am_val,
                        reported_log10_k=rep_log_k,
                        evidence=ev,
                    )
                )

        return mentions

    def _find_compound_in_sentence(self, sentence: str) -> Optional[str]:
        from .chem_validator import REFERENCE_COMPOUNDS

        low = sentence.lower()
        # Match longest name first to avoid partial substring collisions (e.g. "4-Chlorophenol" before "Phenol")
        candidates: List[Tuple[int, str]] = []
        for comp in REFERENCE_COMPOUNDS:
            for term in (comp.name, *comp.aliases):
                if len(term) < 4:
                    continue
                pat = r"(?<![A-Za-z0-9])" + re.escape(term.lower()) + r"(?![A-Za-z0-9])"
                if re.search(pat, low):
                    candidates.append((len(term), comp.name))
        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
        return None

    def _extract_named_sub_k(self, sentence: str, var_regex: str) -> Optional[float]:
        pat = re.compile(
            var_regex
            + r"\s*=\s*([^\s,;,\)]+(?:\s*(?:±|\\pm)\s*[^\s,;]+)?(?:\s*(?:×|\\times|\\cdot|x|\*)\s*[lI1][0O]\s*\^\s*\{?[+-]?\d+\}?)?(?:\s*(?:mM|M|L)\s*[^\s,;\.\)]*s\^?\{?-1\}?)?)",
            flags=re.IGNORECASE,
        )
        m = pat.search(sentence)
        if not m:
            return None
        pr = parse_scientific_rate_string(
            m.group(1), default_units="M^-1 s^-1", enable_ocr_repair=self.enable_ocr_repair
        )
        if not pr:
            return None
        norm_v, _, _, _ = normalize_rate_units(pr.k_value, pr.units or "M^-1 s^-1")
        return norm_v

    def _extract_rate_from_sentence(self, sentence: str) -> Optional[ParsedRateValue]:
        # Match explicit k = ... or rate constant of ... with units, or standalone log k = ...
        patterns = [
            # Explicit k assignment e.g. k_O3 = (3.2 ± 0.4) × 10^5 M^-1 s^-1 or k_O3 < 0.1 M^-1 s^-1
            r"(k(?:\s*_?\s*\{?(?:O_?3|•OH|\\bullet\s*OH|OH|SO_?4\^?\{?•?-\}?|app|obs)\}?)?\s*(?:=|<|<=|>|>=|~)\s*(?:\([^\)]+\)|[+-]?\d+(?:\.\d+)?(?:\s*(?:±|\\pm)\s*\d+(?:\.\d+)?)?)(?:\s*(?:×|\\times|\\cdot|·|x|\*)\s*[lI1][0O]\s*\^\s*\{?[+-]?\d+\}?|[eE][+-]?\d+|[\u2070\u00b9\u00b2\u00b3\u2074-\u2079\u207a\u207b]+)?(?:\s*(?:cm\^?3\s*molecule\^?-1\s*s\^?-1|L\s*(?:mmol|mol)\^?\{?-1\}?\s*(?:s|min)\^?\{?-1\}?|(?:mM|μM|uM|M)\^?\{?-1\}?\s*(?:s|min)\^?\{?-1\}?|(?:min|s)\^?\{?-1\}?))?)",
            # Value followed by bimolecular rate constant units without leading 'k ='
            r"((?:\(\s*[+-]?\d+(?:\.\d+)?\s*(?:±|\\pm)\s*\d+(?:\.\d+)?\s*\)|(?:<=|>=|<|>|~)?\s*[+-]?\d+(?:\.\d+)?)\s*(?:(?:×|\\times|\\cdot|·|x|\*)\s*[lI1][0O]\s*\^\s*\{?[+-]?\d+\}?|[eE][+-]?\d+)?\s*(?:cm\^?3\s*molecule\^?-1\s*s\^?-1|L\s*(?:mmol|mol)\^?\{?-1\}?\s*(?:s|min)\^?\{?-1\}?|(?:mM|μM|uM|M)\^?\{?-1\}?\s*(?:s|min)\^?\{?-1\}?))",
            # Standalone log k = 5.51
            r"(log(?:_?\{?10\}?)?\s*\(?k[^=:\)]*\)?\s*(?:=|:)\s*[+-]?\d+(?:\.\d+)?)",
        ]
        # Exclude intrinsic k_HA / k_A- if looking for apparent k, unless no apparent k exists
        s_search = re.sub(
            r"k_?\{?(?:HA|neutral|A\-|A\^\-|anion)\}?\s*=\s*[^,;,\)]+",
            " ",
            sentence,
            flags=re.IGNORECASE,
        )
        for pat in patterns:
            m = re.search(pat, s_search, flags=re.IGNORECASE)
            if m:
                cand = m.group(1).strip()
                parsed = parse_scientific_rate_string(
                    cand, default_units="M^-1 s^-1", enable_ocr_repair=self.enable_ocr_repair
                )
                if parsed is not None:
                    return parsed
        return None

    def _prose_mention_to_record(
        self,
        pm: ProseMention,
        doc_id: str,
        rec_idx: int,
        defaults: Dict[str, Any],
    ) -> ReactionKineticsRecord:
        assert pm.parsed_rate is not None
        assert pm.oxidant is not None
        ref = lookup_reference_compound(pm.compound_name)
        units_str = pm.parsed_rate.units or "M^-1 s^-1"
        if self.enable_ocr_repair:
            std_k, _, _, _ = normalize_rate_units(pm.parsed_rate.k_value, units_str)
        else:
            std_k = pm.parsed_rate.k_value

        temp_c = pm.temperature_c if pm.temperature_c is not None else defaults.get("temperature_c", 20.0)
        temp_k = round(temp_c + 273.15, 4)

        return ReactionKineticsRecord(
            record_id=f"{doc_id}_R{rec_idx:02d}",
            doc_id=doc_id,
            compound_name=pm.compound_name,
            cas_number=ref.cas if ref else None,
            compound_class=ref.compound_class if ref else "Phenols",
            raw_smiles=None,
            oxidant=pm.oxidant,
            raw_k_string=pm.parsed_rate.raw_string,
            inequality=pm.parsed_rate.inequality,
            k_value=pm.parsed_rate.k_value,
            k_uncertainty=pm.parsed_rate.uncertainty,
            k_units=units_str,
            standardized_k=std_k,
            reported_log10_k=pm.reported_log10_k,
            ph=pm.ph,
            pka=pm.pka if pm.pka is not None else (ref.pka if ref else None),
            pka2=ref.pka2 if ref else None,
            k_ha=pm.k_ha,
            k_a_minus=pm.k_a_minus,
            temperature_c=temp_c,
            temperature_k=temp_k,
            temperature_defaulted=pm.temperature_c is None,
            water_matrix=pm.water_matrix or "phosphate_buffer",
            radical_scavenger=pm.radical_scavenger,
            evidence=pm.evidence,
            prose_evidence=pm.evidence,
            metadata={
                "ocr_repairs": pm.parsed_rate.ocr_repairs,
                "extracted_from_prose": True,
            },
        )
