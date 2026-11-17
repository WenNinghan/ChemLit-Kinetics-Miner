"""ChemLit-Kinetics-Miner: Evidence-Traceable Scientific Literature Extraction & Cheminformatics Verification."""

from .chem_validator import (
    REFERENCE_COMPOUNDS,
    SMOLUCHOWSKI_AQUEOUS_LIMIT,
    ChemValidator,
    lookup_reference_compound,
    normalize_rate_units,
    validate_chemical_structure,
    verify_speciation_physics,
)
from .cross_verifier import CrossVerificationEngine, ExtractionRunOutput, LLMBackendConfig
from .mcp_server import TOOL_DEFINITIONS, ChemLitMCPServer, run_mcp_stdio_server
from .parser import (
    ParsedRateValue,
    ProseMention,
    ScientificKineticsParser,
    parse_ph_and_matrix,
    parse_scientific_rate_string,
    parse_temperature,
)
from .schema import (
    AuditStatus,
    ChemAuditReport,
    ChemicalValidationResult,
    CrossCheckAuditLog,
    EvidenceSpan,
    ReactionKineticsRecord,
    SpeciationCheckResult,
)

__version__ = "0.1.0"

__all__ = [
    "AuditStatus",
    "ChemAuditReport",
    "ChemLitMCPServer",
    "ChemValidator",
    "ChemicalValidationResult",
    "CrossCheckAuditLog",
    "CrossVerificationEngine",
    "EvidenceSpan",
    "ExtractionRunOutput",
    "LLMBackendConfig",
    "ParsedRateValue",
    "ProseMention",
    "REFERENCE_COMPOUNDS",
    "ReactionKineticsRecord",
    "SMOLUCHOWSKI_AQUEOUS_LIMIT",
    "ScientificKineticsParser",
    "SpeciationCheckResult",
    "TOOL_DEFINITIONS",
    "lookup_reference_compound",
    "normalize_rate_units",
    "parse_ph_and_matrix",
    "parse_scientific_rate_string",
    "parse_temperature",
    "run_mcp_stdio_server",
    "validate_chemical_structure",
    "verify_speciation_physics",
]
