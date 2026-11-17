"""Model Context Protocol (MCP) JSON-RPC 2.0 Stdio Server & Tool Registry.

Exposes deterministic cheminformatics & domain-physics verification tools for
Claude Code, Codex, OmniHarness, and AI4Science literature agents:
  1. `extract_kinetics_records`: Dual-agent literature extraction + RDKit/physics audit.
  2. `validate_smiles_and_units`: Canonical RDKit structure + unit/log10(k)/diffusion audit.
  3. `verify_speciation_kinetics`: Henderson-Hasselbalch speciation physics verification.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Dict, List, Optional

from .chem_validator import ChemValidator, verify_speciation_physics
from .cross_verifier import CrossVerificationEngine, PipelineMode

SERVER_NAME = "chemlit-kinetics-miner"
SERVER_VERSION = "0.1.0"
DEFAULT_MCP_PROTOCOL_VERSION = "2024-11-05"


TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "name": "extract_kinetics_records",
        "description": (
            "Extract aqueous oxidation rate constants (O3, •OH, SO4•-, ClO•), experimental "
            "conditions (pH, T, pKa, water matrix), canonical RDKit SMILES/InChIKey, and exact "
            "character-offset EvidenceSpan provenance from Markdown/LaTeX/OCR paper excerpts."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Scientific Markdown, LaTeX tabular, or OCR prose excerpt.",
                },
                "doc_id": {
                    "type": "string",
                    "default": "MCP_DOC_01",
                    "description": "Source document identifier or DOI.",
                },
                "page_num": {
                    "type": "integer",
                    "default": 1,
                    "minimum": 1,
                    "description": "1-based page number.",
                },
                "pipeline_mode": {
                    "type": "string",
                    "enum": ["baseline", "rdkit_gate", "full"],
                    "default": "full",
                    "description": "Verification depth ('full' runs Dual-Agent + RDKit + Physics Audit).",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "validate_smiles_and_units",
        "description": (
            "Validate a compound name / raw SMILES via RDKit (catching valence & broken-ring "
            "hallucinations, computing canonical SMILES, InChIKey, formula, exact MW, charge) "
            "and audit rate constant units, log10(k) parity, and Smoluchowski diffusion bounds."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "compound_name": {
                    "type": "string",
                    "description": "Compound common name or CAS number.",
                },
                "raw_smiles": {
                    "type": ["string", "null"],
                    "default": None,
                    "description": "Raw SMILES string to sanitize and canonicalize.",
                },
                "k_value": {
                    "type": "number",
                    "description": "Numeric rate constant value in reported units.",
                },
                "k_units": {
                    "type": "string",
                    "default": "M^-1 s^-1",
                    "description": "Reported rate constant units (e.g. 'M^-1 s^-1', 'mM^-1 s^-1', 'M^-1 min^-1').",
                },
                "reported_log10_k": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Reported log10(k) for logarithmic cross-checking.",
                },
                "ph": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Experimental pH.",
                },
                "pka": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Compound acid dissociation constant pKa.",
                },
                "temperature_c": {
                    "type": "number",
                    "default": 20.0,
                    "description": "Experimental temperature in degrees Celsius (°C).",
                },
            },
            "required": ["compound_name", "k_value"],
        },
    },
    {
        "name": "verify_speciation_kinetics",
        "description": (
            "Verify acid-base speciation kinetics consistency using the Henderson-Hasselbalch "
            "law k_app(pH) = alpha_0*k_HA + alpha_1*k_A- (and diprotic extension), catching "
            "pH <-> pKa column swaps and unphysical apparent rate constants."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ph": {"type": "number", "description": "Experimental aqueous pH."},
                "pka": {"type": "number", "description": "Acid dissociation constant pKa."},
                "k_ha": {
                    "type": "number",
                    "description": "Intrinsic protonated/acid species rate constant k_HA (M^-1 s^-1).",
                },
                "k_a_minus": {
                    "type": "number",
                    "description": "Intrinsic deprotonated/base species rate constant k_A- (M^-1 s^-1).",
                },
                "reported_k_app": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Reported apparent second-order rate constant k_app(pH) in M^-1 s^-1.",
                },
                "pka2": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Optional second dissociation constant pKa2 for diprotic compounds.",
                },
                "k_a2_minus": {
                    "type": ["number", "null"],
                    "default": None,
                    "description": "Optional intrinsic rate constant of doubly deprotonated species (M^-1 s^-1).",
                },
                "tolerance_log": {
                    "type": "number",
                    "default": 0.15,
                    "description": "Maximum allowed |log10(reported_k_app) - log10(predicted_k_app)|.",
                },
            },
            "required": ["ph", "pka", "k_ha", "k_a_minus"],
        },
    },
]


class ChemLitMCPServer:
    """Standard Model Context Protocol (MCP) JSON-RPC 2.0 server."""

    def __init__(self) -> None:
        self.validator = ChemValidator()
        self._tool_handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "extract_kinetics_records": self._tool_extract_kinetics_records,
            "validate_smiles_and_units": self._tool_validate_smiles_and_units,
            "verify_speciation_kinetics": self._tool_verify_speciation_kinetics,
        }

    def _tool_extract_kinetics_records(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "text" not in args or not isinstance(args["text"], str):
            raise ValueError("Parameter 'text' (string) is required.")
        mode: PipelineMode = args.get("pipeline_mode", "full")
        if mode not in ("baseline", "rdkit_gate", "full"):
            raise ValueError(f"Invalid pipeline_mode '{mode}'.")
        doc_id = str(args.get("doc_id", "MCP_DOC_01"))
        page_num = int(args.get("page_num", 1))

        engine = CrossVerificationEngine(mode=mode, validator=self.validator)
        out = engine.run(text=args["text"], doc_id=doc_id, page_num=page_num)
        return {
            "doc_id": out.doc_id,
            "pipeline_mode": out.pipeline_mode,
            "num_accepted": len(out.accepted_records),
            "num_rejected": len(out.rejected_records),
            "accepted_records": [r.model_dump() for r in out.accepted_records],
            "rejected_records": [r.model_dump() for r in out.rejected_records],
            "audit_reports": [rep.model_dump() for rep in out.audit_reports],
        }

    def _tool_validate_smiles_and_units(self, args: Dict[str, Any]) -> Dict[str, Any]:
        if "compound_name" not in args or "k_value" not in args:
            raise ValueError("Parameters 'compound_name' and 'k_value' are required.")
        res = self.validator.validate(
            compound_name=str(args["compound_name"]),
            k_value=float(args["k_value"]),
            k_units=str(args.get("k_units", "M^-1 s^-1")),
            raw_smiles=args.get("raw_smiles"),
            reported_log10_k=(
                float(args["reported_log10_k"]) if args.get("reported_log10_k") is not None else None
            ),
            ph=float(args["ph"]) if args.get("ph") is not None else None,
            pka=float(args["pka"]) if args.get("pka") is not None else None,
            temperature_c=float(args.get("temperature_c", 20.0)),
        )
        return res.model_dump()

    def _tool_verify_speciation_kinetics(self, args: Dict[str, Any]) -> Dict[str, Any]:
        for req_key in ("ph", "pka", "k_ha", "k_a_minus"):
            if req_key not in args:
                raise ValueError(f"Parameter '{req_key}' is required.")
        spec_res = verify_speciation_physics(
            ph=float(args["ph"]),
            pka=float(args["pka"]),
            k_ha=float(args["k_ha"]),
            k_a_minus=float(args["k_a_minus"]),
            reported_k_app=(
                float(args["reported_k_app"]) if args.get("reported_k_app") is not None else None
            ),
            pka2=float(args["pka2"]) if args.get("pka2") is not None else None,
            k_a2_minus=float(args["k_a2_minus"]) if args.get("k_a2_minus") is not None else None,
            tolerance_log=float(args.get("tolerance_log", 0.15)),
        )
        return spec_res.model_dump()

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single JSON-RPC 2.0 message and return response dict (or None for notifications)."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        # Notifications do not have an "id" and expect no response
        if "id" not in request:
            return None

        if method == "initialize":
            client_version = params.get("protocolVersion", DEFAULT_MCP_PROTOCOL_VERSION)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": client_version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": TOOL_DEFINITIONS},
            }

        if method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments") or {}
            if tool_name not in self._tool_handlers:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {
                        "code": -32601,
                        "message": f"Unknown MCP tool '{tool_name}'. Available: {list(self._tool_handlers)}",
                    },
                }
            try:
                payload = self._tool_handlers[tool_name](arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(payload, ensure_ascii=False, indent=2),
                            }
                        ],
                        "structuredContent": payload,
                        "isError": False,
                    },
                }
            except Exception as exc:  # noqa: BLE001
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": f"Tool execution error: {exc}"}],
                        "isError": True,
                    },
                }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def serve_stdio(self) -> None:
        """Run JSON-RPC 2.0 server over standard input/output (supports NDJSON & Content-Length)."""
        if hasattr(sys.stdin, "reconfigure"):
            sys.stdin.reconfigure(encoding="utf-8")
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")

        while True:
            line = sys.stdin.readline()
            if not line:
                break
            stripped = line.strip()
            if not stripped:
                continue

            use_content_length = False
            if stripped.lower().startswith("content-length:"):
                use_content_length = True
                length = int(stripped.split(":", 1)[1].strip())
                # Consume blank header separator line
                while True:
                    hdr = sys.stdin.readline()
                    if not hdr or hdr.strip() == "":
                        break
                raw_body = sys.stdin.read(length)
            else:
                raw_body = stripped

            try:
                req = json.loads(raw_body)
            except json.JSONDecodeError as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                self._write_response(err_resp, use_content_length)
                continue

            resp = self.handle_request(req)
            if resp is not None:
                self._write_response(resp, use_content_length)

    @staticmethod
    def _write_response(resp: Dict[str, Any], use_content_length: bool = False) -> None:
        encoded = json.dumps(resp, ensure_ascii=False)
        if use_content_length:
            byte_len = len(encoded.encode("utf-8"))
            sys.stdout.write(f"Content-Length: {byte_len}\r\n\r\n{encoded}")
        else:
            sys.stdout.write(encoded + "\n")
        sys.stdout.flush()


def run_mcp_stdio_server() -> None:
    """Entrypoint for `chemlit-mcp` console script."""
    server = ChemLitMCPServer()
    server.serve_stdio()


if __name__ == "__main__":
    run_mcp_stdio_server()
