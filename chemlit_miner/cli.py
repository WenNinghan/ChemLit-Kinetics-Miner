"""Command-line interface for ChemLit-Kinetics-Miner."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .chem_validator import ChemValidator, verify_speciation_physics
from .cross_verifier import CrossVerificationEngine, PipelineMode
from .mcp_server import run_mcp_stdio_server


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chemlit-miner",
        description=(
            "ChemLit-Kinetics-Miner: Evidence-Traceable Scientific Literature Extraction & "
            "Cheminformatics/Physics Verification Pipeline."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand: extract
    p_ext = sub.add_parser("extract", help="Extract and audit kinetics records from a document file or text.")
    p_ext.add_argument("--input", "-i", type=str, help="Path to Markdown/LaTeX/txt file (reads stdin if omitted).")
    p_ext.add_argument("--doc-id", type=str, default="CLI_DOC_01", help="Source document ID or DOI.")
    p_ext.add_argument("--page", type=int, default=1, help="1-based source page number.")
    p_ext.add_argument(
        "--mode",
        choices=["baseline", "rdkit_gate", "full"],
        default="full",
        help="Verification pipeline mode (default: full).",
    )
    p_ext.add_argument("--output", "-o", type=str, help="Optional JSON output file path.")

    # Subcommand: validate
    p_val = sub.add_parser("validate", help="Validate SMILES structure, rate units, log10(k), and physical bounds.")
    p_val.add_argument("--compound", "-c", required=True, type=str, help="Compound name or CAS number.")
    p_val.add_argument("--smiles", "-s", type=str, default=None, help="Raw SMILES string.")
    p_val.add_argument("--k-value", "-k", required=True, type=float, help="Numeric rate constant.")
    p_val.add_argument("--units", "-u", type=str, default="M^-1 s^-1", help="Rate constant units.")
    p_val.add_argument("--log-k", type=float, default=None, help="Reported log10(k) for parity check.")
    p_val.add_argument("--ph", type=float, default=None, help="Experimental pH.")
    p_val.add_argument("--pka", type=float, default=None, help="Acid dissociation constant pKa.")
    p_val.add_argument("--temp-c", type=float, default=20.0, help="Temperature in °C.")

    # Subcommand: speciation
    p_spec = sub.add_parser("speciation", help="Verify Henderson-Hasselbalch speciation kinetics consistency.")
    p_spec.add_argument("--ph", required=True, type=float, help="Experimental pH.")
    p_spec.add_argument("--pka", required=True, type=float, help="Compound pKa.")
    p_spec.add_argument("--k-ha", required=True, type=float, help="Intrinsic k_HA (M^-1 s^-1).")
    p_spec.add_argument("--k-a-minus", required=True, type=float, help="Intrinsic k_A- (M^-1 s^-1).")
    p_spec.add_argument("--reported-k-app", type=float, default=None, help="Reported k_app(pH) (M^-1 s^-1).")

    # Subcommand: mcp
    sub.add_parser("mcp", help="Start the Model Context Protocol (MCP) JSON-RPC 2.0 stdio server.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "mcp":
        run_mcp_stdio_server()
        return 0

    if args.command == "extract":
        if args.input:
            text = Path(args.input).read_text(encoding="utf-8")
        else:
            text = sys.stdin.read()
        mode: PipelineMode = args.mode
        engine = CrossVerificationEngine(mode=mode)
        out = engine.run(text=text, doc_id=args.doc_id, page_num=args.page)
        payload = {
            "doc_id": out.doc_id,
            "pipeline_mode": out.pipeline_mode,
            "num_accepted": len(out.accepted_records),
            "num_rejected": len(out.rejected_records),
            "accepted_records": [r.model_dump() for r in out.accepted_records],
            "rejected_records": [r.model_dump() for r in out.rejected_records],
            "audit_reports": [rep.model_dump() for rep in out.audit_reports],
        }
        rendered = json.dumps(payload, ensure_ascii=False, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            print(rendered)
        return 0

    if args.command == "validate":
        validator = ChemValidator()
        res = validator.validate(
            compound_name=args.compound,
            k_value=args.k_value,
            k_units=args.units,
            raw_smiles=args.smiles,
            reported_log10_k=args.log_k,
            ph=args.ph,
            pka=args.pka,
            temperature_c=args.temp_c,
        )
        print(json.dumps(res.model_dump(), ensure_ascii=False, indent=2))
        return 0 if res.is_valid_structure and res.is_unit_consistent and not res.exceeds_diffusion_limit else 1

    if args.command == "speciation":
        spec = verify_speciation_physics(
            ph=args.ph,
            pka=args.pka,
            k_ha=args.k_ha,
            k_a_minus=args.k_a_minus,
            reported_k_app=args.reported_k_app,
        )
        print(json.dumps(spec.model_dump(), ensure_ascii=False, indent=2))
        return 0 if spec.is_consistent else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
