"""Comprehensive pytest suite for ChemLit-Kinetics-Miner."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from chemlit_miner.chem_validator import (
    SMOLUCHOWSKI_AQUEOUS_LIMIT,
    ChemValidator,
    normalize_rate_units,
    validate_chemical_structure,
    verify_speciation_physics,
)
from chemlit_miner.cli import main as cli_main
from chemlit_miner.cross_verifier import CrossVerificationEngine
from chemlit_miner.mcp_server import ChemLitMCPServer
from chemlit_miner.parser import (
    ScientificKineticsParser,
    parse_ph_and_matrix,
    parse_scientific_rate_string,
    parse_temperature,
)


class TestScientificParser:
    """Verify scientific notations, OCR glyphs, Kelvin/Celsius, LaTeX, and provenance spans."""

    def test_nonfinite_measurements_do_not_match_gold(self):
        from scripts.run_evaluation import record_matches_gold

        corpus = json.loads((Path(__file__).parents[1] / "data/benchmark_corpus.json").read_text(encoding="utf-8"))
        doc = corpus["documents"][0]
        rec = CrossVerificationEngine(mode="full").run(doc["text"], doc["doc_id"]).accepted_records[0]
        gold = next(g for g in doc["gold_records"] if record_matches_gold(rec, g))
        for field in ("ph", "temperature_c", "standardized_k"):
            assert not record_matches_gold(rec.model_copy(update={field: float("nan")}), gold)

    @pytest.mark.parametrize("quote", ["", "a rate copied from another document"])
    def test_model_quote_missing_from_source_fails_provenance(self, quote):
        from scripts.run_live_llm_evaluation import _bind_evidence_span

        text = "Table header\nPhenol reacted with ozone."
        assert not _bind_evidence_span("D1", text, quote).verify_against_source(text)
        exact = _bind_evidence_span("D1", text, "Phenol reacted with ozone.")
        assert exact.verify_against_source(text)
        assert exact.char_start == text.index("Phenol")

    def test_parenthesized_uncertainty_scientific(self) -> None:
        res = parse_scientific_rate_string("(3.2 ± 0.4) × 10^5 M^-1 s^-1")
        assert res is not None
        assert math.isclose(res.k_value, 3.2e5, rel_tol=1e-9)
        assert res.uncertainty is not None and math.isclose(res.uncertainty, 0.4e5, rel_tol=1e-9)
        assert res.units == "M^-1 s^-1"
        assert res.inequality is None

    def test_latex_cdot_negative_exponent(self) -> None:
        res = parse_scientific_rate_string(r"1.8 \cdot 10^{-2} L mol^{-1} s^{-1}")
        assert res is not None
        assert math.isclose(res.k_value, 1.8e-2, rel_tol=1e-9)
        assert res.units == "L mol^-1 s^-1"

    def test_inequality_upper_bound(self) -> None:
        res = parse_scientific_rate_string("k_O3 < 0.1 M^-1 s^-1")
        assert res is not None
        assert res.inequality == "<"
        assert math.isclose(res.k_value, 0.1, rel_tol=1e-9)

    def test_standalone_log_k_parsing(self) -> None:
        res = parse_scientific_rate_string("log k = 5.51")
        assert res is not None
        assert res.derived_from_log_k is True
        assert res.reported_log10_k == pytest.approx(5.51)
        assert math.isclose(res.k_value, 10.0**5.51, rel_tol=1e-6)

    def test_ocr_confusions_unicode_and_letter_l_O(self) -> None:
        r_sup = parse_scientific_rate_string("4.5 × 10⁵ M⁻¹ s⁻¹")
        assert r_sup is not None
        assert math.isclose(r_sup.k_value, 4.5e5, rel_tol=1e-9)
        assert len(r_sup.ocr_repairs) >= 1

        r_l0 = parse_scientific_rate_string("2.4 × l0^6 M^-1 s^-1", enable_ocr_repair=True)
        assert r_l0 is not None
        assert math.isclose(r_l0.k_value, 2.4e6, rel_tol=1e-9)
        assert any("l0^6" in msg for msg in r_l0.ocr_repairs)

        r_1o = parse_scientific_rate_string("6.1 × 1O^3 M^-1 s^-1", enable_ocr_repair=True)
        assert r_1o is not None
        assert math.isclose(r_1o.k_value, 6.1e3, rel_tol=1e-9)

        # Without OCR repair enabled, raw OCR token 'l0^6' is not silently accepted
        assert parse_scientific_rate_string("2.4 × l0^6 M^-1 s^-1", enable_ocr_repair=False) is None

    def test_ph_and_temperature_conversions(self) -> None:
        ph, matrix = parse_ph_and_matrix("pH = 7.0 (phosphate buffer)")
        assert ph == pytest.approx(7.0)
        assert matrix == "phosphate_buffer"

        tc_k, tk_k = parse_temperature("T = 298 K")  # type: ignore[misc]
        assert tk_k == pytest.approx(298.0)
        assert tc_k == pytest.approx(24.85)

        tc_c, tk_c = parse_temperature("25 °C")  # type: ignore[misc]
        assert tc_c == pytest.approx(25.0)
        assert tk_c == pytest.approx(298.15)

    def test_markdown_and_latex_exact_provenance_spans(self) -> None:
        latex_doc = (
            "Experiments in phosphate buffer at pH = 7.0 and T = 293.15 K.\n"
            "\\begin{tabular}{l c c c}\n"
            "\\hline\n"
            "Compound & Oxidant & $k$ ($\\mathrm{M^{-1}\\,s^{-1}}$) & $\\log k$ \\\\\n"
            "\\hline\n"
            "Sulfamethoxazole & O3 & $5.5 \\cdot 10^{5}$ & 5.74 \\\\\n"
            "\\hline\n"
            "\\end{tabular}\n"
        )
        parser = ScientificKineticsParser()
        records, _ = parser.parse_document(latex_doc, doc_id="LATEX_TEST")
        assert len(records) == 1
        rec = records[0]
        assert rec.compound_name == "Sulfamethoxazole"
        assert math.isclose(rec.standardized_k, 5.5e5, rel_tol=1e-6)
        assert rec.evidence.verify_against_source(latex_doc) is True


class TestRDKitAndPhysicsValidator:
    """Verify RDKit sanitization, unit normalization, log10(k) parity, Smoluchowski & speciation."""

    def test_rdkit_canonicalization_and_stereochemistry(self) -> None:
        res, _ = validate_chemical_structure("Phenol", "c1ccc(O)cc1")
        assert res.is_valid_structure is True
        assert res.canonical_smiles == "Oc1ccccc1"
        assert res.inchikey == "ISWSIDIOOBJBQZ-UHFFFAOYSA-N"
        assert res.molecular_formula == "C6H6O"
        assert res.formal_charge == 0
        assert res.exact_mw == pytest.approx(94.041865, rel=1e-5)

        # Verify E/Z stereochemical InChIKey retention
        res_cinn, _ = validate_chemical_structure("trans-Cinnamic acid", "O=C(O)/C=C/c1ccccc1")
        assert res_cinn.is_valid_structure is True
        assert res_cinn.inchikey == "WBYWAXJHAXSJNI-VOTSOKGWSA-N"

    def test_rejects_pentavalent_carbon_and_broken_rings(self) -> None:
        # Pentavalent carbon C(=O)(=O)C
        res_penta, _ = validate_chemical_structure("Acetone-corrupted", "CC(=O)(=O)C")
        assert res_penta.is_valid_structure is False
        assert "INVALID_VALENCE" in res_penta.error_codes

        # Unclosed aromatic ring
        res_ring, _ = validate_chemical_structure("Benzene-broken", "c1ccccc")
        assert res_ring.is_valid_structure is False
        assert "UNCLOSED_RING" in res_ring.error_codes

        # Name-SMILES skeleton mismatch
        res_mismatch, _ = validate_chemical_structure("Sulfamethoxazole", "c1ccccc1")
        assert res_mismatch.is_valid_structure is False
        assert "NAME_STRUCTURE_MISMATCH" in res_mismatch.error_codes

    def test_unit_conversions_and_pseudo_first_order_rejection(self) -> None:
        k_mm, fac_mm, ok_mm, _ = normalize_rate_units(6.2e2, "mM^-1 s^-1")
        assert ok_mm is True
        assert fac_mm == pytest.approx(1000.0)
        assert k_mm == pytest.approx(6.2e5)

        k_min, fac_min, ok_min, _ = normalize_rate_units(420.0, "M^-1 min^-1")
        assert ok_min is True
        assert fac_min == pytest.approx(1.0 / 60.0)
        assert k_min == pytest.approx(7.0)

        _, _, ok_pfo, err_pfo = normalize_rate_units(0.085, "min^-1")
        assert ok_pfo is False
        assert err_pfo == "PSEUDO_FIRST_ORDER_UNIT"

    def test_log_k_parity_and_diffusion_and_negative_kelvin(self) -> None:
        validator = ChemValidator()

        # Catch 100x OCR exponent shift against reported log10(k)
        v_ocr = validator.validate(
            compound_name="Carbamazepine",
            k_value=3.0e3,
            k_units="M^-1 s^-1",
            reported_log10_k=5.48,
        )
        assert v_ocr.is_unit_consistent is False
        assert "OCR_EXPONENT_MISMATCH" in v_ocr.error_codes

        # Catch Smoluchowski diffusion limit (> 2.5e10 M^-1 s^-1)
        v_diff = validator.validate(
            compound_name="Atrazine",
            k_value=6.0e11,
            k_units="M^-1 s^-1",
        )
        assert v_diff.exceeds_diffusion_limit is True
        assert SMOLUCHOWSKI_AQUEOUS_LIMIT == pytest.approx(2.5e10)
        assert "EXCEEDS_SMOLUCHOWSKI_DIFFUSION_LIMIT" in v_diff.error_codes

        # Catch negative absolute temperature (-10 K = -283.15 °C)
        v_temp = validator.validate(
            compound_name="Atrazine",
            k_value=6.0,
            k_units="M^-1 s^-1",
            temperature_c=-283.15,
        )
        assert v_temp.invalid_temperature is True
        assert "NEGATIVE_ABSOLUTE_TEMPERATURE" in v_temp.error_codes

    def test_henderson_hasselbalch_speciation_and_ph_pka_swap(self) -> None:
        # Triclosan at pH 7.0, pKa 7.90, k_HA=13, k_A-=3.8e7 -> k_app = 4.25e6
        spec_ok = verify_speciation_physics(
            ph=7.0, pka=7.90, k_ha=13.0, k_a_minus=3.8e7, reported_k_app=4.25e6
        )
        assert spec_ok.is_consistent is True
        assert spec_ok.swapped_ph_pka_detected is False

        # Swapped pH=7.90, pKa=7.00 with same reported_k_app=4.25e6 -> detects pH <-> pKa swap!
        spec_swap = verify_speciation_physics(
            ph=7.90, pka=7.00, k_ha=13.0, k_a_minus=3.8e7, reported_k_app=4.25e6
        )
        assert spec_swap.is_consistent is False
        assert spec_swap.swapped_ph_pka_detected is True


class TestDualAgentCrossVerifierAndMCP:
    """Verify dual-agent table-prose reconciliation, MCP JSON-RPC 2.0 server, and CLI."""

    def test_dual_agent_auto_repairs_ocr_exponent_and_ph_pka_swap(self) -> None:
        excerpt = (
            "Ozonation was conducted at pH = 7.0 (phosphate buffer) and T = 298 K.\n\n"
            "| Compound | SMILES | Oxidant | k_app (M^-1 s^-1) | log k | pH | pKa | k_HA (M^-1 s^-1) | k_A- (M^-1 s^-1) |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
            "| Triclosan | Oc1cc(Cl)ccc1Oc1ccc(Cl)cc1Cl | O3 | 4.25 × 10^4 | 6.63 | 7.90 | 7.00 | 13 | 3.8 × 10^7 |\n\n"
            "At pH 7.0, Triclosan (pKa = 7.90) reacted rapidly with ozone with k_O3 = 4.25 × 10^6 M^-1 s^-1 (log k = 6.63)."
        )
        engine = CrossVerificationEngine(mode="full")
        out = engine.run(excerpt, doc_id="TEST_SWAP_EXP")
        assert len(out.accepted_records) == 1
        rec = out.accepted_records[0]
        assert rec.audit_report is not None
        assert rec.audit_report.status == "AUTO_REPAIRED"
        assert rec.ph == pytest.approx(7.0)
        assert rec.pka == pytest.approx(7.90)
        assert rec.standardized_k == pytest.approx(4.25e6)
        assert rec.evidence.verify_against_source(excerpt)
        assert rec.prose_evidence is not None and rec.prose_evidence.verify_against_source(excerpt)

    def test_mcp_jsonrpc_protocol_and_all_three_tools(self) -> None:
        server = ChemLitMCPServer()

        # 1. initialize
        init_resp = server.handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        assert init_resp is not None
        assert init_resp["result"]["serverInfo"]["name"] == "chemlit-kinetics-miner"

        # 2. notifications/initialized -> None
        assert server.handle_request({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

        # 3. tools/list
        list_resp = server.handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        assert list_resp is not None
        tool_names = {t["name"] for t in list_resp["result"]["tools"]}
        assert tool_names == {
            "extract_kinetics_records",
            "validate_smiles_and_units",
            "verify_speciation_kinetics",
        }

        # 4. tools/call -> validate_smiles_and_units
        val_resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "validate_smiles_and_units",
                "arguments": {
                    "compound_name": "Atrazine",
                    "raw_smiles": "CCNc1nc(NC(C)C)nc(Cl)n1",
                    "k_value": 6.0,
                    "k_units": "M^-1 s^-1",
                    "reported_log10_k": 0.78,
                },
            },
        })
        assert val_resp is not None and val_resp["result"]["isError"] is False
        sc = val_resp["result"]["structuredContent"]
        assert sc["is_valid_structure"] is True
        assert sc["is_unit_consistent"] is True

        # 5. tools/call -> verify_speciation_kinetics
        spec_resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "verify_speciation_kinetics",
                "arguments": {
                    "ph": 7.0,
                    "pka": 9.95,
                    "k_ha": 1300.0,
                    "k_a_minus": 1.4e9,
                    "reported_k_app": 1.57e6,
                },
            },
        })
        assert spec_resp is not None and spec_resp["result"]["isError"] is False
        assert spec_resp["result"]["structuredContent"]["is_consistent"] is True

        # 6. tools/call -> extract_kinetics_records
        ext_resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "extract_kinetics_records",
                "arguments": {
                    "doc_id": "MCP_TEST",
                    "text": "At pH = 7.0 and 25 °C, Carbamazepine reacted with ozone with k_O3 = 3.0 × 10^5 M^-1 s^-1.",
                },
            },
        })
        assert ext_resp is not None and ext_resp["result"]["isError"] is False
        assert ext_resp["result"]["structuredContent"]["num_accepted"] == 1

    def test_cli_validate_and_extract(self, tmp_path: Path) -> None:
        sample_md = tmp_path / "sample.md"
        out_json = tmp_path / "out.json"
        sample_md.write_text(
            "| Compound | Oxidant | k (M^-1 s^-1) | pH | T (°C) |\n"
            "| --- | --- | --- | --- | --- |\n"
            "| Phenol | O3 | 1.57 × 10^6 | 7.0 | 20.0 |\n",
            encoding="utf-8",
        )
        rc = cli_main(["extract", "--input", str(sample_md), "--output", str(out_json)])
        assert rc == 0
        parsed = json.loads(out_json.read_text(encoding="utf-8"))
        assert parsed["num_accepted"] == 1
        assert parsed["accepted_records"][0]["inchikey"] == "ISWSIDIOOBJBQZ-UHFFFAOYSA-N"

    def test_diprotic_speciation_and_mcp_error_handling(self) -> None:
        # Diprotic test: pKa1=1.6, pKa2=5.7, k0=500, k1=4.7e4, k2=5.7e5 at pH=7.0
        diprotic = verify_speciation_physics(
            ph=7.0,
            pka=1.6,
            pka2=5.7,
            k_ha=500.0,
            k_a_minus=4.7e4,
            k_a2_minus=5.7e5,
            reported_k_app=5.45e5,
        )
        assert diprotic.is_consistent is True
        assert diprotic.alpha_2 is not None and diprotic.alpha_2 > 0.94

        server = ChemLitMCPServer()
        err_resp = server.handle_request({
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool", "arguments": {}},
        })
        assert err_resp is not None
        assert err_resp["error"]["code"] == -32601

    def test_benchmark_corpus_full_provenance_exact_match(self) -> None:
        corpus_path = Path(__file__).resolve().parents[1] / "data" / "benchmark_corpus.json"
        corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
        assert corpus["num_documents"] >= 45
        engine = CrossVerificationEngine(mode="full")
        for doc in corpus["documents"]:
            out = engine.run(doc["text"], doc["doc_id"])
            for rec in out.accepted_records:
                assert rec.evidence.verify_against_source(doc["text"]) is True
                assert rec.inchikey is not None and len(rec.inchikey) == 27

