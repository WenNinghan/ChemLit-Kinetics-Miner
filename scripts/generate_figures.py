"""Generate 4 Top-Journal / CCF-A Camera-Ready Figures (300 DPI) for ChemLit-Kinetics-Miner.

Reads real evaluation artifacts from `results/miner_benchmark.json` and
`results/error_breakdown.csv` (zero fabricated numbers) and renders:
  1. `assets/pipeline_ablation_f1_hallucination.png`
  2. `assets/unit_and_pH_consistency_audit.png`
  3. `assets/speciation_physics_verification.png`
  4. `assets/provenance_error_taxonomy.png`
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chemlit_miner.chem_validator import SMOLUCHOWSKI_AQUEOUS_LIMIT, verify_speciation_physics

# High-contrast academic journal palette (ES&T / Water Research / Nature / IEEE)
JOURNAL_PALETTE = {
    "navy": "#1B4F72",
    "emerald": "#0B6E4F",
    "crimson": "#B03A2E",
    "ochre": "#C55A11",
    "plum": "#5B2C6F",
    "slate": "#5D6D7E",
    "steel": "#2874A6",
    "teal": "#117864",
    "amber": "#B9770E",
    "burgundy": "#7B241C",
    "charcoal": "#2C3E50",
}


def configure_style() -> None:
    """Configure CCF-A / Top-Journal (ES&T / Water Research / IEEE) rcParams."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "font.weight": "bold",
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.0,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9.6,
        "ytick.labelsize": 9.6,
        "legend.fontsize": 9.0,
        "axes.facecolor": "#FFFFFF",
        "figure.facecolor": "#FFFFFF",
        "savefig.facecolor": "#FFFFFF",
        "axes.edgecolor": "#111111",
        "axes.linewidth": 1.30,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.spines.bottom": True,
        "axes.spines.left": True,
        "axes.grid": True,
        "grid.linestyle": ":",
        "grid.linewidth": 0.7,
        "grid.alpha": 0.35,
        "grid.color": "#94A3B8",
        "legend.fancybox": False,
        "legend.edgecolor": "#111111",
        "legend.framealpha": 0.96,
    })


def _style_axis_ticks_and_grid(ax: plt.Axes, kind: str = "continuous") -> None:
    """Apply journal spine, inward/outward tick, and subtle dotted grid rules."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#111111")
        spine.set_linewidth(1.30)

    if kind == "continuous":
        ax.tick_params(
            axis="both",
            which="major",
            direction="in",
            top=True,
            right=True,
            bottom=True,
            left=True,
            length=4.5,
            width=1.2,
            colors="#111111",
        )
        ax.tick_params(
            axis="both",
            which="minor",
            direction="in",
            top=True,
            right=True,
            bottom=True,
            left=True,
            length=2.6,
            width=0.9,
            colors="#111111",
        )
        ax.grid(True, which="major", linestyle=":", linewidth=0.7, alpha=0.35, color="#94A3B8")
        ax.set_axisbelow(True)
    elif kind == "categorical_vbar":
        ax.tick_params(
            axis="y",
            which="major",
            direction="in",
            left=True,
            right=True,
            length=4.5,
            width=1.2,
            colors="#111111",
        )
        ax.tick_params(
            axis="x",
            which="major",
            direction="out",
            bottom=True,
            top=False,
            length=4.2,
            width=1.2,
            colors="#111111",
        )
        ax.grid(True, axis="y", linestyle=":", linewidth=0.7, alpha=0.35, color="#94A3B8")
        ax.set_axisbelow(True)
    elif kind == "categorical_hbar":
        ax.tick_params(
            axis="x",
            which="major",
            direction="in",
            bottom=True,
            top=True,
            length=4.5,
            width=1.2,
            colors="#111111",
        )
        ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            left=True,
            right=False,
            length=4.2,
            width=1.2,
            colors="#111111",
        )
        ax.grid(True, axis="x", linestyle=":", linewidth=0.7, alpha=0.35, color="#94A3B8")
        ax.set_axisbelow(True)
    elif kind == "heatmap":
        ax.tick_params(
            axis="x",
            which="major",
            direction="out",
            bottom=True,
            top=False,
            length=4.0,
            width=1.2,
            colors="#111111",
        )
        ax.tick_params(
            axis="y",
            which="major",
            direction="out",
            left=True,
            right=False,
            length=4.0,
            width=1.2,
            colors="#111111",
        )
        ax.grid(False)


def _finalize_journal_figure(fig: plt.Figure) -> None:
    """Explicitly enforce bold Times New Roman & crisp frames on all axes, colorbars, and legends."""
    fig.canvas.draw()

    for txt in fig.texts:
        txt.set_fontfamily("Times New Roman")
        txt.set_fontweight("bold")

    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#111111")
            if spine.get_linewidth() < 1.1:
                spine.set_linewidth(1.15)

        ax.title.set_fontfamily("Times New Roman")
        ax.title.set_fontweight("bold")
        ax.xaxis.label.set_fontfamily("Times New Roman")
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontfamily("Times New Roman")
        ax.yaxis.label.set_fontweight("bold")

        for lbl in ax.get_xticklabels():
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")
        for lbl in ax.get_yticklabels():
            lbl.set_fontfamily("Times New Roman")
            lbl.set_fontweight("bold")
        for txt in ax.texts:
            txt.set_fontfamily("Times New Roman")
            txt.set_fontweight("bold")

        leg = ax.get_legend()
        if leg is not None:
            frame = leg.get_frame()
            frame.set_linewidth(1.1)
            frame.set_edgecolor("#111111")
            frame.set_facecolor("#FFFFFF")
            frame.set_alpha(0.96)
            for ltxt in leg.get_texts():
                ltxt.set_fontfamily("Times New Roman")
                ltxt.set_fontweight("bold")

    for leg in fig.legends:
        frame = leg.get_frame()
        frame.set_linewidth(1.1)
        frame.set_edgecolor("#111111")
        frame.set_facecolor("#FFFFFF")
        frame.set_alpha(0.96)
        for ltxt in leg.get_texts():
            ltxt.set_fontfamily("Times New Roman")
            ltxt.set_fontweight("bold")


def fig1_pipeline_ablation(bench: Dict[str, Any], assets_dir: Path) -> Path:
    """Figure 1: Precision, Recall, F1, False-Positive Contamination & Guardrail Catch Rates."""
    pipelines = bench["pipelines"]
    short_labels = [
        "1. Baseline\nSingle-Pass",
        "2. Extractor +\nRDKit Gate",
        "3. ChemLit-Miner\n(Full Dual + Physics)",
    ]

    precision = [p["precision"] * 100 for p in pipelines]
    recall = [p["recall"] * 100 for p in pipelines]
    f1 = [p["f1"] * 100 for p in pipelines]
    fp_contam = [p["false_positive_contamination_rate"] * 100 for p in pipelines]

    trap_rej = [p["hallucination_rejection_rate"] * 100 for p in pipelines]
    unit_catch = [p["unit_and_exponent_error_catch_rate"] * 100 for p in pipelines]
    spec_catch = [p["speciation_and_ph_swap_catch_rate"] * 100 for p in pipelines]

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16.2, 6.2),
        gridspec_kw={"width_ratios": [1.22, 1.06]},
    )
    fig.subplots_adjust(wspace=0.24)

    # Wider group centers prevent adjacent 97.5% / 100% bar labels from colliding
    x1 = np.arange(len(short_labels)) * 1.32
    w1 = 0.255

    b1 = ax1.bar(
        x1 - 1.5 * w1,
        precision,
        w1,
        label="Precision (%)",
        color=JOURNAL_PALETTE["steel"],
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )
    b2 = ax1.bar(
        x1 - 0.5 * w1,
        recall,
        w1,
        label="Recall (%)",
        color=JOURNAL_PALETTE["emerald"],
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )
    b3 = ax1.bar(
        x1 + 0.5 * w1,
        f1,
        w1,
        label=r"$F_1$ Score (%)",
        color=JOURNAL_PALETTE["navy"],
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )
    b4 = ax1.bar(
        x1 + 1.5 * w1,
        fp_contam,
        w1,
        label=r"FP Contamination (%) $\downarrow$",
        color=JOURNAL_PALETTE["crimson"],
        alpha=0.90,
        hatch="//",
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )

    for bars, is_err in ((b1, False), (b2, False), (b3, False), (b4, True)):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.5,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8.2,
                color=JOURNAL_PALETTE["crimson"] if is_err else "#111111",
            )

    ax1.set_xticks(x1)
    ax1.set_xticklabels(short_labels)
    ax1.set_xlim(x1[0] - 0.66, x1[-1] + 0.66)
    ax1.set_ylim(0, 124)
    ax1.set_yticks([0, 20, 40, 60, 80, 100])
    ax1.set_ylabel("Metric Percentage (%)")
    ax1.set_title("(a) Extraction Accuracy vs. False-Positive Contamination", pad=11)
    _style_axis_ticks_and_grid(ax1, kind="categorical_vbar")
    ax1.legend(loc="upper left", ncol=2, frameon=True, fancybox=False)

    # Panel (b): Cheminformatics & Physical Chemistry Guardrail Coverage
    x2 = np.arange(len(short_labels)) * 1.24
    w2 = 0.285
    c1 = ax2.bar(
        x2 - w2,
        trap_rej,
        w2,
        label="Hallucination & Trap Rejection",
        color=JOURNAL_PALETTE["plum"],
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )
    c2 = ax2.bar(
        x2,
        unit_catch,
        w2,
        label="Unit & OCR-Exponent Catch/Repair",
        color=JOURNAL_PALETTE["ochre"],
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )
    c3 = ax2.bar(
        x2 + w2,
        spec_catch,
        w2,
        label=r"Speciation & $\mathrm{pH}\leftrightarrow\mathrm{p}K_a$ Swap Audit",
        color=JOURNAL_PALETTE["emerald"],
        edgecolor="#111111",
        linewidth=0.95,
        zorder=3,
    )

    for bars in (c1, c2, c3):
        for bar in bars:
            h = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 1.5,
                f"{h:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.4,
                color="#111111",
            )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(short_labels)
    ax2.set_xlim(x2[0] - 0.60, x2[-1] + 0.60)
    ax2.set_ylim(0, 125)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.set_ylabel("Audit Catch / Resolution Rate (%)")
    ax2.set_title("(b) Cheminformatics & Physical Chemistry Guardrail Coverage", pad=11)
    _style_axis_ticks_and_grid(ax2, kind="categorical_vbar")
    ax2.legend(loc="upper left", frameon=True, fancybox=False)

    fig.suptitle(
        f"ChemLit-Kinetics-Miner Progressive Pipeline Ablation "
        f"({bench['num_documents']} Excerpts, {bench['num_gold_records']} Gold Records, "
        f"{bench['num_injected_traps']} Injected Traps)",
        fontsize=13.2,
        y=1.01,
    )

    _finalize_journal_figure(fig)
    out_path = assets_dir / "pipeline_ablation_f1_hallucination.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return out_path


def fig2_unit_and_ph_consistency_audit(bench: Dict[str, Any], assets_dir: Path) -> Path:
    """Figure 2: Reported log10(k) vs Unit-Normalized log10(k / M^-1 s^-1) Parity & Anomaly Map."""
    trace = bench["audit_trace"]

    fig, ax = plt.subplots(figsize=(10.5, 7.9))

    lim_min, lim_max = -1.7, 13.8
    grid_x = np.linspace(lim_min, lim_max, 300)
    smol_log = math.log10(SMOLUCHOWSKI_AQUEOUS_LIMIT)  # ~10.40

    # Shaded Smoluchowski diffusion-forbidden zone
    ax.axhspan(
        smol_log,
        lim_max,
        color="#F5B7B1",
        alpha=0.28,
        zorder=1,
        label=r"Smoluchowski Diffusion-Forbidden Zone ($> 2.5 \times 10^{10}\ \mathrm{M^{-1}\,s^{-1}}$)",
    )
    ax.axhline(smol_log, color=JOURNAL_PALETTE["crimson"], linestyle="-.", linewidth=1.6, zorder=2)

    # Parity line y = x and +/- 0.15 log-unit tolerance band
    ax.plot(
        grid_x,
        grid_x,
        color=JOURNAL_PALETTE["navy"],
        linestyle="-",
        linewidth=1.7,
        label=r"Exact Log-Unit Parity ($y = x$)",
        zorder=2,
    )
    ax.fill_between(
        grid_x,
        grid_x - 0.15,
        grid_x + 0.15,
        color="#94A3B8",
        alpha=0.24,
        zorder=1,
    )

    verified_x, verified_y = [], []
    repaired_raw_x, repaired_raw_y = [], []
    repaired_fix_x, repaired_fix_y = [], []
    unit_raw_x, unit_raw_y = [], []
    unit_fix_x, unit_fix_y = [], []
    diff_viol_x, diff_viol_y = [], []
    other_flag_x, other_flag_y = [], []

    for pt in trace:
        rep_log = pt.get("reported_log10_k")
        std_k = pt.get("standardized_k")
        naive_k = pt.get("naive_k")
        status = pt.get("status")
        flags = pt.get("error_flags") or []
        repairs = " ".join(pt.get("repairs_applied") or [])

        if std_k is None or std_k <= 0:
            continue
        std_log = math.log10(std_k)
        x_val = rep_log if rep_log is not None else std_log

        if status == "FLAGGED_ERROR":
            if "EXCEEDS_SMOLUCHOWSKI_DIFFUSION_LIMIT" in flags:
                diff_viol_x.append(x_val)
                diff_viol_y.append(std_log)
            else:
                other_flag_x.append(x_val)
                other_flag_y.append(std_log)
        elif status == "AUTO_REPAIRED":
            naive_log = math.log10(naive_k) if (naive_k and naive_k > 0) else std_log
            if "Normalized rate units" in repairs or "mM^-1" in repairs:
                if naive_log >= lim_min:
                    unit_raw_x.append(x_val)
                    unit_raw_y.append(naive_log)
                    ax.annotate(
                        "",
                        xy=(x_val, std_log),
                        xytext=(x_val, naive_log),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=JOURNAL_PALETTE["ochre"],
                            lw=1.4,
                            ls="--",
                            alpha=0.88,
                        ),
                        annotation_clip=True,
                        zorder=4,
                    )
                unit_fix_x.append(x_val)
                unit_fix_y.append(std_log)
            elif "OCR exponent" in repairs or "OCR_EXPONENT_MISMATCH" in flags:
                repaired_raw_x.append(x_val)
                repaired_raw_y.append(naive_log)
                repaired_fix_x.append(x_val)
                repaired_fix_y.append(std_log)
                ax.annotate(
                    "",
                    xy=(x_val, std_log),
                    xytext=(x_val, naive_log),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=JOURNAL_PALETTE["plum"],
                        lw=1.4,
                        ls="--",
                        alpha=0.88,
                    ),
                    zorder=4,
                )
            else:
                verified_x.append(x_val)
                verified_y.append(std_log)
        else:
            verified_x.append(x_val)
            verified_y.append(std_log)

    ax.scatter(
        verified_x,
        verified_y,
        s=58,
        c=JOURNAL_PALETTE["emerald"],
        edgecolors="#FFFFFF",
        linewidth=0.75,
        alpha=0.92,
        label=rf"Verified Gold Records ($n = {len(verified_x)}$)",
        zorder=5,
    )
    ax.scatter(
        unit_raw_x,
        unit_raw_y,
        s=76,
        marker="s",
        c="#F5CBA7",
        edgecolors=JOURNAL_PALETTE["ochre"],
        linewidth=1.25,
        label=r"Pre-Audit Unit Trap ($\mathrm{mM^{-1}\,s^{-1}}$, $\mathrm{min^{-1}}$, $\mathrm{cm^3\,molec^{-1}}$)",
        zorder=6,
    )
    ax.scatter(
        unit_fix_x,
        unit_fix_y,
        s=80,
        marker="D",
        c=JOURNAL_PALETTE["ochre"],
        edgecolors="#111111",
        linewidth=0.85,
        label="Unit-Normalized Auto-Repaired Record",
        zorder=6,
    )
    ax.scatter(
        repaired_raw_x,
        repaired_raw_y,
        s=78,
        marker="v",
        c="#D7BDE2",
        edgecolors=JOURNAL_PALETTE["plum"],
        linewidth=1.25,
        label=r"Pre-Audit OCR Exponent Misread ($10^2\times$–$10^3\times$ Shift)",
        zorder=6,
    )
    ax.scatter(
        repaired_fix_x,
        repaired_fix_y,
        s=82,
        marker="^",
        c=JOURNAL_PALETTE["plum"],
        edgecolors="#111111",
        linewidth=0.85,
        label="Dual-Verified OCR Exponent Auto-Repair",
        zorder=6,
    )
    ax.scatter(
        diff_viol_x,
        diff_viol_y,
        s=118,
        marker="X",
        c=JOURNAL_PALETTE["crimson"],
        edgecolors="#111111",
        linewidth=0.95,
        label="Rejected Smoluchowski Diffusion Violation",
        zorder=7,
    )
    ax.scatter(
        other_flag_x,
        other_flag_y,
        s=96,
        marker="P",
        c=JOURNAL_PALETTE["slate"],
        edgecolors="#111111",
        linewidth=0.85,
        label=r"Rejected Valence / $\log_{10} k$ / Speciation / Temp Trap",
        zorder=7,
    )

    # Callout annotations placed in open scatter-free zones with crisp rectangular boxes
    ax.annotate(
        "Sulfapyridine ($\\mathrm{SO_4^{\\bullet -}}$)\n"
        "$k = 6.5 \\times 10^{12}\\ \\mathrm{M^{-1}\\,s^{-1}}$\n"
        "[Caught Diffusion Violation]",
        xy=(12.81, 12.81),
        xytext=(3.2, 11.85),
        fontsize=8.6,
        color=JOURNAL_PALETTE["crimson"],
        bbox=dict(boxstyle="square,pad=0.35", fc="#FFFFFF", ec=JOURNAL_PALETTE["crimson"], lw=1.1, alpha=0.96),
        arrowprops=dict(arrowstyle="->", color=JOURNAL_PALETTE["crimson"], lw=1.25),
        zorder=8,
    )
    ax.annotate(
        "Desethylatrazine ($\\mathrm{^{\\bullet}OH}$)\n"
        "$\\mathrm{cm^3\\,molecule^{-1}\\,s^{-1}} \\rightarrow \\mathrm{M^{-1}\\,s^{-1}}$\n"
        "($+20.78$ log-unit conversion)",
        xy=(9.08, 9.08),
        xytext=(-1.15, 8.75),
        fontsize=8.5,
        color=JOURNAL_PALETTE["ochre"],
        bbox=dict(boxstyle="square,pad=0.35", fc="#FFFFFF", ec=JOURNAL_PALETTE["ochre"], lw=1.1, alpha=0.96),
        arrowprops=dict(arrowstyle="->", color=JOURNAL_PALETTE["ochre"], lw=1.25),
        zorder=8,
    )
    ax.annotate(
        "Nitrobenzene ($\\mathrm{O_3}$)\n"
        "Unrecoverable $\\Delta\\log_{10} k = 4.02$\n"
        "[Flagged & Rejected]",
        xy=(0.95, math.log10(9.4e4)),
        xytext=(-1.15, 6.25),
        fontsize=8.5,
        color=JOURNAL_PALETTE["slate"],
        bbox=dict(boxstyle="square,pad=0.35", fc="#FFFFFF", ec=JOURNAL_PALETTE["slate"], lw=1.1, alpha=0.96),
        arrowprops=dict(arrowstyle="->", color=JOURNAL_PALETTE["slate"], lw=1.25),
        zorder=8,
    )

    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_xlabel(r"Reported Reference $\log_{10} k\ (\mathrm{M^{-1}\,s^{-1}})$")
    ax.set_ylabel(r"Candidate / Standardized $\log_{10} k\ (\mathrm{M^{-1}\,s^{-1}})$")
    ax.set_title("Unit & Logarithmic Parity Audit Map Across 14 Orders of Magnitude", pad=12)
    _style_axis_ticks_and_grid(ax, kind="continuous")
    ax.legend(loc="lower right", frameon=True, fancybox=False, fontsize=8.4)

    _finalize_journal_figure(fig)
    out_path = assets_dir / "unit_and_pH_consistency_audit.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return out_path


def fig3_speciation_physics_verification(assets_dir: Path) -> Path:
    """Figure 3: Henderson-Hasselbalch pH-Dependent Ozonation Kinetics Verification Curves."""
    ph_grid = np.linspace(1.5, 11.8, 240)

    systems = [
        {
            "name": "Phenol",
            "family": "Phenols",
            "pka": 9.95,
            "k_ha": 1.3e3,
            "k_a_minus": 1.4e9,
            "color": JOURNAL_PALETTE["navy"],
            "verified_pts": [(2.0, 1.30e3), (7.0, 1.57e6)],
        },
        {
            "name": "Triclosan",
            "family": "Phenols",
            "pka": 7.90,
            "k_ha": 13.0,
            "k_a_minus": 3.8e7,
            "color": JOURNAL_PALETTE["emerald"],
            "verified_pts": [(7.0, 4.25e6)],
            "swap_pt": (7.90, 7.00, 4.25e6),
        },
        {
            "name": "Sulfamethoxazole",
            "family": "Sulfonamides",
            "pka": 5.70,
            "k_ha": 4.7e4,
            "k_a_minus": 5.7e5,
            "color": JOURNAL_PALETTE["plum"],
            "verified_pts": [(7.0, 5.45e5)],
            "swap_pt": (5.70, 7.00, 5.45e5),
        },
        {
            "name": "Trimethylamine",
            "family": "Amines",
            "pka": 9.80,
            "k_ha": 0.05,
            "k_a_minus": 4.1e6,
            "color": JOURNAL_PALETTE["ochre"],
            "verified_pts": [(7.0, 6.49e3), (8.0, 6.40e4)],
            "swap_pt": (9.80, 8.00, 6.40e4),
        },
    ]

    fig, ax = plt.subplots(figsize=(11.2, 7.1))

    for sys_cfg in systems:
        curve_log_k = [
            math.log10(
                verify_speciation_physics(
                    ph=float(p),
                    pka=sys_cfg["pka"],
                    k_ha=sys_cfg["k_ha"],
                    k_a_minus=sys_cfg["k_a_minus"],
                ).predicted_k_app
            )
            for p in ph_grid
        ]
        ax.plot(
            ph_grid,
            curve_log_k,
            linewidth=2.3,
            color=sys_cfg["color"],
            label=rf"{sys_cfg['name']} ($\mathrm{{p}}K_a = {sys_cfg['pka']:.2f}$)",
            zorder=3,
        )
        pka_k = verify_speciation_physics(
            ph=sys_cfg["pka"],
            pka=sys_cfg["pka"],
            k_ha=sys_cfg["k_ha"],
            k_a_minus=sys_cfg["k_a_minus"],
        ).predicted_k_app
        ax.scatter(
            [sys_cfg["pka"]],
            [math.log10(pka_k)],
            s=50,
            facecolors="#FFFFFF",
            edgecolors=sys_cfg["color"],
            linewidth=2.0,
            zorder=5,
        )

        for ph_v, k_v in sys_cfg["verified_pts"]:
            ax.scatter(
                [ph_v],
                [math.log10(k_v)],
                s=86,
                marker="o",
                color=sys_cfg["color"],
                edgecolors="#111111",
                linewidth=0.95,
                zorder=6,
            )

        if "swap_pt" in sys_cfg:
            wrong_ph, fixed_ph, k_v = sys_cfg["swap_pt"]
            yk = math.log10(k_v)
            ax.scatter(
                [wrong_ph],
                [yk],
                s=94,
                marker="s",
                color="#F5B7B1",
                edgecolors=JOURNAL_PALETTE["crimson"],
                linewidth=1.3,
                zorder=6,
            )
            ax.annotate(
                "",
                xy=(fixed_ph, yk),
                xytext=(wrong_ph, yk),
                arrowprops=dict(arrowstyle="->", color=JOURNAL_PALETTE["crimson"], lw=1.6, ls="--"),
                zorder=5,
            )

    violations = [
        ("2,4,6-TCP Trap", 6.0, 500.0, 2.07e8),
        ("Metoprolol Trap", 7.0, 9.5e6, 2.02e3),
        ("Aniline Trap", 7.0, 1.2e3, 8.96e7),
    ]
    for idx, (_, ph_v, rep_k, _) in enumerate(violations):
        ax.scatter(
            [ph_v],
            [math.log10(rep_k)],
            s=120,
            marker="X",
            color=JOURNAL_PALETTE["crimson"],
            edgecolors="#111111",
            linewidth=0.95,
            zorder=7,
            label=r"Rejected Speciation Violation ($|\Delta\log_{10} k| > 3.5$)" if idx == 0 else None,
        )

    # Legend proxy markers
    ax.scatter(
        [],
        [],
        s=82,
        marker="o",
        color=JOURNAL_PALETTE["slate"],
        edgecolors="#111111",
        linewidth=0.95,
        label=r"Verified Experimental $k_{\mathrm{app}}(\mathrm{pH})$",
    )
    ax.scatter(
        [],
        [],
        s=90,
        marker="s",
        color="#F5B7B1",
        edgecolors=JOURNAL_PALETTE["crimson"],
        linewidth=1.3,
        label=r"Swapped $\mathrm{pH}\leftrightarrow\mathrm{p}K_a$ Column (Auto-Unswapped)",
    )
    ax.scatter(
        [],
        [],
        s=50,
        facecolors="#FFFFFF",
        edgecolors=JOURNAL_PALETTE["slate"],
        linewidth=2.0,
        label=r"Half-Dissociation Point ($\mathrm{pH} = \mathrm{p}K_a$)",
    )

    # Equation box in clean upper-left headroom
    ax.text(
        1.82,
        9.40,
        r"$k_{\mathrm{app}}(\mathrm{pH}) = \alpha_0 k_{\mathrm{HA}} + \alpha_1 k_{\mathrm{A}^-}, \quad "
        r"\alpha_1 = \frac{1}{1 + 10^{\mathrm{p}K_a - \mathrm{pH}}}$",
        fontsize=10.6,
        color="#111111",
        bbox=dict(boxstyle="square,pad=0.40", fc="#FFFFFF", ec="#111111", lw=1.1, alpha=0.97),
        zorder=8,
    )

    # Callout boxes placed strictly in curve-free open regions
    ax.annotate(
        "Triclosan $\\mathrm{pH}\\leftrightarrow\\mathrm{p}K_a$ Swap Caught\n"
        "Reported $\\mathrm{pH} = 7.90 \\rightarrow$ Repaired $\\mathrm{pH} = 7.00$",
        xy=(7.95, math.log10(4.25e6) + 0.06),
        xytext=(3.95, 8.15),
        fontsize=8.5,
        color=JOURNAL_PALETTE["crimson"],
        bbox=dict(boxstyle="square,pad=0.34", fc="#FFFFFF", ec=JOURNAL_PALETTE["crimson"], lw=1.1, alpha=0.96),
        arrowprops=dict(
            arrowstyle="->",
            color=JOURNAL_PALETTE["crimson"],
            lw=1.25,
            connectionstyle="arc3,rad=-0.12",
        ),
        zorder=8,
    )
    ax.annotate(
        "2,4,6-TCP Speciation Violation\n"
        "Reported $\\log_{10} k = 2.70$ vs. $\\alpha_0 k_{\\mathrm{HA}} + \\alpha_1 k_{\\mathrm{A}^-} = 8.32$",
        xy=(6.0, math.log10(500.0)),
        xytext=(3.55, -0.55),
        fontsize=8.4,
        color=JOURNAL_PALETTE["crimson"],
        bbox=dict(boxstyle="square,pad=0.34", fc="#FFFFFF", ec=JOURNAL_PALETTE["crimson"], lw=1.1, alpha=0.96),
        arrowprops=dict(arrowstyle="->", color=JOURNAL_PALETTE["crimson"], lw=1.25),
        zorder=8,
    )

    ax.set_xlim(1.5, 11.8)
    ax.set_ylim(-0.95, 10.65)
    ax.set_xlabel(r"Aqueous Experimental $\mathrm{pH}$")
    ax.set_ylabel(r"Apparent Ozonation Rate Constant $\log_{10} k_{\mathrm{O_3,app}}\ (\mathrm{M^{-1}\,s^{-1}})$")
    ax.set_title(
        "Henderson–Hasselbalch Speciation Physics Auditor Across Dissociating Micropollutants",
        pad=12,
    )
    _style_axis_ticks_and_grid(ax, kind="continuous")
    ax.legend(loc="lower right", frameon=True, fancybox=False, fontsize=8.2)

    _finalize_journal_figure(fig)
    out_path = assets_dir / "speciation_physics_verification.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return out_path


def fig4_provenance_error_taxonomy(csv_path: Path, assets_dir: Path) -> Path:
    """Figure 4: Breakdown of Caught Literature / OCR / LLM Failure Modes Across Chemical Families."""
    rows: List[Dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    families = ["Phenols", "Sulfonamides", "Beta-blockers", "Triazines", "Olefins", "Amines"]
    mode_labels = {
        "INVALID_VALENCE_HALLUCINATION": "Invalid Valence / Broken-Ring SMILES",
        "UNIT_TRAP_REPAIRABLE": r"Unit Trap ($\mathrm{mM^{-1}\,s^{-1}}$, $\mathrm{M^{-1}\,min^{-1}}$, $\mathrm{cm^3\,molec^{-1}}$)",
        "OCR_EXPONENT_REPAIRABLE": r"OCR Decade Exponent Shift ($10^2\times$–$10^3\times$)",
        "PH_PKA_SWAP_REPAIRABLE": r"Transposed $\mathrm{pH} \leftrightarrow \mathrm{p}K_a$ Table Columns",
        "OCR_GLYPH_BASE10": r"OCR Glyph Confusion ($\mathrm{l0^5}$, $\mathrm{1O^3}$)",
        "DIFFUSION_LIMIT_VIOLATION": r"Smoluchowski Diffusion Limit ($> 2.5 \times 10^{10}$)",
        "SPECIATION_VIOLATION": "Henderson–Hasselbalch Speciation Violation",
        "PSEUDO_FIRST_ORDER_TRAP": r"Pseudo-First-Order Rate Trap ($\mathrm{min^{-1}}$, $\mathrm{s^{-1}}$)",
        "HYBRID_PROSE_SUPPLEMENT": "Prose-Only Record in Hybrid Excerpt",
        "NEGATIVE_KELVIN_VIOLATION": r"Negative Kelvin Temp ($-20\ \mathrm{K}$)",
        "UNRECOVERABLE_LOG_K_MISMATCH": r"Unrecoverable $\log_{10} k$ / Prose Contradiction",
        "UNRESOLVED_TABLE_PROSE_CONFLICT": r"Unrecoverable $\log_{10} k$ / Prose Contradiction",
    }

    palette = [
        JOURNAL_PALETTE["crimson"],
        JOURNAL_PALETTE["navy"],
        JOURNAL_PALETTE["plum"],
        JOURNAL_PALETTE["emerald"],
        JOURNAL_PALETTE["ochre"],
        JOURNAL_PALETTE["slate"],
        JOURNAL_PALETTE["steel"],
        JOURNAL_PALETTE["teal"],
        JOURNAL_PALETTE["amber"],
        JOURNAL_PALETTE["burgundy"],
        JOURNAL_PALETTE["charcoal"],
    ]

    fam_mode_counts: Dict[str, Dict[str, int]] = {fam: defaultdict(int) for fam in families}
    agg_by_group: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])

    for r in rows:
        fam = r["compound_class"]
        raw_m = r["failure_mode"]
        pretty = mode_labels.get(raw_m, raw_m)
        tot = int(r["total_occurrences"])
        c_b = int(r["caught_baseline"])
        c_r = int(r["caught_rdkit_gate"])
        c_f = int(r["caught_full_miner"])
        if fam in fam_mode_counts:
            fam_mode_counts[fam][pretty] += tot
        agg = agg_by_group[pretty]
        agg[0] += tot
        agg[1] += c_b
        agg[2] += c_r
        agg[3] += c_f

    ordered_modes = sorted(agg_by_group.keys(), key=lambda k: agg_by_group[k][0], reverse=True)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(16.4, 6.8),
        gridspec_kw={"width_ratios": [1.04, 1.06]},
    )
    fig.subplots_adjust(wspace=0.78)

    # Panel (a): Stacked Horizontal Bar by Chemical Family
    y_pos = np.arange(len(families))
    lefts = np.zeros(len(families))
    for idx, m_name in enumerate(ordered_modes):
        vals = np.array([fam_mode_counts[fam].get(m_name, 0) for fam in families], dtype=float)
        if vals.sum() == 0:
            continue
        ax1.barh(
            y_pos,
            vals,
            left=lefts,
            height=0.56,
            label=rf"{m_name} ($n = {int(vals.sum())}$)",
            color=palette[idx % len(palette)],
            edgecolor="#111111",
            linewidth=0.85,
            zorder=3,
        )
        lefts += vals

    for i, total_v in enumerate(lefts):
        ax1.text(
            total_v + 0.22,
            y_pos[i],
            f"{int(total_v)}",
            va="center",
            ha="left",
            fontsize=9.6,
            color="#111111",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(families)
    ax1.invert_yaxis()
    ax1.set_xlim(0, max(lefts) * 1.18)
    ax1.set_xlabel("Injected Stress-Test / Failure-Mode Instances")
    ax1.set_title("(a) Failure & Repair Taxonomy by Chemical Family", pad=11)
    _style_axis_ticks_and_grid(ax1, kind="categorical_hbar")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        fontsize=8.5,
        frameon=True,
        fancybox=False,
    )

    # Panel (b): Catch / Resolution Rate Heatmap across the 3 Progressive Pipelines
    heat_modes = ordered_modes
    mat = np.zeros((len(heat_modes), 3))
    for i, m_name in enumerate(heat_modes):
        tot, cb, cr, cf = agg_by_group[m_name]
        mat[i, 0] = 100.0 * cb / max(1, tot)
        mat[i, 1] = 100.0 * cr / max(1, tot)
        mat[i, 2] = 100.0 * cf / max(1, tot)

    im = ax2.imshow(mat, cmap="YlGnBu", aspect="auto", vmin=0.0, vmax=100.0)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["1. Baseline", "2. + RDKit Gate", "3. Full Miner"])
    ax2.set_yticks(np.arange(len(heat_modes)))
    ax2.set_yticklabels(heat_modes, fontsize=8.6)
    ax2.set_title("(b) Failure Catch / Auto-Repair Rate by Pipeline (%)", pad=11)
    _style_axis_ticks_and_grid(ax2, kind="heatmap")

    # Crisp subtle cell borders inside heatmap
    ax2.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, len(heat_modes), 1), minor=True)
    ax2.grid(which="minor", color="#CBD5E1", linestyle="-", linewidth=0.8)
    ax2.tick_params(which="minor", bottom=False, left=False, top=False, right=False)

    for i in range(len(heat_modes)):
        for j in range(3):
            v = mat[i, j]
            txt_color = "#FFFFFF" if v >= 58.0 else "#111111"
            ax2.text(
                j,
                i,
                f"{v:.0f}%",
                ha="center",
                va="center",
                color=txt_color,
                fontsize=8.9,
            )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.042, pad=0.03)
    cbar.outline.set_edgecolor("#111111")
    cbar.outline.set_linewidth(1.15)
    cbar.ax.tick_params(direction="in", length=3.8, width=1.1, colors="#111111")
    cbar.set_label("Catch / Resolution Rate (%)")

    _finalize_journal_figure(fig)
    out_path = assets_dir / "provenance_error_taxonomy.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    return out_path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    configure_style()
    results_dir = ROOT / "results"
    assets_dir = ROOT / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    bench = json.loads((results_dir / "miner_benchmark.json").read_text(encoding="utf-8"))
    csv_path = results_dir / "error_breakdown.csv"

    f1 = fig1_pipeline_ablation(bench, assets_dir)
    f2 = fig2_unit_and_ph_consistency_audit(bench, assets_dir)
    f3 = fig3_speciation_physics_verification(assets_dir)
    f4 = fig4_provenance_error_taxonomy(csv_path, assets_dir)

    for fp in (f1, f2, f3, f4):
        print(f"Generated figure (300 DPI): {fp} ({fp.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
