#!/usr/bin/env python3
"""
Mesh Convergence Study for the Superconducting Diode Effect
============================================================

Tests the sensitivity of I_c^+, I_c^-, and diode efficiency eta to the
finite-volume mesh resolution by running identical simulations at several
values of max_edge_length.

All other parameters are held fixed at their baseline values:
    W = 2 xi,  L = 8 xi,  notch_depth = 1.0 xi,  B = 0.10 B_c2,
    current_max = 0.05 J_0,  current_steps = 60,  solve_time = 100 tau_0.

Outputs:
    outputs_diode/mesh_convergence/convergence_results.json
    outputs_diode/mesh_convergence/mesh_convergence.png
    outputs_diode/mesh_convergence/mesh_convergence_table.txt

Usage:
    python mesh_convergence.py
"""

import json
import logging
import time
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse device and Ic infrastructure from diode_effect
from diode_effect import (
    DiodeConfig,
    create_asymmetric_wire,
    find_critical_current,
    ensure_dir,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Mesh sizes to test (from coarse to fine).
# 0.25 is known to be too coarse (solver instability for some configs).
MESH_SIZES = [0.50, 0.40, 0.30, 0.25, 0.20, 0.15]

OUTPUT_DIR = Path("outputs_diode") / "mesh_convergence"


def baseline_config(max_edge_length: float) -> DiodeConfig:
    """Return the baseline DiodeConfig with the given mesh size."""
    return DiodeConfig(
        wire_length=8.0,
        wire_width=2.0,
        notch_depth=1.0,
        notch_width=1.0,
        n_notches=2,
        notch_spacing=2.0,
        current_max=0.05,
        current_steps=60,
        solve_time=100.0,
        dt=0.1,
        thickness=0.1,
        lambda_london=2.0,
        max_edge_length=max_edge_length,
        applied_field=0.1,
        output_dir=str(OUTPUT_DIR),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_convergence_study():
    out = ensure_dir(str(OUTPUT_DIR))

    results = []

    logger.info("=" * 65)
    logger.info("  MESH CONVERGENCE STUDY")
    logger.info("=" * 65)

    for mesh_size in MESH_SIZES:
        logger.info("")
        logger.info("-" * 65)
        logger.info(f"  max_edge_length = {mesh_size:.2f} xi")
        logger.info("-" * 65)

        config = baseline_config(mesh_size)

        # Build device
        try:
            device = create_asymmetric_wire(config)
        except Exception as e:
            logger.error(f"  Mesh generation failed: {e}")
            results.append({
                "mesh_size": mesh_size,
                "n_sites": None,
                "n_elements": None,
                "Ic_plus": None,
                "Ic_minus": None,
                "eta": None,
                "time_s": None,
                "error": str(e),
            })
            continue

        n_sites = len(device.mesh.sites)
        n_elements = len(device.mesh.elements)
        logger.info(f"  Mesh: {n_sites} sites, {n_elements} elements")

        # --- I_c^+ ---
        t0 = time.time()
        try:
            Ic_plus, diag_plus, _ = find_critical_current(
                device, config, current_direction=+1,
                field=config.applied_field, output_dir=None,
            )
        except Exception as e:
            logger.error(f"  I_c^+ solver failed: {e}")
            Ic_plus = None
            diag_plus = None

        # --- I_c^- ---
        try:
            Ic_minus, diag_minus, _ = find_critical_current(
                device, config, current_direction=-1,
                field=config.applied_field, output_dir=None,
            )
        except Exception as e:
            logger.error(f"  I_c^- solver failed: {e}")
            Ic_minus = None
            diag_minus = None

        elapsed = time.time() - t0

        if Ic_plus is not None and Ic_minus is not None:
            eta = (Ic_plus - Ic_minus) / (Ic_plus + Ic_minus) if (Ic_plus + Ic_minus) > 0 else 0.0
        else:
            eta = None

        logger.info(f"  I_c^+ = {Ic_plus}, I_c^- = {Ic_minus}, eta = {eta}")
        logger.info(f"  Wall time: {elapsed:.1f} s")

        results.append({
            "mesh_size": mesh_size,
            "n_sites": n_sites,
            "n_elements": n_elements,
            "Ic_plus": float(Ic_plus) if Ic_plus is not None else None,
            "Ic_minus": float(Ic_minus) if Ic_minus is not None else None,
            "eta": float(eta) if eta is not None else None,
            "eta_pct": float(eta * 100) if eta is not None else None,
            "time_s": round(elapsed, 1),
            "error": None,
        })

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------
    with open(out / "convergence_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {out / 'convergence_results.json'}")

    # ------------------------------------------------------------------
    # Print table
    # ------------------------------------------------------------------
    valid = [r for r in results if r["eta"] is not None]
    if not valid:
        logger.error("No valid results obtained. Stopping.")
        return

    header = (
        f"{'h (xi)':>8}  {'Sites':>7}  {'Elems':>7}  "
        f"{'Ic+':>9}  {'Ic-':>9}  {'eta (%)':>9}  {'Time (s)':>9}"
    )
    sep = "-" * len(header)
    table_lines = [sep, header, sep]
    for r in results:
        if r["eta"] is not None:
            line = (
                f"{r['mesh_size']:>8.2f}  {r['n_sites']:>7d}  {r['n_elements']:>7d}  "
                f"{r['Ic_plus']:>9.5f}  {r['Ic_minus']:>9.5f}  {r['eta_pct']:>9.2f}  "
                f"{r['time_s']:>9.1f}"
            )
        else:
            line = f"{r['mesh_size']:>8.2f}  {'FAILED':>7}  {'':>7}  {'':>9}  {'':>9}  {'':>9}  {'':>9}"
        table_lines.append(line)
    table_lines.append(sep)

    table_str = "\n".join(table_lines)
    print("\n" + table_str + "\n")

    with open(out / "mesh_convergence_table.txt", "w") as f:
        f.write(table_str + "\n")

    # ------------------------------------------------------------------
    # Convergence plot
    # ------------------------------------------------------------------
    plot_convergence(valid, out / "mesh_convergence.png")

    logger.info("Mesh convergence study complete.")


def plot_convergence(results, filepath):
    """Plot I_c^+, I_c^-, and eta vs mesh size."""

    h_vals = [r["mesh_size"] for r in results]
    Ic_p = [r["Ic_plus"] for r in results]
    Ic_m = [r["Ic_minus"] for r in results]
    eta = [r["eta_pct"] for r in results]
    n_sites = [r["n_sites"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)

    # --- Panel (a): I_c^+ and I_c^- vs h ---
    ax = axes[0]
    ax.plot(h_vals, Ic_p, "o-", color="tab:blue", label=r"$I_c^+$", markersize=7)
    ax.plot(h_vals, Ic_m, "s-", color="tab:red", label=r"$I_c^-$", markersize=7)
    ax.set_xlabel(r"Max edge length $h\;[\xi]$")
    ax.set_ylabel(r"Critical current $[J_0]$")
    ax.set_title("(a) Critical currents")
    ax.legend(fontsize=9)
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    # --- Panel (b): eta vs h ---
    ax = axes[1]
    ax.plot(h_vals, eta, "D-", color="tab:green", markersize=7)
    ax.set_xlabel(r"Max edge length $h\;[\xi]$")
    ax.set_ylabel(r"Diode efficiency $\eta$ (%)")
    ax.set_title(r"(b) Efficiency $\eta$")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)
    # Highlight the baseline mesh
    if 0.20 in h_vals:
        idx_base = h_vals.index(0.20)
        ax.axhline(eta[idx_base], ls="--", color="gray", alpha=0.5)
        ax.annotate(
            f"Baseline ({eta[idx_base]:.1f}%)",
            xy=(0.20, eta[idx_base]),
            xytext=(0.30, eta[idx_base] + 3),
            fontsize=8, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # --- Panel (c): n_sites vs h (computational cost) ---
    ax = axes[2]
    ax.plot(h_vals, n_sites, "^-", color="tab:orange", markersize=7)
    ax.set_xlabel(r"Max edge length $h\;[\xi]$")
    ax.set_ylabel("Number of mesh sites")
    ax.set_title("(c) Mesh density")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.3)

    fig.suptitle("Mesh convergence study", fontsize=13, fontweight="bold")
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Plot saved to {filepath}")


if __name__ == "__main__":
    run_convergence_study()
