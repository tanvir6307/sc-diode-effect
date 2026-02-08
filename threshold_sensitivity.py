"""
Threshold sensitivity analysis for the superconducting diode effect.

Re-analyzes existing IV curve data with different voltage thresholds
to demonstrate robustness of the reported Ic and eta values.
No new simulations required — uses saved results.json files.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def find_Ic_from_iv(I_values, V_values, abs_threshold=1e-3, rel_threshold=10, v_floor=1e-4):
    """
    Re-extract critical current from saved IV data using given thresholds.
    
    Mirrors the switching criterion from diode_effect.py:
      voltage_increase > abs_threshold  OR  (relative_increase > rel_threshold AND V > v_floor)
    """
    I_arr = np.array(I_values)
    V_arr = np.array(V_values)
    
    if len(I_arr) == 0:
        return 0.0
    
    baseline_V = V_arr[0]
    
    for idx in range(1, len(I_arr)):
        voltage_increase = V_arr[idx] - baseline_V
        relative_increase = V_arr[idx] / max(baseline_V, 1e-10)
        
        if voltage_increase > abs_threshold or (relative_increase > rel_threshold and V_arr[idx] > v_floor):
            return I_arr[idx]
    
    # If no switch detected, return max current
    return I_arr[-1]


def analyze_threshold_sensitivity(data_dirs, param_name, param_values, thresholds, output_dir):
    """
    Analyze how Ic+, Ic-, and eta depend on the voltage threshold.
    
    Args:
        data_dirs: list of paths to results folders
        param_name: "applied_field" or "notch_depth"
        param_values: parameter values for labeling
        thresholds: dict mapping label -> (abs_thresh, rel_thresh, v_floor)
        output_dir: where to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    
    for label, (abs_th, rel_th, v_fl) in thresholds.items():
        Ic_plus_list = []
        Ic_minus_list = []
        eta_list = []
        
        for d, pval in zip(data_dirs, param_values):
            rpath = Path(d) / "results.json"
            if not rpath.exists():
                logger.warning(f"Missing {rpath}")
                Ic_plus_list.append(np.nan)
                Ic_minus_list.append(np.nan)
                eta_list.append(np.nan)
                continue
            
            with open(rpath) as f:
                res = json.load(f)
            
            Ic_p = find_Ic_from_iv(
                res["diagnostics_plus"]["I"],
                res["diagnostics_plus"]["V"],
                abs_threshold=abs_th, rel_threshold=rel_th, v_floor=v_fl,
            )
            Ic_m = find_Ic_from_iv(
                res["diagnostics_minus"]["I"],
                res["diagnostics_minus"]["V"],
                abs_threshold=abs_th, rel_threshold=rel_th, v_floor=v_fl,
            )
            eta = (Ic_p - Ic_m) / (Ic_p + Ic_m) if (Ic_p + Ic_m) > 0 else 0.0
            
            Ic_plus_list.append(Ic_p)
            Ic_minus_list.append(Ic_m)
            eta_list.append(eta)
        
        all_results[label] = {
            "Ic_plus": Ic_plus_list,
            "Ic_minus": Ic_minus_list,
            "eta": eta_list,
        }
    
    return all_results


def main():
    base = Path("outputs_diode")
    out = base / "threshold_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    
    # Define threshold variations
    # Baseline: abs=1e-3, rel=10, v_floor=1e-4
    # "Factor of 2" means halving and doubling the absolute threshold
    thresholds = {
        r"$V_\mathrm{th} = 5\times10^{-4}$": (5e-4, 10, 5e-5),    # 0.5x baseline
        r"$V_\mathrm{th} = 1\times10^{-3}$ (baseline)": (1e-3, 10, 1e-4),  # 1x (baseline)
        r"$V_\mathrm{th} = 2\times10^{-3}$": (2e-3, 10, 2e-4),    # 2x baseline
        r"$V_\mathrm{th} = 5\times10^{-3}$": (5e-3, 10, 5e-4),    # 5x baseline
    }
    
    # =============================================
    # 1. Field sweep threshold sensitivity
    # =============================================
    field_values = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
    field_dirs = [base / f"applied_field_{b:.3f}" for b in field_values]
    
    logger.info("Analyzing field sweep threshold sensitivity...")
    field_results = analyze_threshold_sensitivity(
        field_dirs, "applied_field", field_values, thresholds, out
    )
    
    # =============================================
    # 2. Depth sweep threshold sensitivity
    # =============================================
    depth_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    depth_dirs = [base / f"notch_depth_{d:.3f}" for d in depth_values]
    
    logger.info("Analyzing depth sweep threshold sensitivity...")
    depth_results = analyze_threshold_sensitivity(
        depth_dirs, "notch_depth", depth_values, thresholds, out
    )
    
    # =============================================
    # 3. Generate publication figure
    # =============================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    colors = ['#2196F3', '#000000', '#F44336', '#FF9800']
    linestyles = ['--', '-', '--', ':']
    markers = ['s', 'o', 'D', '^']
    
    # (a) Field sweep: Ic+ and Ic-
    ax = axes[0, 0]
    for i, (label, data) in enumerate(field_results.items()):
        ax.plot(field_values, data["Ic_plus"], color=colors[i], linestyle=linestyles[i],
                marker=markers[i], markersize=5, label=label + r" ($I_c^+$)")
        ax.plot(field_values, data["Ic_minus"], color=colors[i], linestyle=linestyles[i],
                marker=markers[i], markersize=5, fillstyle='none')
    ax.set_xlabel(r"$B/B_{c2}$")
    ax.set_ylabel(r"$I_c$ ($J_0$)")
    ax.set_title("(a) Field sweep: critical currents")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # (b) Field sweep: eta
    ax = axes[0, 1]
    for i, (label, data) in enumerate(field_results.items()):
        ax.plot(field_values, [e * 100 for e in data["eta"]], color=colors[i],
                linestyle=linestyles[i], marker=markers[i], markersize=6, label=label)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(r"$B/B_{c2}$")
    ax.set_ylabel(r"$\eta$ (%)")
    ax.set_title(r"(b) Field sweep: diode efficiency")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    # (c) Depth sweep: Ic+ and Ic-
    ax = axes[1, 0]
    for i, (label, data) in enumerate(depth_results.items()):
        ax.plot(depth_values, data["Ic_plus"], color=colors[i], linestyle=linestyles[i],
                marker=markers[i], markersize=5, label=label + r" ($I_c^+$)")
        ax.plot(depth_values, data["Ic_minus"], color=colors[i], linestyle=linestyles[i],
                marker=markers[i], markersize=5, fillstyle='none')
    ax.set_xlabel(r"$d_\mathrm{notch}/\xi$")
    ax.set_ylabel(r"$I_c$ ($J_0$)")
    ax.set_title("(c) Depth sweep: critical currents")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    # (d) Depth sweep: eta
    ax = axes[1, 1]
    for i, (label, data) in enumerate(depth_results.items()):
        ax.plot(depth_values, [e * 100 for e in data["eta"]], color=colors[i],
                linestyle=linestyles[i], marker=markers[i], markersize=6, label=label)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.set_xlabel(r"$d_\mathrm{notch}/\xi$")
    ax.set_ylabel(r"$\eta$ (%)")
    ax.set_title(r"(d) Depth sweep: diode efficiency")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out / "threshold_sensitivity.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figure saved: {out / 'threshold_sensitivity.png'}")
    
    # =============================================
    # 4. Print summary table
    # =============================================
    print("\n" + "=" * 80)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    print("\n--- FIELD SWEEP ---")
    print(f"{'Threshold':<45} ", end="")
    for b in field_values:
        print(f"  B={b:.2f}", end="")
    print()
    for label, data in field_results.items():
        short = label.replace(r"$V_\mathrm{th} = ", "").replace("$", "").replace(r"\times", "x")
        print(f"  {short:<43} ", end="")
        for e in data["eta"]:
            print(f"  {e*100:+6.1f}", end="")
        print()
    
    print("\n--- DEPTH SWEEP ---")
    print(f"{'Threshold':<45} ", end="")
    for d in depth_values:
        print(f"  d={d:.1f}", end="")
    print()
    for label, data in depth_results.items():
        short = label.replace(r"$V_\mathrm{th} = ", "").replace("$", "").replace(r"\times", "x")
        print(f"  {short:<43} ", end="")
        for e in data["eta"]:
            print(f"  {e*100:+6.1f}", end="")
        print()
    
    # =============================================
    # 5. Compute max deviation from baseline
    # =============================================
    baseline_label = r"$V_\mathrm{th} = 1\times10^{-3}$ (baseline)"
    print("\n--- MAX DEVIATION FROM BASELINE ---")
    for label, data in field_results.items():
        if label == baseline_label:
            continue
        base_eta = np.array(field_results[baseline_label]["eta"])
        this_eta = np.array(data["eta"])
        max_dev = np.nanmax(np.abs(this_eta - base_eta)) * 100
        short = label.replace(r"$V_\mathrm{th} = ", "").replace("$", "").replace(r"\times", "x")
        print(f"  Field sweep, {short}: max |Δη| = {max_dev:.1f} pp")
    
    for label, data in depth_results.items():
        if label == baseline_label:
            continue
        base_eta = np.array(depth_results[baseline_label]["eta"])
        this_eta = np.array(data["eta"])
        max_dev = np.nanmax(np.abs(this_eta - base_eta)) * 100
        short = label.replace(r"$V_\mathrm{th} = ", "").replace("$", "").replace(r"\times", "x")
        print(f"  Depth sweep, {short}: max |Δη| = {max_dev:.1f} pp")
    
    print("\n" + "=" * 80)
    
    # Save numerical results
    save_data = {
        "thresholds": {k: v for k, v in zip(
            ["0.5x", "1x (baseline)", "2x", "5x"],
            thresholds.values()
        )},
        "field_sweep": {
            "param_values": field_values,
        },
        "depth_sweep": {
            "param_values": depth_values,
        },
    }
    for label, data in field_results.items():
        key = label.split("=")[1].strip().rstrip("$").replace(" (baseline)", "")
        save_data["field_sweep"][key] = {
            "eta": data["eta"],
            "Ic_plus": data["Ic_plus"],
            "Ic_minus": data["Ic_minus"],
        }
    for label, data in depth_results.items():
        key = label.split("=")[1].strip().rstrip("$").replace(" (baseline)", "")
        save_data["depth_sweep"][key] = {
            "eta": data["eta"],
            "Ic_plus": data["Ic_plus"],
            "Ic_minus": data["Ic_minus"],
        }
    
    with open(out / "threshold_sensitivity.json", "w") as f:
        json.dump(save_data, f, indent=2)
    logger.info(f"Data saved: {out / 'threshold_sensitivity.json'}")


if __name__ == "__main__":
    main()
