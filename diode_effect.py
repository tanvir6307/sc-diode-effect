"""
Superconducting Diode Effect Study
===================================

This script studies the superconducting diode effect (SDE) in asymmetric
nanowire geometries using time-dependent Ginzburg-Landau (TDGL) simulations.

The diode effect arises when spatial inversion symmetry is broken, leading to:
    I_c^+ ≠ I_c^-

Diode efficiency is defined as:
    η = (I_c^+ - I_c^-) / (I_c^+ + I_c^-)

Physics Background:
------------------
In a symmetric wire, vortices/phase slips nucleate identically for both
current directions. With asymmetric geometry (e.g., triangular notches),
vortex entry barriers differ for +I vs -I, creating non-reciprocal transport.

Author: Tanvir Hassan
Date: February 2026
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection
import matplotlib.tri as mtri
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import json
import time
import logging

# TDGL imports
import tdgl
from tdgl.geometry import box, circle
from tdgl import Device, Layer, Polygon
from tdgl.solver.solve import solve
from tdgl.solver.options import SolverOptions

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class DiodeConfig:
    """Configuration for diode effect study."""
    # Geometry parameters
    wire_length: float = 10.0       # Total wire length (ξ units)
    wire_width: float = 2.0         # Wire width (ξ units)
    
    # Asymmetric feature parameters
    notch_depth: float = 0.5        # Depth of triangular notch (into wire)
    notch_width: float = 1.0        # Base width of triangle
    notch_position: float = 0.0     # x-position of notch center (0 = middle)
    n_notches: int = 1              # Number of notches
    notch_spacing: float = 2.0      # Spacing between notches
    
    # Simulation parameters
    current_max: float = 0.15       # Maximum current to sweep
    current_steps: int = 30         # Number of current steps
    solve_time: float = 100.0       # Time to run at each current
    dt: float = 0.1                 # Time step
    
    # Film parameters  
    thickness: float = 0.1          # Film thickness (ξ units)
    lambda_london: float = 2.0      # London penetration depth (ξ units)
    
    # Mesh
    max_edge_length: float = 0.2    # Mesh resolution
    
    # Applied field (optional - enhances diode effect)
    applied_field: float = 0.0      # External magnetic field
    
    # Output
    output_dir: str = "outputs_diode"


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_asymmetric_wire(config: DiodeConfig) -> Device:
    """
    Create a superconducting wire with asymmetric triangular notches.
    
    The notches are on ONE side only (top), breaking inversion symmetry.
    This creates different vortex entry barriers for +I vs -I.
    
    Geometry:
    
        ←───────────────────────────→  wire_length
        
        ┌──────────────────────────┐  ↑
        │                          │  │ wire_width
        │    ╱╲    ╱╲    ╱╲       │  │
        │   ╱  ╲  ╱  ╲  ╱  ╲      │  │
        └──────────────────────────┘  ↓
        
        Source                    Drain
        (left)                   (right)
    """
    L = config.wire_length
    W = config.wire_width
    
    # Main wire outline (start from bottom-left, go counter-clockwise)
    # Bottom edge
    points = [
        (-L/2, -W/2),  # Bottom-left
        (L/2, -W/2),   # Bottom-right
        (L/2, W/2),    # Top-right
    ]
    
    # Top edge with triangular notches (going right to left)
    if config.n_notches > 0:
        # Calculate notch positions
        if config.n_notches == 1:
            notch_centers = [config.notch_position]
        else:
            # Distribute notches symmetrically around center
            total_span = (config.n_notches - 1) * config.notch_spacing
            start = config.notch_position - total_span / 2
            notch_centers = [start + i * config.notch_spacing for i in range(config.n_notches)]
        
        # Sort from right to left (since we're going right to left on top edge)
        notch_centers = sorted(notch_centers, reverse=True)
        
        # Build top edge with notches
        current_x = L/2
        for nc in notch_centers:
            notch_left = nc - config.notch_width / 2
            notch_right = nc + config.notch_width / 2
            notch_tip_y = W/2 - config.notch_depth
            
            # From current position to right edge of notch
            if current_x > notch_right:
                points.append((notch_right, W/2))
            
            # Notch: down to tip, up to left edge
            points.append((nc, notch_tip_y))  # Tip of triangle
            points.append((notch_left, W/2))  # Left edge of notch
            
            current_x = notch_left
        
        # From last notch to top-left corner
        if current_x > -L/2:
            points.append((-L/2, W/2))
    else:
        # No notches - simple top edge
        points.append((-L/2, W/2))
    
    # Create the wire polygon
    wire_polygon = Polygon("film", points=points)
    
    # Create the layer
    layer = Layer(
        london_lambda=config.lambda_london,
        thickness=config.thickness,
        coherence_length=1.0,  # ξ = 1 in our units
    )
    
    # Create terminals using Polygon with box coordinates
    source = Polygon(points=box(config.max_edge_length * 0.5, W * 0.8, center=(-L/2, 0))).set_name("source")
    drain = Polygon(points=box(config.max_edge_length * 0.5, W * 0.8, center=(L/2, 0))).set_name("drain")
    
    # Create device with terminals
    device = Device(
        "asymmetric_wire",
        layer=layer,
        film=wire_polygon,
        terminals=[source, drain],
        probe_points=[(-L/4, 0), (L/4, 0)],  # Voltage probes
    )
    
    # Generate mesh
    device.make_mesh(max_edge_length=config.max_edge_length, smooth=100)
    
    return device


def create_ratchet_wire(config: DiodeConfig) -> Device:
    """
    Create a wire with sawtooth/ratchet profile (alternative asymmetric geometry).
    
    This creates an even stronger asymmetry - good for enhanced diode effect.
    
    Geometry (side view of one tooth):
    
         ╱│
        ╱ │
       ╱  │
      ╱   │
     ─────┘
     
    Vortices slide easily down the gradual slope but face a barrier at the cliff.
    """
    L = config.wire_length
    W = config.wire_width
    
    # Parameters for ratchet teeth
    tooth_width = config.notch_spacing if config.notch_spacing > 0 else 1.5
    tooth_height = config.notch_depth
    n_teeth = max(1, int(L * 0.6 / tooth_width))  # Fill 60% of wire
    
    # Start building outline
    points = [
        (-L/2, -W/2),  # Bottom-left
        (L/2, -W/2),   # Bottom-right
        (L/2, W/2),    # Top-right
    ]
    
    # Add ratchet teeth on top (right to left)
    teeth_start = (n_teeth - 1) * tooth_width / 2
    for i in range(n_teeth):
        x_right = teeth_start - i * tooth_width
        x_left = x_right - tooth_width
        
        # Gradual slope down (right to left = toward source)
        points.append((x_right, W/2))
        points.append((x_left, W/2 - tooth_height))
        # Cliff back up
        points.append((x_left, W/2))
    
    # Close to top-left
    points.append((-L/2, W/2))
    
    wire_polygon = Polygon("film", points=points)
    
    layer = Layer(
        london_lambda=config.lambda_london,
        thickness=config.thickness,
        coherence_length=1.0,
    )
    
    # Create terminals using Polygon with box coordinates
    source = Polygon(points=box(config.max_edge_length * 0.5, W * 0.8, center=(-L/2, 0))).set_name("source")
    drain = Polygon(points=box(config.max_edge_length * 0.5, W * 0.8, center=(L/2, 0))).set_name("drain")
    
    device = Device(
        "ratchet_wire",
        layer=layer,
        film=wire_polygon,
        terminals=[source, drain],
        probe_points=[(-L/4, 0), (L/4, 0)],
    )
    
    device.make_mesh(max_edge_length=config.max_edge_length, smooth=100)
    
    return device


def find_critical_current(
    device: Device,
    config: DiodeConfig,
    current_direction: int = 1,  # +1 or -1
    field: float = 0.0,
    output_dir: Optional[Path] = None,
) -> Tuple[float, Dict[str, Any], Optional[str]]:
    """
    Find the critical current by ramping current until a voltage appears.
    
    Args:
        device: TDGL device
        config: Configuration
        current_direction: +1 for positive current, -1 for negative
        field: Applied magnetic field
        output_dir: Directory to save snapshots (if provided)
        
    Returns:
        (I_c, diagnostics_dict, snapshot_path)
    """
    # Current values to test - start from a small fraction of max
    currents = np.linspace(config.current_max * 0.01, config.current_max, config.current_steps)
    if current_direction < 0:
        currents = -currents
    
    # Store results
    I_values = []
    V_values = []
    switched = False
    I_c = np.abs(currents[-1])  # Default to max if no switch
    snapshot_path = None
    
    # First, get baseline voltage at very low current
    baseline_V = 0.0
    
    for idx, I in enumerate(currents):
        # Define terminal currents
        terminal_currents = {"source": float(I), "drain": float(-I)}
        
        # Solver options
        options = SolverOptions(
            solve_time=config.solve_time,
            dt_init=config.dt,
            dt_max=config.dt * 10,  # Allow adaptive stepping up to 10x dt
            adaptive=True,
            save_every=100,
            progress_interval=0,
        )
        
        try:
            # Run solver
            solution = solve(
                device=device,
                options=options,
                terminal_currents=terminal_currents,
                applied_vector_potential=field,
            )
            
            # Get voltage from solution dynamics
            dynamics = solution.dynamics
            if dynamics is not None and dynamics.mu is not None:
                # Get time-averaged voltage between probe points 0 and 1
                # Average over last 50% of simulation to capture steady state
                tmin = config.solve_time * 0.5
                V_avg = np.abs(dynamics.mean_voltage(i=0, j=1, tmin=tmin))
            else:
                V_avg = 0.0
            
            I_values.append(np.abs(I))
            V_values.append(V_avg)
            
            # Set baseline from first measurement
            if idx == 0:
                baseline_V = V_avg
            
            # Log every few points
            if idx % 3 == 0 or idx < 3:
                logger.info(f"  I = {np.abs(I):.4f}, V = {V_avg:.6e}")
            
            # Check for switching - voltage must be significantly above baseline
            # Use both absolute and relative thresholds
            voltage_increase = V_avg - baseline_V
            relative_increase = V_avg / max(baseline_V, 1e-10)
            
            # Switching criteria:
            # 1. Absolute voltage > 1e-3 (in normalized units)
            # 2. OR voltage increased by more than 10x from baseline
            # 3. AND we're not at the first point
            if idx > 0 and not switched:
                if voltage_increase > 1e-3 or (relative_increase > 10 and V_avg > 1e-4):
                    switched = True
                    I_c = np.abs(I)
                    logger.info(f"*** Switch detected at I = {I_c:.4f} (direction: {'+' if current_direction > 0 else '-'})")
                    logger.info(f"    Baseline V = {baseline_V:.6e}, Current V = {V_avg:.6e}")
                    
                    # CAPTURE PHYSICS SNAPSHOT
                    # Re-run this specific step saving the output to a file
                    if output_dir:
                        direction_str = "plus" if current_direction > 0 else "minus"
                        fname = f"snapshot_Ic_{direction_str}.h5"
                        snapshot_path = str(output_dir / fname)
                        logger.info(f"    Re-running simulation to capture vortex physics at {snapshot_path}...")
                        
                        options.output_file = snapshot_path
                        options.save_every = 50  # Save frames for animation
                        solve(
                            device=device,
                            options=options,
                            terminal_currents=terminal_currents,
                            applied_vector_potential=field,
                        )
                    
                    # Continue to get more data points for I-V curve
                    # break  # Removed to collect full I-V curve
                
        except Exception as e:
            logger.warning(f"Solver failed at I = {I:.4f}: {e}")
            # If solver fails, likely we're above Ic
            if not switched:
                I_c = np.abs(I)
                switched = True
            break
    
    diagnostics = {
        "I_values": np.array(I_values),
        "V_values": np.array(V_values),
        "switched": switched,
    }
    
    return I_c, diagnostics, snapshot_path


def run_diode_study(config: DiodeConfig, geometry: str = "notch") -> Dict[str, Any]:
    """
    Run complete diode effect study.
    
    Args:
        config: Study configuration
        geometry: "notch" for triangular notches, "ratchet" for sawtooth
        
    Returns:
        Dictionary with all results
    """
    out = ensure_dir(config.output_dir)
    
    logger.info("=" * 60)
    logger.info("SUPERCONDUCTING DIODE EFFECT STUDY")
    logger.info("=" * 60)
    
    # Create device
    logger.info(f"Creating {geometry} geometry...")
    if geometry == "ratchet":
        device = create_ratchet_wire(config)
    else:
        device = create_asymmetric_wire(config)
    
    logger.info(f"Mesh: {len(device.mesh.sites)} nodes, {len(device.mesh.elements)} elements")
    
    # Plot geometry
    plot_geometry(device, out / "geometry.png", config)
    
    # Measure Ic for positive and negative currents
    logger.info("Measuring I_c^+ (positive current direction)...")
    t0 = time.time()
    Ic_plus, diag_plus, snap_plus = find_critical_current(
        device, config, current_direction=+1, 
        field=config.applied_field, output_dir=out
    )
    t_plus = time.time() - t0
    logger.info(f"I_c^+ = {Ic_plus:.5f} (took {t_plus:.1f}s)")
    
    logger.info("Measuring I_c^- (negative current direction)...")
    t0 = time.time()
    Ic_minus, diag_minus, snap_minus = find_critical_current(
        device, config, current_direction=-1, 
        field=config.applied_field, output_dir=out
    )
    t_minus = time.time() - t0
    logger.info(f"I_c^- = {Ic_minus:.5f} (took {t_minus:.1f}s)")
    
    # Calculate diode efficiency
    eta = (Ic_plus - Ic_minus) / (Ic_plus + Ic_minus) if (Ic_plus + Ic_minus) > 0 else 0.0
    
    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"I_c^+  = {Ic_plus:.5f}")
    logger.info(f"I_c^-  = {Ic_minus:.5f}")
    logger.info(f"ΔI_c   = {Ic_plus - Ic_minus:.5f}")
    logger.info(f"η (Diode Efficiency) = {eta:.4f} = {eta*100:.2f}%")
    logger.info("=" * 60)
    
    # Compile results
    results = {
        "config": {
            "geometry": geometry,
            "wire_length": config.wire_length,
            "wire_width": config.wire_width,
            "notch_depth": config.notch_depth,
            "notch_width": config.notch_width,
            "n_notches": config.n_notches,
            "applied_field": config.applied_field,
        },
        "Ic_plus": float(Ic_plus),
        "Ic_minus": float(Ic_minus),
        "delta_Ic": float(Ic_plus - Ic_minus),
        "diode_efficiency": float(eta),
        "diagnostics_plus": {
            "I": diag_plus["I_values"].tolist(),
            "V": diag_plus["V_values"].tolist(),
            "snapshot_path": snap_plus,
        },
        "diagnostics_minus": {
            "I": diag_minus["I_values"].tolist(),
            "V": diag_minus["V_values"].tolist(),
            "snapshot_path": snap_minus,
        },
    }
    
    # Save results
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    plot_iv_curves(results, out / "iv_curves.png")
    plot_differential_resistance(results, out / "differential_resistance.png")
    plot_diode_summary(results, out / "diode_summary.png")
    
    # Plot snapshots if available
    if snap_plus:
        plot_physics_snapshot(snap_plus, out / "snapshot_Ic_plus.png", f"Vortex State at I_c^+ ({Ic_plus:.4f})")
    if snap_minus:
        plot_physics_snapshot(snap_minus, out / "snapshot_Ic_minus.png", f"Vortex State at I_c^- ({Ic_minus:.4f})")
    
    logger.info(f"Results saved to {out}")
    
    return results


def run_parameter_sweep(
    param_name: str,
    param_values: List[float],
    base_config: DiodeConfig,
    geometry: str = "notch"
) -> Dict[str, Any]:
    """
    Sweep a parameter and measure diode efficiency vs that parameter.
    
    Args:
        param_name: Name of parameter to sweep (e.g., "notch_depth", "applied_field")
        param_values: List of values to test
        base_config: Base configuration
        geometry: Geometry type
        
    Returns:
        Sweep results
    """
    out = ensure_dir(base_config.output_dir)
    
    logger.info(f"Parameter sweep: {param_name}")
    logger.info(f"Values: {param_values}")
    
    results = []
    
    for val in param_values:
        logger.info(f"\n{'='*40}")
        logger.info(f"{param_name} = {val}")
        logger.info(f"{'='*40}")
        
        # Create config with modified parameter
        config_dict = {
            "wire_length": base_config.wire_length,
            "wire_width": base_config.wire_width,
            "notch_depth": base_config.notch_depth,
            "notch_width": base_config.notch_width,
            "n_notches": base_config.n_notches,
            "notch_spacing": base_config.notch_spacing,
            "current_max": base_config.current_max,
            "current_steps": base_config.current_steps,
            "solve_time": base_config.solve_time,
            "dt": base_config.dt,
            "thickness": base_config.thickness,
            "lambda_london": base_config.lambda_london,
            "max_edge_length": base_config.max_edge_length,
            "applied_field": base_config.applied_field,
            "output_dir": str(out / f"{param_name}_{val:.3f}"),
        }
        config_dict[param_name] = val
        config = DiodeConfig(**config_dict)
        
        # Run study
        result = run_diode_study(config, geometry=geometry)
        result["param_value"] = val
        results.append(result)
    
    # Compile sweep results
    sweep_results = {
        "param_name": param_name,
        "param_values": param_values,
        "Ic_plus": [r["Ic_plus"] for r in results],
        "Ic_minus": [r["Ic_minus"] for r in results],
        "diode_efficiency": [r["diode_efficiency"] for r in results],
    }
    
    # Save sweep results
    with open(out / f"sweep_{param_name}.json", "w") as f:
        json.dump(sweep_results, f, indent=2)
    
    # Plot sweep
    plot_parameter_sweep(sweep_results, out / f"sweep_{param_name}.png")
    
    return sweep_results


def plot_geometry(device: Device, filepath: Path, config: DiodeConfig):
    """Plot the device geometry with mesh."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Geometry outline
    ax = axes[0]
    film_points = np.array(device.film.points)
    ax.fill(film_points[:, 0], film_points[:, 1], alpha=0.3, color='blue', label='Superconductor')
    ax.plot(np.append(film_points[:, 0], film_points[0, 0]), 
            np.append(film_points[:, 1], film_points[0, 1]), 'b-', lw=2)
    
    # Mark terminals
    ax.axvline(-config.wire_length/2, color='red', linestyle='--', label='Source')
    ax.axvline(config.wire_length/2, color='green', linestyle='--', label='Drain')
    
    # Arrow showing current direction
    ax.annotate('', xy=(config.wire_length/4, 0), xytext=(-config.wire_length/4, 0),
                arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    ax.text(0, -config.wire_width/3, 'I+', fontsize=12, ha='center', color='orange')
    
    ax.set_xlabel('x (ξ)')
    ax.set_ylabel('y (ξ)')
    ax.set_title('Asymmetric Wire Geometry')
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Right: Mesh
    ax = axes[1]
    mesh = device.mesh
    ax.triplot(mesh.sites[:, 0], mesh.sites[:, 1], mesh.elements, 'k-', lw=0.3, alpha=0.5)
    ax.fill(film_points[:, 0], film_points[:, 1], alpha=0.2, color='blue')
    ax.set_xlabel('x (ξ)')
    ax.set_ylabel('y (ξ)')
    ax.set_title(f'Mesh ({len(mesh.sites)} nodes)')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Geometry plot saved: {filepath}")


def plot_iv_curves(results: Dict, filepath: Path):
    """Plot I-V curves for both current directions."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Positive direction
    I_plus = results["diagnostics_plus"]["I"]
    V_plus = results["diagnostics_plus"]["V"]
    ax.plot(I_plus, V_plus, 'ro-', label=f'I > 0 (I_c⁺ = {results["Ic_plus"]:.4f})', markersize=4)
    
    # Negative direction
    I_minus = results["diagnostics_minus"]["I"]
    V_minus = results["diagnostics_minus"]["V"]
    ax.plot(I_minus, V_minus, 'bs-', label=f'I < 0 (I_c⁻ = {results["Ic_minus"]:.4f})', markersize=4)
    
    # Mark Ic values
    ax.axvline(results["Ic_plus"], color='red', linestyle='--', alpha=0.5)
    ax.axvline(results["Ic_minus"], color='blue', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('|I| (dimensionless)', fontsize=12)
    ax.set_ylabel('V (dimensionless)', fontsize=12)
    ax.set_title('Current-Voltage Characteristics', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"I-V curves saved: {filepath}")


def plot_diode_summary(results: Dict, filepath: Path):
    """Create summary figure showing diode effect."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Bar chart of Ic+ and Ic-
    ax = axes[0]
    x = [0, 1]
    heights = [results["Ic_plus"], results["Ic_minus"]]
    colors = ['red', 'blue']
    labels = ['I_c⁺', 'I_c⁻']
    bars = ax.bar(x, heights, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14)
    ax.set_ylabel('Critical Current (dimensionless)', fontsize=12)
    ax.set_title('Critical Current Asymmetry', fontsize=14)
    
    # Add value labels on bars
    for bar, val in zip(bars, heights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                f'{val:.4f}', ha='center', va='bottom', fontsize=11)
    
    ax.set_ylim(0, max(heights) * 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Right: Diode efficiency gauge
    ax = axes[1]
    eta = results["diode_efficiency"]
    
    # Create a simple visualization
    ax.barh([0], [eta * 100], color='green' if eta > 0 else 'red', alpha=0.7, height=0.5)
    ax.axvline(0, color='black', linewidth=2)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Diode Efficiency η (%)', fontsize=12)
    ax.set_yticks([])
    ax.set_title(f'Diode Efficiency: η = {eta*100:.2f}%', fontsize=14)
    
    # Add annotations
    ax.text(eta * 100 / 2, 0, f'{eta*100:.2f}%', ha='center', va='center', 
            fontsize=14, fontweight='bold', color='white')
    ax.text(-45, 0.7, 'I_c⁺ < I_c⁻', fontsize=10, color='gray')
    ax.text(25, 0.7, 'I_c⁺ > I_c⁻', fontsize=10, color='gray')
    
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Diode summary saved: {filepath}")


def plot_parameter_sweep(sweep_results: Dict, filepath: Path):
    """Plot parameter sweep results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    param_name = sweep_results["param_name"]
    param_vals = sweep_results["param_values"]
    
    # Left: Ic+ and Ic- vs parameter
    ax = axes[0]
    ax.plot(param_vals, sweep_results["Ic_plus"], 'ro-', label='I_c⁺', markersize=8)
    ax.plot(param_vals, sweep_results["Ic_minus"], 'bs-', label='I_c⁻', markersize=8)
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Critical Current', fontsize=12)
    ax.set_title(f'Critical Current vs {param_name}', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Right: Diode efficiency vs parameter
    ax = axes[1]
    eta_percent = [e * 100 for e in sweep_results["diode_efficiency"]]
    ax.plot(param_vals, eta_percent, 'go-', markersize=8, linewidth=2)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.fill_between(param_vals, eta_percent, alpha=0.3, color='green')
    ax.set_xlabel(param_name, fontsize=12)
    ax.set_ylabel('Diode Efficiency η (%)', fontsize=12)
    ax.set_title(f'Diode Efficiency vs {param_name}', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Parameter sweep plot saved: {filepath}")


def plot_differential_resistance(results: Dict, filepath: Path):
    """Plot differential resistance dV/dI."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Positive direction
    I_plus = np.array(results["diagnostics_plus"]["I"])
    V_plus = np.array(results["diagnostics_plus"]["V"])
    
    # Negative direction
    I_minus = np.array(results["diagnostics_minus"]["I"])
    V_minus = np.array(results["diagnostics_minus"]["V"])
    
    # Calculate dV/dI using gradient
    # Avoid division by zero at I=0 if present, though I array usually starts > 0
    dVdI_plus = np.gradient(V_plus, I_plus)
    dVdI_minus = np.gradient(V_minus, I_minus)
    
    ax.plot(I_plus, dVdI_plus, 'r-', linewidth=2, label='Direction +')
    ax.plot(I_minus, dVdI_minus, 'b-', linewidth=2, label='Direction -')
    
    ax.set_xlabel('|Current| (dimensionless)', fontsize=12)
    ax.set_ylabel('Differential Resistance dV/dI', fontsize=12)
    ax.set_title('Differential Resistance vs Current', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add inset for zoomed view if needed or save log scale version
    
    plt.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Differential resistance plot saved: {filepath}")


def plot_physics_snapshot(h5_path: str, filepath: Path, title: str):
    """
    Plot snapshot of order parameter |psi|^2 from HDF5 solution file.
    
    Shows the spatial distribution of superconductivity at the critical point.
    """
    try:
        from tdgl.solution import Solution
    except ImportError:
        # Fallback if import path is different in user's env
        try:
            from tdgl import Solution
        except ImportError:
            logger.warning("Could not import Solution class. Skipping snapshot plot.")
            return

    try:
        solution = Solution.from_hdf5(h5_path)
        # Get the last frame
        psi = solution.tdgl_data.psi[-1]
        psi_sq = np.abs(psi)**2
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create triangulation for plotting
        x = solution.device.mesh.sites[:, 0]
        y = solution.device.mesh.sites[:, 1]
        triang = mtri.Triangulation(x, y, solution.device.mesh.elements)
        
        # Plot |psi|^2
        tpc = ax.tripcolor(triang, psi_sq, cmap='viridis', shading='gouraud', vmin=0, vmax=1)
        
        # Overlay geometry
        film_points = np.array(solution.device.film.points)
        ax.plot(np.append(film_points[:, 0], film_points[0, 0]), 
                np.append(film_points[:, 1], film_points[0, 1]), 'w-', lw=1)
        
        cbar = plt.colorbar(tpc, ax=ax, label='$|\\psi|^2$')
        ax.set_xlabel('x (ξ)')
        ax.set_ylabel('y (ξ)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Snapshot saved: {filepath}")
        
    except Exception as e:
        logger.warning(f"Failed to plot snapshot from {h5_path}: {e}")


def main():
    """Main entry point for diode effect study."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Superconducting Diode Effect Study")
    parser.add_argument("--geometry", choices=["notch", "ratchet"], default="notch",
                        help="Geometry type")
    parser.add_argument("--sweep", choices=["none", "depth", "field", "width"], default="none",
                        help="Parameter to sweep")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with reduced parameters")
    args = parser.parse_args()
    
    # Base configuration
    if args.quick:
        # Quick test configuration
        # NOTE: Applied field breaks time-reversal symmetry, which combined with
        # the asymmetric geometry should produce different Ic for +/- current
        config = DiodeConfig(
            wire_length=8.0,
            wire_width=2.0,
            notch_depth=1.0,  # 50% of wire width - Strong asymmetry
            notch_width=1.0,  # Compatible with coherence length xi=1
            n_notches=2,
            notch_spacing=2.0,
            current_max=0.10,  # Reduced max current to resolve low-current region
            current_steps=40,  # High resolution (step ~0.0025)
            solve_time=100.0,  # Keep long time to ensure lattice relaxation
            dt=0.2,
            max_edge_length=0.20, # Finer mesh for better pinning resolution
            applied_field=0.1,  # Intermediate field
            output_dir="outputs_diode_quick",
        )
    else:
        # Full study configuration (Optimized to match successful Quick test)
        config = DiodeConfig(
            wire_length=8.0,         # Match Quick (was 10.0)
            wire_width=2.0,
            notch_depth=1.0,
            notch_width=1.0,         # Match Quick (was 1.2)
            n_notches=2,             # Match Quick (was 3)
            notch_spacing=2.0,
            current_max=0.05,
            current_steps=60,        # Keep moderate resolution
            solve_time=100.0,
            dt=0.1,
            max_edge_length=0.20,    # Finer mesh (Critical for physics!)
            applied_field=0.1,       # CRITICAL: Breaks time-reversal symmetry
            output_dir="outputs_diode",
        )
    
    if args.sweep == "none":
        # Single run
        run_diode_study(config, geometry=args.geometry)
        
    elif args.sweep == "depth":
        # Sweep notch depth
        depths = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        run_parameter_sweep("notch_depth", depths, config, geometry=args.geometry)
        
    elif args.sweep == "field":
        # Sweep magnetic field
        fields = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        run_parameter_sweep("applied_field", fields, config, geometry=args.geometry)
        
    elif args.sweep == "width":
        # Sweep wire width
        widths = [1.5, 2.0, 2.5, 3.0, 3.5]
        run_parameter_sweep("wire_width", widths, config, geometry=args.geometry)


if __name__ == "__main__":
    main()

