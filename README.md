# Giant Efficiency and Sign Reversal of the Superconducting Diode Effect in Asymmetric Notched Nanowires

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![pyTDGL](https://img.shields.io/badge/solver-pyTDGL-green.svg)](https://github.com/loganbvh/py-tdgl)

This repository contains the simulation code, analysis scripts, and pre-computed output data for the paper:

> **Giant Efficiency and Sign Reversal of the Superconducting Diode Effect in Asymmetric Notched Nanowires**
> Tanvir Hassan
> Department of Physics, Jagannath University, Dhaka, Bangladesh

## Overview

We use time-dependent Ginzburg–Landau (TDGL) simulations to study the superconducting diode effect (SDE) in a mesoscopic nanowire patterned with asymmetric triangular notches. The key findings include:

- **Giant diode efficiency** up to η ≈ 57 % (field-driven) and η ≈ 66 % (geometry-driven)
- **Field-driven sign reversal** of η between B = 0.06 and 0.08 B<sub>c2</sub>
- **Geometry-driven sign reversals** (two) as the notch depth is swept from 0.2 ξ to 1.0 ξ
- η = 0 at B = 0, confirming the fundamental symmetry requirement

The simulations are performed with the open-source [pyTDGL](https://github.com/loganbvh/py-tdgl) package.

## Device Geometry

Two identical triangular notches (depth d<sub>notch</sub>, base width 1.0 ξ) are cut into **one edge only**, breaking spatial inversion symmetry. A perpendicular magnetic field B breaks time-reversal symmetry, together enabling a non-reciprocal critical current (I<sub>c</sub><sup>+</sup> ≠ I<sub>c</sub><sup>−</sup>).

## Repository Structure

```
├── diode_effect.py               # Main simulation script
├── mesh_convergence.py           # Mesh convergence study
├── threshold_sensitivity.py      # Voltage-threshold sensitivity analysis
├── README.md                     # This file
│
└── outputs_diode/                # Pre-computed simulation data & figures
    ├── results.json              # Baseline run (B = 0.10, d_notch = 1.0)
    ├── geometry.png              # Device geometry + mesh visualization
    ├── iv_curves.png             # I–V curves (baseline)
    ├── sweep_applied_field.json  # Field sweep summary data
    ├── sweep_applied_field.png   # η vs B plot
    ├── sweep_notch_depth.json    # Depth sweep summary data
    ├── sweep_notch_depth.png     # η vs d_notch plot
    ├── vortex_snapshots.png      # |ψ|² maps at Ic (publication figure)
    ├── snapshot_Ic_plus.h5       # HDF5 solution at Ic+ (baseline)
    ├── snapshot_Ic_minus.h5      # HDF5 solution at Ic− (baseline)
    │
    ├── applied_field_0.000/      # Per-field-value results
    ├── applied_field_0.020/
    ├── applied_field_0.040/
    ├── applied_field_0.060/
    ├── applied_field_0.080/
    ├── applied_field_0.100/
    │   ├── results.json          #   Ic+, Ic−, η, full I–V data
    │   ├── iv_curves.png         #   I–V characteristic
    │   ├── geometry.png          #   Mesh for this run
    │   ├── snapshot_Ic_plus.h5   #   Order-parameter snapshot
    │   └── snapshot_Ic_minus.h5
    │
    ├── notch_depth_0.200/        # Per-depth-value results
    │   ...                       #   (same structure as field dirs)
    ├── notch_depth_1.000/
    │
    ├── mesh_convergence/
    │   ├── convergence_results.json
    │   ├── mesh_convergence.png
    │   └── mesh_convergence_table.txt
    │
    └── threshold_sensitivity/
        ├── threshold_sensitivity.json
        └── threshold_sensitivity.png
```

## Installation

### Prerequisites

- Python ≥ 3.9
- [pyTDGL](https://github.com/loganbvh/py-tdgl) (and its dependencies: NumPy, SciPy, Matplotlib, h5py, meshpy)

### Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/superconducting-diode-effect.git
cd superconducting-diode-effect

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows

# Install dependencies
pip install tdgl numpy scipy matplotlib h5py
```

## Usage

### 1. Reproduce the baseline diode measurement

Run a single simulation at the default parameters (B = 0.10 B<sub>c2</sub>, d<sub>notch</sub> = 1.0 ξ):

```bash
python diode_effect.py
```

### 2. Sweep the applied magnetic field

```bash
python diode_effect.py --sweep field
```

Runs simulations at B / B<sub>c2</sub> = 0.00, 0.02, 0.04, 0.06, 0.08, 0.10 and produces `outputs_diode/sweep_applied_field.png`.

### 3. Sweep the notch depth

```bash
python diode_effect.py --sweep depth
```

Runs simulations at d<sub>notch</sub> / ξ = 0.2, 0.3, …, 1.0 and produces `outputs_diode/sweep_notch_depth.png`.

### 4. Quick test (reduced resolution)

```bash
python diode_effect.py --quick
```

Uses fewer current steps and saves to `outputs_diode_quick/` for fast verification.

### 5. Mesh convergence study

```bash
python mesh_convergence.py
```

Tests mesh sizes h / ξ = 0.50, 0.40, 0.30, 0.25, 0.20, 0.15 and generates `outputs_diode/mesh_convergence/mesh_convergence.png`.

### 6. Threshold sensitivity analysis

```bash
python threshold_sensitivity.py
```

Re-analyzes the saved I–V data with four voltage thresholds (no new simulations required). Produces `outputs_diode/threshold_sensitivity/threshold_sensitivity.png`.

## Key Results

### Diode Efficiency vs. Applied Field (d<sub>notch</sub> = 1.0 ξ)

| B / B<sub>c2</sub> | I<sub>c</sub><sup>+</sup> (J₀) | I<sub>c</sub><sup>−</sup> (J₀) | η (%) |
|:---:|:---:|:---:|:---:|
| 0.00 | 0.0139 | 0.0139 | 0.0 |
| 0.02 | 0.0139 | 0.0131 | +3.1 |
| 0.04 | 0.0148 | 0.0114 | +12.8 |
| 0.06 | 0.0081 | 0.0022 | **+57.4** |
| 0.08 | 0.00134 | 0.00218 | **−23.9** |
| 0.10 | 0.0039 | 0.0030 | +12.2 |

### Diode Efficiency vs. Notch Depth (B = 0.10 B<sub>c2</sub>)

| d<sub>notch</sub> / ξ | I<sub>c</sub><sup>+</sup> (J₀) | I<sub>c</sub><sup>−</sup> (J₀) | η (%) |
|:---:|:---:|:---:|:---:|
| 0.2 | 0.0198 | 0.0164 | +9.3 |
| 0.3 | 0.0139 | 0.0131 | +3.1 |
| 0.4 | 0.0072 | 0.0064 | +6.2 |
| 0.5 | 0.0097 | 0.0139 | −17.7 |
| 0.6 | 0.0030 | 0.0114 | **−58.2** |
| 0.7 | 0.0148 | 0.0190 | −12.4 |
| 0.8 | 0.0131 | 0.0089 | +19.1 |
| 0.9 | 0.0106 | 0.0022 | **+65.8** |
| 1.0 | 0.0039 | 0.0030 | +12.2 |

## Simulation Parameters

| Parameter | Symbol | Value |
|:---|:---:|:---:|
| Wire length | L | 8 ξ |
| Wire width | W | 2 ξ |
| London penetration depth | λ | 2 ξ |
| GL parameter | κ | 2 |
| Film thickness | d | 0.1 ξ |
| Inelastic scattering | γ | 10 |
| Notch base width | w | 1.0 ξ |
| Number of notches | — | 2 |
| Max mesh edge length | h | 0.20 ξ |
| Current steps | N | 60 |
| Max bias current | I<sub>max</sub> | 0.05 J₀ |
| Evolution time per step | — | 100 τ₀ |
| Voltage threshold | V<sub>th</sub> | 10⁻³ V₀ |

## Output Data Format

Each simulation directory contains a `results.json` file with the following structure:

```json
{
  "config": {
    "geometry": "notch",
    "wire_length": 8.0,
    "wire_width": 2.0,
    "notch_depth": 1.0,
    "notch_width": 1.0,
    "n_notches": 2,
    "applied_field": 0.1
  },
  "Ic_plus": 0.00386,
  "Ic_minus": 0.00302,
  "delta_Ic": 0.00084,
  "diode_efficiency": 0.122,
  "diagnostics_plus": { "I": [...], "V": [...] },
  "diagnostics_minus": { "I": [...], "V": [...] }
}
```

HDF5 snapshot files (`snapshot_Ic_plus.h5`, `snapshot_Ic_minus.h5`) contain the full spatial order-parameter field ψ(r) at the critical current and can be loaded with `tdgl.Solution.from_hdf5()`.

## Citing This Work

If you use this code or data, please cite:

```bibtex
@article{Hassan2026,
  title   = {Giant Efficiency and Sign Reversal of the Superconducting
             Diode Effect in Asymmetric Notched Nanowires},
  author  = {Hassan, Tanvir},
  journal = {},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

The TDGL solver used in this work:

```bibtex
@article{PyTDGL,
  title   = {pyTDGL: Time-dependent Ginzburg-Landau in Python},
  author  = {Bishop-Van Horn, Logan},
  journal = {Computer Physics Communications},
  year    = {2023},
  doi     = {10.1016/j.cpc.2023.108799}
}
```

## License

This project is released under the [MIT License](LICENSE).

## Contact

Tanvir Hassan — [tanvir6307@gmail.com](mailto:tanvir6307@gmail.com)
Department of Physics, Jagannath University, Dhaka, Bangladesh
