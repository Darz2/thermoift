# ThermoIFT - Thermodynamic Interfacial Tension Toolkit

A Python library for thermodynamic property prediction using PC-SAFT/cDFT calculations and machine learning.

## Overview

ThermoIFT provides an end-to-end pipeline for:
1. **Feed Generation** - Generate thermodynamic feed compositions
2. **Data Collection** - Run simulations and collect interfacial properties
3. **ML Preprocessing** - Analyze and visualize data before training
4. **ML Training** - Train machine learning models *(TODO)*
5. **ML Postprocessing** - Evaluate and visualize model performance

---

## Module Structure

```
thermoift/
├── __init__.py                 # Package exports
├── README.md                   # This file
│
├── ML_feeds.py                 # Feed generation for ML datasets
├── ML_preprocessing.py         # Data analysis & visualization
├── ML_postprocessing.py        # Model evaluation & diagnostics
│
├── FeosPlugin.py               # PC-SAFT/cDFT interface (FeOs)
├── Analyzer.py                 # Composition analysis & validation
│
├── PLOT_SETTINGS.py            # Publication-quality plot styling
├── semi_emperical_correlations.py  # Semi-empirical IFT correlations
├── rng_utils.py                # Random number generation utilities
│
└── BINARY_INTERACTION_PARAMETERS/
    └── KIJ/                    # Binary interaction parameters
        ├── BinaryInteractions.py
        └── PLOT_SETTINGS.py
```

---

## Pipeline Workflow

### 1. Feed Generation (`ML_feeds.py`)

Generate diverse feed compositions for training data.

```python
from thermoift import FeedsBuilder

# Build feed compositions
builder = FeedsBuilder(
    components=["carbon dioxide", "hydrogen", "methane"],
    n_feeds=1000,
    seed=42
)
feeds = builder.generate()
```

### 2. Data Collection (`FeosPlugin.py`)

Run PC-SAFT/cDFT calculations to compute interfacial properties.
The module is class-based — use the individual static-method classes directly.

```python
from thermoift.FeosPlugin import (
    RegistryManager, CompositionHandler, ParameterBuilder,
    VLECalculator, InterfacialTensionCalculator, DataProcessor, PlottingEngine,
)
import si_units as si
import feos

# 1. Load component registry (once per session)
RegistryManager.load_registry("path/to/parameters.json")

# 2. Build feed compositions
feeds = CompositionHandler.make_feeds([0.95, 0.03, 0.02])   # e.g. CO2/H2/Ar

# 3. Build PC-SAFT parameters
components = ["carbon dioxide", "hydrogen", "argon"]
params = ParameterBuilder.build_parameters(components, T_K=250.0)
eos = feos.HelmholtzEnergyFunctional.pcsaft(params)

# 4. Compute critical point and phase envelope
z = CompositionHandler.normalize_z(feeds[0])
feed_mol = CompositionHandler.compute_feed_moles(z)
Tc, Pc = VLECalculator.compute_critical_point(eos, feed_mol, T_guess=300.0)
T_vals = VLECalculator.make_T_grid(T_min=220.0, Tc_K=float(Tc / si.KELVIN), N_T=80)
T_bub, P_bub, T_dew, P_dew = VLECalculator.compute_phase_envelope(
    eos, T_vals, feed_mol, verbose=False, Tc=float(Tc / si.KELVIN)
)

# 5. TP flash + cDFT at a single state point
eq, x, y, rho_L, rho_V = VLECalculator.tp_flash(
    eos, T=250 * si.KELVIN, P=60 * si.BAR, feed=feed_mol,
    molar_masses=[44.01, 2.016, 39.948],
)
interface = InterfacialTensionCalculator.build_planar_interface(
    eq, critical_temperature=Tc, n_grid=1024, l_grid=100.0
)
gamma_mN_m, thickness_nm, enrichment = InterfacialTensionCalculator.solve_interface_properties(interface)
```

### 3. ML Preprocessing (`ML_preprocessing.py`)

Analyze data quality and visualize feature-target relationships.

```python
from thermoift import MLPreprocessing

# Initialize with data
prep = MLPreprocessing(
    df=combined_df,
    features=["temperature", "pressure", "z_carbon dioxide"],
    target="gamma"
)

# Correlation analysis
prep.plot_spearman_matrix(save_path="correlation_matrix")
corr_df = prep.feature_target_correlations()
low_corr = prep.drop_low_correlation_features(threshold=0.1)

# Exploratory plots
prep.plot_target_vs_temperature(save_path="gamma_vs_T")
prep.plot_target_vs_pressure(save_path="gamma_vs_P")
prep.plot_target_distribution(save_path="gamma_distribution")
prep.plot_target_vs_liquid_density(save_path="gamma_vs_rhoL")
prep.plot_target_vs_vapor_density(save_path="gamma_vs_rhoV")
prep.plot_target_phase_envelope(save_path="gamma_phase_envelope")
```

### 4. ML Training *(not yet included)*

Model training is intentionally left to the user's preferred framework
(scikit-learn, PyTorch, XGBoost, etc.).  See the **Quick Start** section
below for a minimal sklearn example using ``MLPreprocessing`` output data.

### 5. ML Postprocessing (`ML_postprocessing.py`)

Evaluate model performance and generate diagnostic plots.

```python
from thermoift import MLPostprocessing

# Initialize with predictions
post = MLPostprocessing(
    y_true=y_test,
    y_pred=model.predict(X_test),
    target="gamma",  # or "P_dew", "P_bubble"
    feature_importances=model.feature_importances_,
    feature_names=features
)

# Feature importance (linear and log scale)
post.plot_feature_importance(scale="linear", save_path="importance_linear")
post.plot_feature_importance(scale="log", save_path="importance_log")

# Model diagnostics
post.plot_parity(save_path="parity_plot")
post.plot_residual_distribution(save_path="residual_dist")
post.plot_residual_vs_predicted(save_path="residual_vs_pred")

# Print summary
post.print_summary()
```

---

## Utilities

### Analyzer (`Analyzer.py`)

Validate and format composition data from VLE calculations.

```python
from thermoift import Analyzer

# Load and validate CSV
result = Analyzer.load_csv("vle_results.csv")
print(f"Valid rows: {result['n_valid_rows']}/{result['n_rows']}")

# Format single row
table = Analyzer.format_table(filepath="vle_results.csv", row_index=0)

# Validate compositions sum to 1
validation = Analyzer.validate_sums(z=[0.9, 0.1], x=[0.95, 0.05], y=[0.85, 0.15])
```

### Plot Settings (`PLOT_SETTINGS.py`)

Consistent, publication-quality plot styling.

```python
import thermoift.PLOT_SETTINGS as ps

# Initialize figure
fig, ax = ps.plot_init()

# Plot data
ax.scatter(x, y, color=ps.colors[0])

# Apply styling
ps.apply_axis_style(ax)
ps.style_legend(ax)

# Save figure (PNG + PDF)
ps.save_plot(fig, "my_figure")
```

**Available settings:**
- `colors`, `color_10`, `color_8` - Color palettes
- `dual_colors` - Face/edge color pairs
- `markers` - Marker styles
- `label_map` - Column names → LaTeX labels
- `symbol_map` - Column names → LaTeX symbols
- `component_map` - Chemical names → LaTeX formulas

### Random Utilities (`rng_utils.py`)

Reproducible random number generation with selectable bit generators.

```python
from thermoift.rng_utils import get_rng, describe_rngs

# Default PCG64 generator (recommended)
rng = get_rng(seed=42)

# Select a different bit generator
rng = get_rng(seed=42, rng_type="MT19937")   # classic Mersenne Twister

# List all available generators
describe_rngs()
```

**Available `rng_type` options:** ``"PCG64"`` (default), ``"MT19937"``,
``"SFC64"``, ``"Philox"``.

### Semi-Empirical Correlations (`semi_emperical_correlations.py`)

Classical IFT correlations (Parachor, WSD, Redlich-Kister) benchmarked
against full cDFT results.

```python
from thermoift.semi_emperical_correlations import semi_emperical_correlations

sec = semi_emperical_correlations(n_grid=1024, l_grid=100.0)

# Populate pure-component cache at a given temperature
sec.batch_pure_component_cDFT(["carbon dioxide", "methane"], T_K=270.0)

# Parachor numbers for the cached components
parachors = sec.batch_parachor_numbers(["carbon dioxide", "methane"], T_K=270.0)

# Single-point mixture IFT via the three methods
gamma_par = sec.parachor_mixture_IFT(x, y, rho_l, rho_v, parachors)
gamma_wsd = sec.wsd_mixture_IFT(T_K=270.0, x=x, y=y, rho_l=rho_l, rho_v=rho_v)

# Multi-point sweep over the two-phase PT region
data = sec.compute_PT_colormap_data(...)

# Export to DataFrame / CSV
df = sec.data_to_dataframe(data, component_names=["carbon dioxide", "methane"])
```

---

## Target Variables

The library supports multiple target variables with automatic color coding:

| Target | Description | Plot Colors |
|--------|-------------|-------------|
| `gamma` | Interfacial tension [mN/m] | Green (light/dark) |
| `P_dew` | Dew point pressure [bar] | Red (light/dark) |
| `P_bubble` | Bubble point pressure [bar] | Blue (light/dark) |

---

## Installation

```bash
# From the thermoift directory
pip install -e .
```

**Dependencies:**
- numpy, pandas, scipy
- matplotlib, seaborn, scienceplots
- scikit-learn (for ML modules)
- feos (for PC-SAFT/cDFT calculations)

---

## Quick Start

```python
import pandas as pd
from thermoift import MLPreprocessing, MLPostprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("interfacial_results.csv")

# Define features and target
features = ["temperature", "pressure", "z_carbon dioxide", "z_hydrogen"]
target = "gamma"

# Preprocessing
prep = MLPreprocessing(df=df, features=features, target=target)
prep.plot_spearman_matrix(save_path="correlation")

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Postprocessing
post = MLPostprocessing(
    y_true=y_test,
    y_pred=model.predict(X_test),
    target="gamma",
    feature_importances=model.feature_importances_,
    feature_names=features
)
post.plot_parity(save_path="parity")
post.print_summary()
```

---

## License

MIT License

## Author

Darshan
