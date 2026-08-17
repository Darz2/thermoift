# thermoift

Thermodynamics and interfacial tension tools for N-component mixtures, built on PC-SAFT via [`feos`](https://github.com/feos-org/feos).

`thermoift` computes phase equilibria and vapour–liquid interfacial properties of multi-component mixtures — in particular CO2-rich mixtures with impurities (N2, O2, Ar, CH4, CO, H2, H2S) relevant to CO2 transport and storage. It is also used to generate training data for machine-learning surrogate models of phase equilibria and interfacial tension.

## Features

- PC-SAFT / PCP-SAFT property calculations for pure fluids and N-component mixtures
- Vapour–liquid equilibrium and interfacial tension of multi-component mixtures
- Binary interaction parameters supplied through the [`BINARY_INTERACTION_PARAMETERS`](https://github.com/Darz2/BINARY_INTERACTION_PARAMETERS) submodule
- Batch data generation for ML training sets

## Installation

```bash
git clone --recurse-submodules https://github.com/Darz2/thermoift.git
cd thermoift
pip install .
```

Requires Python >= 3.9. See `pyproject.toml` for the full dependency list.

## Citation

If you use this code, please cite:

> Raju, D.; Skartlien, R.; Ramdin, M.; Vlugt, T. J. H. *Vapor–Liquid Interfacial Properties of CO2 Mixtures for Sequestration Applications: Molecular Simulations, Classical Density Functional Theory, and Equations of State.* Industrial & Engineering Chemistry Research (2026). https://doi.org/10.1021/acs.iecr.5c04932

## License

MIT — see [LICENSE](LICENSE).
