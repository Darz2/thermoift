"""
FeosPlugin.py
=============
Utilities for PC-SAFT phase equilibrium and cDFT interfacial tension
computations for N-component mixtures using the feos (v0.9.4+) framework.

Class-based architecture:
- RegistryManager: Component parameter management
- CompositionHandler: Feed generation and composition utilities
- ParameterBuilder: Build feos Parameters with T-dependent kij
- PropertyCalculator: Density conversions and unit handling
- VLECalculator: Vapor-liquid equilibrium computations
- InterfacialTensionCalculator: cDFT planar interface computations
- DataProcessor: Results aggregation and export for Machine Learning datasets
- PlottingEngine: Visualization and plot generation
- UtilityFunctions: Miscellaneous formatting utilities
"""

#!/usr/bin/env python

############# Required Packages ############
import numpy as np
import feos
import json
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import si_units as si
from molmass import Formula
from pathlib import Path
from itertools import combinations
from scipy.interpolate import griddata, PchipInterpolator
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_erosion

from . import PLOT_SETTINGS as ps
from .BINARY_INTERACTION_PARAMETERS.KIJ import BinaryInteractions as KIJ
from .rng_utils import get_rng, describe_rngs

_HERE = Path(__file__).parent

############# CLASS 1: RegistryManager #############

class RegistryManager:
    """
    Central management of component registry and KIJ labels.
    Uses class attributes for lazy-loaded, shared state.
    """
    _registry = None
    _kij_labels = None

    @classmethod
    def load_registry(cls, json_path=None):
        """
        Load component parameters from JSON and populate registry.

        Parameters
        ----------
        json_path : str or Path, optional
            Path to parameters.json. If None, uses default location.

        Returns
        -------
        tuple
            (registry_dict, kij_labels_dict)
        """
        if json_path is None:
            json_path = _HERE / "parameters.json"

        with open(json_path, "r") as f:
            raw = json.load(f)

        registry = {}
        kij_labels_map = {}

        for entry in raw:
            name = entry["identifier"]["name"]
            p = entry

            kwargs = {
                "molarweight": p["molarweight"],
                "m": p["m"],
                "sigma": p["sigma"],
                "epsilon_k": p["epsilon_k"],
            }

            if "mu" in p:
                kwargs["mu"] = p["mu"]
            if "association_sites" in p:
                kwargs["association_sites"] = p["association_sites"]

            registry[name] = feos.PureRecord(
                feos.Identifier(
                    cas=p["identifier"]["cas"],
                    name=name,
                    iupac_name=p["identifier"]["iupac_name"],
                ),
                **kwargs,
            )
            kij_labels_map[name] = p["kij_label"]

        cls._registry = registry
        cls._kij_labels = kij_labels_map
        return registry, kij_labels_map

    @classmethod
    def _ensure_loaded(cls):
        """Lazy-load registry if not already loaded."""
        if cls._registry is None or cls._kij_labels is None:
            cls.load_registry()

    @classmethod
    def get_component_names(cls, parameters):
        """
        Extract component names from a feos parameters object.

        Parameters
        ----------
        parameters : feos.Parameters
            Parameters object with pure_records

        Returns
        -------
        tuple of str
            Component names extracted from identifiers
        """
        cls._ensure_loaded()
        labels = []
        for pr in parameters.pure_records:
            id_str = str(pr.identifier)
            match = re.search(r"name='([^']+)'", id_str)
            if not match:
                match = re.search(r'name=([^,)]+)', id_str)
            name = match.group(1).strip()
            labels.append(cls._kij_labels.get(name, name))
        return tuple(labels)

    @classmethod
    def get_pure_record(cls, component_name):
        """
        Get PureRecord for a component.

        Parameters
        ----------
        component_name : str
            Component name

        Returns
        -------
        feos.PureRecord
            The pure record for the component
        """
        cls._ensure_loaded()
        if component_name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(f"Unknown component: {component_name}. Available: {available}")
        return cls._registry[component_name]

    @classmethod
    def get_kij_label(cls, component_name):
        """
        Get KIJ label for a component.

        Parameters
        ----------
        component_name : str
            Component name

        Returns
        -------
        str
            KIJ label for the component
        """
        cls._ensure_loaded()
        if component_name not in cls._kij_labels:
            available = sorted(cls._kij_labels.keys())
            raise ValueError(f"Unknown component: {component_name}. Available: {available}")
        return cls._kij_labels[component_name]

    @classmethod
    def kij_labels_from_names(cls, component_names):
        """
        Convert component names to KIJ labels.

        Parameters
        ----------
        component_names : list of str
            Component names

        Returns
        -------
        list of str
            KIJ labels for each component
        """
        cls._ensure_loaded()
        missing = [n for n in component_names if n not in cls._kij_labels]
        if missing:
            available = sorted(cls._kij_labels.keys())
            raise ValueError(f"Unknown component(s): {missing}. Available: {available}")
        return [cls._kij_labels[n] for n in component_names]

    @classmethod
    def get_available_components(cls):
        """
        Get all available component names from registry.

        Returns
        -------
        list of str
            Sorted list of all registered component names
        """
        cls._ensure_loaded()
        return sorted(cls._registry.keys())

    @classmethod
    def map_csv_to_components(cls, csv_columns=None):
        """
        Map CSV column abbreviations to full component names from registry.

        Parameters
        ----------
        csv_columns : list of str, optional
            CSV column abbreviations (e.g. ['CO2', 'H2', 'Ar', 'N2', ...]).
            If None, defaults to ``['CO2', 'H2', 'Ar', 'N2', 'CH4', 'O2', 'CO', 'H2S']``.

        Returns
        -------
        list of str
            Full component names in same order

        Raises
        ------
        ValueError
            If a CSV column doesn't match any registered component
        """
        if csv_columns is None:
            csv_columns = ['CO2', 'H2', 'Ar', 'N2', 'CH4', 'O2', 'CO', 'H2S']

        cls._ensure_loaded()
        # Build mapping: abbreviation -> full name from registry
        # CO2 -> carbon dioxide, H2 -> hydrogen, etc.
        csv_to_full = {
            'CO2': 'carbon dioxide',
            'H2': 'hydrogen',
            'Ar': 'argon',
            'N2': 'nitrogen',
            'CH4': 'methane',
            'O2': 'oxygen',
            'CO': 'carbon monoxide',
            'H2S': 'hydrogen sulfide',
        }

        missing = [col for col in csv_columns if col not in csv_to_full]
        if missing:
            raise ValueError(
                f"CSV columns {missing} not found in mapping. "
                f"Available: {sorted(csv_to_full.keys())}"
            )

        result = [csv_to_full[col] for col in csv_columns]

        # Verify all mapped names exist in registry
        unknown = [name for name in result if name not in cls._registry]
        if unknown:
            available = sorted(cls._registry.keys())
            raise ValueError(
                f"Component(s) {unknown} not in registry. "
                f"Available: {available}"
            )

        return result


############# CLASS 2: CompositionHandler #############

class CompositionHandler:
    """Feed generation, normalization, and composition utilities."""

    @staticmethod
    def generate_feeds(CO2, n_points):
        """
        Generate feed compositions for a ternary mixture:
        CO2 (C1) (fixed), H2 (C2) (varied), Ar (C3) (remainder).

        Parameters
        ----------
        CO2 : float
            Mole fraction of CO2 (component 1).
        n_points : int
            Number of composition points.

        Returns
        -------
        feeds : np.ndarray
            Array of shape (n_points, 3) with columns [CO2, H2, Ar].
        """
        C2_values = np.linspace(0.0, 1.0 - CO2, n_points)
        feeds = np.array([[CO2, C2, 1.0 - CO2 - C2] for C2 in C2_values])
        return feeds

    @staticmethod
    def load_compositions(csv_path, components=None, n=None):
        """
        Load feed compositions from a CSV file.

        Reads the specified component columns, filters to rows where at
        least one of the non-CO2 components is non-zero, normalizes each
        row, and returns them as a feeds array ready for ``make_feeds``.

        Parameters
        ----------
        csv_path : str or Path
            Path to the CSV file (e.g. ``Combined_compositions.csv``).
        components : list of str, optional
            Column names to extract (e.g. ``["CO2", "H2", "Ar"]``).
            If None, defaults to ``['CO2', 'H2', 'Ar', 'N2', 'CH4', 'O2', 'CO', 'H2S']``.
        n : int, optional
            If given, return only the first *n* compositions.

        Returns
        -------
        feeds : np.ndarray
            Array of shape (n_compositions, len(components)).
        """
        if components is None:
            components = ['CO2', 'H2', 'Ar', 'N2', 'CH4', 'O2', 'CO', 'H2S']

        df = pd.read_csv(csv_path)
        missing = [c for c in components if c not in df.columns]
        if missing:
            raise ValueError(
                f"Columns {missing} not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )
        sub = df[components].values.astype(float)
        # keep only rows where the composition is non-trivial
        row_sums = sub.sum(axis=1)
        sub = sub[row_sums > 0]
        # normalize
        sub = sub / sub.sum(axis=1, keepdims=True)
        if n is not None:
            sub = sub[:n]
        return sub

    @staticmethod
    def make_feeds(*compositions):
        """
        Create a feeds array from one or more explicit compositions.

        Parameters
        ----------
        *compositions : array-like
            Each argument is a composition vector [z1, z2, ...].
            All compositions must have the same length.
            Each will be normalized to sum to 1.

        Returns
        -------
        feeds : np.ndarray
            Array of shape (n_compositions, n_components).

        Examples
        --------
        # Single ternary composition
        feeds = CompositionHandler.make_feeds([0.95, 0.02, 0.03])

        # Multiple compositions
        feeds = CompositionHandler.make_feeds([0.95, 0.02, 0.03],
                                              [0.80, 0.10, 0.10])
        """
        feeds = []
        for z in compositions:
            z = np.asarray(z, dtype=float)
            z = z / z.sum()  # normalize
            feeds.append(z)
        return np.array(feeds)

    @staticmethod
    def normalize_z(z):
        """
        Return z as float array normalized to sum to 1.

        Parameters
        ----------
        z : array-like
            Composition vector

        Returns
        -------
        np.ndarray
            Normalized composition
        """
        z = np.asarray(z, dtype=float)
        s = z.sum()
        if s == 0:
            raise ValueError("Composition z sums to 0.")
        return z / s

    @staticmethod
    def compute_feed_moles(z):
        """
        Convert mole fractions z to mole amounts using n_total.

        Parameters
        ----------
        z : array-like
            Mole fractions

        Returns
        -------
        array
            Mole amounts with si.MOL units
        """
        return np.asarray(z, dtype=float) * si.MOL

    @staticmethod
    def reduce_components(z, components, verbose=False):
        """
        Detect zero-fraction components and reduce the system accordingly.

        Parameters
        ----------
        z : array-like
            Mole fractions of the full component set.
        components : list
            Full list of component objects/names.
        verbose : bool
            Print reduction info if True.

        Returns
        -------
        active_z : np.ndarray
            Renormalized mole fractions of active components.
        active_components : list
            Reduced list of components with non-zero composition.
        is_reduced : bool
            True if any component was dropped.
        """
        z_arr = np.asarray(z, dtype=float)
        mask = ~np.isclose(z_arr, 0.0)
        active_components = [c for c, keep in zip(components, mask) if keep]
        active_z = z_arr[mask]
        active_z = active_z / active_z.sum()
        is_reduced = len(active_components) < len(components)

        if is_reduced and verbose:
            n = len(active_components)
            print(f"  --- Reduced to {n}-component system: {active_components}")

        return active_z, active_components, is_reduced

    @staticmethod
    def print_array2d(feeds, decimals=4):
        """
        Pretty-print a 2D composition array.

        Parameters
        ----------
        feeds : np.ndarray
            2D array of compositions
        decimals : int
            Decimal places to display
        """
        ncols = feeds.shape[1]
        header = " | ".join([f"{'z'+str(j+1):>10}" for j in range(ncols)])
        print(f"{'i':>3} | {header}")
        print("-" * (15 + 14*ncols))
        for i, row in enumerate(feeds, start=1):
            row_str = " ".join(f"{val:12.4f}" for val in row)
            print(f"{i:3d} | {row_str}")


############# CLASS 3: ParameterBuilder #############

class ParameterBuilder:
    """Build feos Parameters objects with T-dependent kij support."""

    @staticmethod
    def build_parameters(
        component_names,
        T_K=None,
        kij_builder=None,
        model_map=None,
    ):
        """
        Build a feos.Parameters object from inline PC-SAFT parameters,
        optionally loading T-dependent kij(T) from a KIJMatrixBuilder.

        Parameters
        ----------
        component_names : list[str]
            Ordered component names, e.g. ["carbon dioxide", "methane", "argon"].
        T_K : float, optional
            Temperature in Kelvin at which to evaluate kij(T).
        kij_builder : KIJMatrixBuilder, optional
            If provided, loads kij polynomials from KIJ.json files.
        model_map : dict, optional
            Passed to KIJMatrixBuilder.load_pairs().

        Returns
        -------
        feos.Parameters

        Raises
        ------
        ValueError
            If unknown component names or invalid kij_builder configuration.
        """
        # Guard
        if kij_builder is not None and T_K is None:
            raise ValueError("T_K must be provided when using kij_builder.")

        # Validate component names
        RegistryManager._ensure_loaded()
        missing = [n for n in component_names if n not in RegistryManager._registry]
        if missing:
            available = sorted(RegistryManager._registry.keys())
            raise ValueError(f"Unknown component(s): {missing}. Available: {available}")

        records = [RegistryManager._registry[name] for name in component_names]

        # Pure component
        if len(records) == 1:
            return feos.Parameters.new_pure(records[0])

        # Build binary_records from kij_pairs
        binary_records = []

        if kij_builder is not None:
            kij_labels = [RegistryManager._kij_labels[n] for n in component_names]
            label_to_record = dict(zip(kij_labels, records))
            pair_data = kij_builder.load_pairs(kij_labels, model_map=model_map)
            _, A, _ = KIJ.KIJMatrixBuilder.build_polynomial_tensor(kij_labels, pair_data)
            K = KIJ.KIJMatrixBuilder.kij_numeric_matrix(T_K, A)

            for (la, lb) in combinations(kij_labels, 2):
                i = kij_labels.index(la)
                j = kij_labels.index(lb)
                binary_records.append(feos.BinaryRecord(
                    id1=label_to_record[la].identifier,
                    id2=label_to_record[lb].identifier,
                    k_ij=float(K[i, j]),
                ))

        return feos.Parameters.from_records(records, binary_records=binary_records)

    @staticmethod
    def reduce_kij_map(active_components, kij_map):
        """
        Filter kij_map to only include pairs of active components.

        Parameters
        ----------
        active_components : list of str
            Reduced list of component names with non-zero composition.
        kij_map : dict
            Original kij map keyed by (component_i, component_j) tuples.
            Keys can be either full names or KIJ labels.

        Returns
        -------
        active_map : dict
            Filtered kij map with only active component pairs.
        """
        RegistryManager._ensure_loaded()
        # Convert active component names to KIJ labels for comparison
        active_labels = set()
        for comp in active_components:
            if comp in RegistryManager._kij_labels:
                active_labels.add(RegistryManager._kij_labels[comp])
            else:
                # If not in registry, assume it's already a KIJ label
                active_labels.add(comp)

        active_map = {
            (i, j): v for (i, j), v in kij_map.items()
            if i in active_labels and j in active_labels
        }
        return active_map


############# CLASS 4: PropertyCalculator #############

class PropertyCalculator:
    """Density conversions and unit handling."""

    @staticmethod
    def value_in(q, unit):
        """
        Return the numeric value of q in the given unit.

        Parameters
        ----------
        q : int, float, or quantity
            Value to extract
        unit : unit
            Target unit

        Returns
        -------
        float
            Numeric value in target unit
        """
        if isinstance(q, (int, float, np.floating)):
            return float(q)
        return np.asarray(q / unit, dtype=float)

    @staticmethod
    def molar_density_mol_m3(rho):
        """
        Convert molar density to mol/m^3.

        Handles multiple unit formats and raises TypeError if unable.
        """
        mol_per_m3 = si.MOL / (si.METER**3)
        try:
            return PropertyCalculator.value_in(rho, mol_per_m3)
        except Exception:
            pass
        try:
            kmol_per_m3 = si.KMOL / (si.METER**3)
            return PropertyCalculator.value_in(rho, kmol_per_m3) * 1000.0
        except Exception:
            pass
        try:
            kmol_per_m3 = (1000.0 * si.MOL) / (si.METER**3)
            return PropertyCalculator.value_in(rho, kmol_per_m3) * 1000.0
        except Exception as e:
            raise TypeError(f"Could not interpret density units") from e

    @staticmethod
    def density_kg_m3(rho_molar, z, molar_masses, molar_mass_unit="g/mol"):
        """
        Convert molar density to mass density (kg/m^3).

        Parameters
        ----------
        rho_molar : quantity
            Molar density
        z : array-like
            Mole fractions
        molar_masses : array-like
            Molar masses
        molar_mass_unit : str
            "g/mol" or "kg/mol"

        Returns
        -------
        float
            Mass density in kg/m^3
        """
        z = np.asarray(z, dtype=float)
        M = np.asarray(molar_masses, dtype=float)

        if molar_mass_unit.lower() == "g/mol":
            M = M / 1000.0
        elif molar_mass_unit.lower() == "kg/mol":
            pass
        else:
            raise ValueError("molar_mass_unit must be 'g/mol' or 'kg/mol'.")

        rho_mol_m3 = PropertyCalculator.molar_density_mol_m3(rho_molar)
        M_mix = np.sum(z * M)
        return rho_mol_m3 * M_mix

    @staticmethod
    def molar_masses(parameters):
        """
        Extract molar masses from a feos parameters object.

        Parameters
        ----------
        parameters : feos.Parameters

        Returns
        -------
        list of float
            Molar masses [g/mol] for each component
        """
        return [Formula(c).mass for c in RegistryManager.get_component_names(parameters)]


############# CLASS 5: VLECalculator #############

class VLECalculator:
    """Vapor-liquid equilibrium calculations."""

    @staticmethod
    def compute_bubble_curve(eos, T_vals, feed, verbose):
        """
        Calculate bubble point pressures across a temperature range.

        Parameters
        ----------
        eos : feos.EquationOfState
            Equation of state
        T_vals : array-like
            Temperature values [K]
        feed : array-like
            Feed composition (mole fractions)
        verbose : bool
            Print error messages if True

        Returns
        -------
        T_bubble : list
            Temperatures where bubble point was computed
        bubble_pressures : list
            Bubble point pressures [bar]
        """
        feed = np.array(feed / si.MOL)
        bubble_pressures = []
        T_bubble = []

        for T_K in T_vals:
            T = T_K * si.KELVIN
            try:
                envelope = feos.PhaseEquilibrium.bubble_point(
                    eos, temperature_or_pressure=T, liquid_molefracs=feed)
                P_bubble = envelope.liquid.pressure()
            except Exception as err:
                if verbose:
                    print(f"Bubble calculation failed at T = {T_K} K: {err}")
                continue

            bubble_pressures.append(P_bubble / si.BAR)
            T_bubble.append(T_K)

        return T_bubble, bubble_pressures

    @staticmethod
    def compute_dew_curve(eos, T_vals, feed, verbose):
        """
        Calculate dew point pressures across a temperature range.

        Parameters
        ----------
        eos : feos.EquationOfState
            Equation of state
        T_vals : array-like
            Temperature values [K]
        feed : array-like
            Feed composition (mole fractions)
        verbose : bool
            Print error messages if True

        Returns
        -------
        T_dew : list
            Temperatures where dew point was computed
        dew_pressures : list
            Dew point pressures [bar]
        """
        feed = np.array(feed / si.MOL)
        T_dew = []
        dew_pressures = []

        for T_K in T_vals:
            T = T_K * si.KELVIN
            try:
                envelope = feos.PhaseEquilibrium.dew_point(
                    eos, temperature_or_pressure=T, vapor_molefracs=feed)
                P_dew = envelope.vapor.pressure()
            except Exception as err:
                if verbose:
                    print(f"Dew calculation failed at T = {T_K} K: {err}")
                break

            dew_pressures.append(P_dew / si.BAR)
            T_dew.append(T_K)

        return T_dew, dew_pressures

    @staticmethod
    def compute_critical_point(eos, z, T_guess):
        """
        Compute critical point temperature and pressure for a composition.

        Parameters
        ----------
        eos : feos.EquationOfState
            Equation of state
        z : array-like
            Composition (mole fractions)
        T_guess : float
            Initial temperature guess [K]

        Returns
        -------
        cp_temperature : quantity
            Critical temperature
        cp_pressure : quantity
            Critical pressure
        """
        cp = feos.State.critical_point(eos, z, initial_temperature=T_guess * si.KELVIN)
        return cp.temperature, cp.pressure()

    @staticmethod
    def tp_flash(eos, T, P, feed, molar_masses):
        """
        Run TP flash and return equilibrium state and phase properties.

        Parameters
        ----------
        eos : feos.EquationOfState
            Equation of state
        T : quantity
            Temperature
        P : quantity
            Pressure
        feed : array-like
            Feed composition
        molar_masses : array-like
            Molar masses [g/mol]

        Returns
        -------
        eq : feos.PhaseEquilibrium
            Equilibrium result
        x : np.ndarray
            Liquid composition
        y : np.ndarray
            Vapor composition
        liquid_density : float
            Liquid density [kg/m^3]
        vapor_density : float
            Vapor density [kg/m^3]
        """
        eq = feos.PhaseEquilibrium.tp_flash(eos, T, P, feed)
        liq, vap = eq.liquid, eq.vapor
        x = np.array(liq.molefracs, dtype=float)
        y = np.array(vap.molefracs, dtype=float)
        liquid_density = PropertyCalculator.density_kg_m3(liq.density, x, molar_masses)
        vapor_density = PropertyCalculator.density_kg_m3(vap.density, y, molar_masses)

        return eq, x, y, liquid_density, vapor_density

############# CLASS 6: InterfacialTensionCalculator #############

class InterfacialTensionCalculator:
    """cDFT planar interface and interfacial tension calculations."""

    @staticmethod
    def build_planar_interface(eq, critical_temperature, n_grid, l_grid):
        """
        Create a planar interface object.

        Parameters
        ----------
        eq : feos.PhaseEquilibrium
            VLE equilibrium
        critical_temperature : quantity
            Critical temperature
        n_grid : int
            Number of grid points
        l_grid : float
            Domain length [Angstrom]

        Returns
        -------
        feos.PlanarInterface
            Interface object
        """
        return feos.PlanarInterface.from_tanh(
            vle=eq,
            n_grid=n_grid,
            l_grid=l_grid * si.ANGSTROM,
            critical_temperature=critical_temperature,
        )

    @staticmethod
    def solve_interface_properties(interface):
        """
        Solve planar interface and extract interfacial properties.

        Parameters
        ----------
        interface : feos.PlanarInterface
            Interface to solve

        Returns
        -------
        gamma_mN_m : float
            Surface tension [mN/m]
        interfacial_thickness_nm : float
            Interfacial thickness [nm]
        enrichment : tuple
            Interfacial enrichment values
        """
        sol = interface.solve()

        surface_tension = sol.surface_tension
        interfacial_thickness = sol.interfacial_thickness
        enrichment = sol.interfacial_enrichment

        gamma_mN_m = float(surface_tension * 1e3 / si.NEWTON * si.METER)
        interfacial_thickness_nm = float(interfacial_thickness() * 1e9 / si.METER)

        return gamma_mN_m, interfacial_thickness_nm, enrichment()


############# CLASS 7: DataProcessor #############

class DataProcessor:
    """Results aggregation, formatting, and export."""

    @staticmethod
    def assemble_row(
        T_K,
        P_bar,
        z,
        Tc,
        Pc,
        P_bubble,
        P_dew,
        liquid_density,
        vapor_density,
        x,
        y,
        gamma_mN_m,
        interfacial_thickness_nm,
        components,
        enrichment,
        gamma0_CO2=None,
        rhoL0_CO2_mol_cm3=None,
        rhoV0_CO2_mol_cm3=None,
        Tc_CO2=None,
        Psat_CO2=None,
        ML_mode=True,
    ):
        """
        Build a results dictionary for one composition case.
        Works for any number of components (binary, ternary, etc.)

        Optionally pads with zeros for all available components from registry
        for ML training (default), or returns only active components.

        Parameters
        ----------
        T_K : float
            Temperature [K]
        P_bar : float
            Pressure [bar]
        z : array-like
            Overall mole fractions
        Tc : float
            Critical temperature [K]
        Pc : float
            Critical pressure [bar]
        P_bubble : float
            Bubble point pressure [bar]
        P_dew : float
            Dew point pressure [bar]
        liquid_density : float
            Liquid density [kg/m^3]
        vapor_density : float
            Vapor density [kg/m^3]
        x : array-like
            Liquid phase mole fractions
        y : array-like
            Vapor phase mole fractions
        gamma_mN_m : float
            Surface tension [mN/m]
        interfacial_thickness_nm : float
            Interfacial thickness [nm]
        components : list of str
            Component names (typically active components)
        enrichment : tuple of float
            Enrichment values, one per component
        gamma0_CO2 : float or None, optional
            Pure CO2 IFT [mN/m] from semi_emperical_correlations._pure_component_cDFT.
        rhoL0_CO2_mol_cm3 : float or None, optional
            Pure CO2 saturated-liquid density [mol/cm3].
        rhoV0_CO2_mol_cm3 : float or None, optional
            Pure CO2 saturated-vapour density [mol/cm3].
        Tc_CO2 : float or None, optional
            Pure CO2 critical temperature [K].
        Psat_CO2 : float or None, optional
            Pure CO2 saturation pressure [bar].
        ML_mode : bool, optional
            If True (default), pad with all registry components for ML.
            If False, include only active components.

        Returns
        -------
        dict
            Results dictionary with thermodynamic and interfacial properties
        """
        row = {}

        # Determine which components to include in output
        RegistryManager._ensure_loaded()
        if ML_mode:
            all_components = list(RegistryManager._registry.keys())
        else:
            all_components = components

        # Thermodynamic state
        row["temperature"] = float(T_K)
        row["pressure"] = float(P_bar)

        # Feed composition (z) - include all components with zeros for inactive
        comp_dict = {c: float(z[i]) for i, c in enumerate(components)}
        for c in all_components:
            row[f"z_{c}"] = comp_dict.get(c, 0.0)

        # Critical points
        row["Tc"] = float(Tc / si.KELVIN)
        row["Pc"] = float(Pc / si.BAR)

        # Reduced properties
        row["Tr"] = float(T_K / row["Tc"])
        row["Pr"] = float(P_bar / row["Pc"])

        # Phase envelope limits
        row["P_bubble"] = float(P_bubble)
        row["P_dew"] = float(P_dew)

        # Bulk densities
        row["liquid_density"] = float(liquid_density)
        row["vapor_density"] = float(vapor_density)

        # Phase compositions (x) - include all components with zeros for inactive
        x_dict = {c: float(x[i]) for i, c in enumerate(components)}
        for c in all_components:
            row[f"x_{c}"] = x_dict.get(c, 0.0)

        # Phase compositions (y) - include all components with zeros for inactive
        y_dict = {c: float(y[i]) for i, c in enumerate(components)}
        for c in all_components:
            row[f"y_{c}"] = y_dict.get(c, 0.0)

        # Interfacial properties
        row["gamma"] = float(gamma_mN_m)
        row["interfacial_thickness"] = float(interfacial_thickness_nm)

        # Interfacial enrichment (E) - include all components with zeros for inactive
        E_dict = {c: float(enrichment[i]) for i, c in enumerate(components)}
        for c in all_components:
            row[f"E_{c}"] = E_dict.get(c, 0.0)

        # Pure CO2 properties (from semi_emperical_correlations)
        # Convert density: mol/cm3 -> kg/m3 = mol/cm3 * 1e6 cm3/m3 * M g/mol * 1e-3 kg/g
        M_CO2 = Formula("CO2").mass  # g/mol
        row["gamma0_CO2"] = float(gamma0_CO2) if gamma0_CO2 is not None else np.nan
        row["rhoL0_CO2"] = float(rhoL0_CO2_mol_cm3 * 1e3 * M_CO2) if rhoL0_CO2_mol_cm3 is not None else np.nan
        row["rhoV0_CO2"] = float(rhoV0_CO2_mol_cm3 * 1e3 * M_CO2) if rhoV0_CO2_mol_cm3 is not None else np.nan
        row["Tc_CO2"] = float(Tc_CO2) if Tc_CO2 is not None else np.nan
        row["Psat_CO2"] = float(Psat_CO2) if Psat_CO2 is not None else np.nan

        return row

    @staticmethod
    def summarize_vle_dft(VLE_DFT):
        """
        Print a summary of VLE-DFT data structure.

        Parameters
        ----------
        VLE_DFT : dict
            Dictionary containing VLE data with structure:
            {feed_key: {"interfacial_data": {T_K: [data_dicts, ...]},
                       "phase_envelope": {...}}}

        Returns
        -------
        dict
            Summary statistics for each feed
        """
        print("Available feeds:", list(VLE_DFT.keys()))

        summary = {}

        for feed_key in VLE_DFT:
            n_bubble = len(VLE_DFT[feed_key]["phase_envelope"]["bubble_curve"])
            n_dew = len(VLE_DFT[feed_key]["phase_envelope"]["dew_curve"])
            n_temps = len(VLE_DFT[feed_key]["interfacial_data"])

            # Count total interfacial data points
            total_interfacial = sum(
                len(data_list)
                for data_list in VLE_DFT[feed_key]["interfacial_data"].values()
            )

            print(f"{feed_key}: {n_bubble} bubble pts, {n_dew} dew pts, "
                  f"{n_temps} isotherms, {total_interfacial} interfacial pts")

            summary[feed_key] = {
                "n_bubble_points": n_bubble,
                "n_dew_points": n_dew,
                "n_isotherms": n_temps,
                "total_interfacial_points": total_interfacial
            }

        return summary

    @staticmethod
    def export_to_csv(VLE_DFT, folder="CSV", verbose=False):
        """
        Save VLE-DFT interfacial and phase envelope data to CSV files
        in separate subfolders.

        Parameters
        ----------
        VLE_DFT : dict
            VLE-DFT data structure
        folder : str
            Output root folder for CSV files
        verbose : bool
            Print progress if True

        Returns
        -------
        dict
            Mapping of feed keys to saved file paths
        """
        try:
            interfacial_folder = os.path.join(folder, "interfacial_results")
            envelope_folder = os.path.join(folder, "phase_envelope")

            os.makedirs(interfacial_folder, exist_ok=True)
            os.makedirs(envelope_folder, exist_ok=True)

        except Exception as e:
            print(f"Warning: Could not create folder '{folder}': {e}. Skipping CSV export.")
            return {}

        saved_files = {}

        for feed_key in VLE_DFT:
            try:
                # Save interfacial data
                all_data = []
                for T_K, data_list in VLE_DFT[feed_key]["interfacial_data"].items():
                    all_data.extend(data_list)

                df = pd.DataFrame(all_data)
                csv_filename = os.path.join(
                    interfacial_folder,
                    f"{feed_key}_interfacial_results.csv"
                )
                df.to_csv(csv_filename, index=False)

                if verbose:
                    print(f"Saved {len(df)} rows to {csv_filename}")

                # Save phase envelope data
                envelope_filename = os.path.join(
                    envelope_folder,
                    f"{feed_key}_phase_envelope.csv"
                )
                pd.DataFrame(
                    VLE_DFT[feed_key]["phase_envelope"]["isothermal_lines"]
                ).to_csv(envelope_filename, index=False)

                if verbose:
                    print(f"Saved phase envelope data to {envelope_filename}")

                saved_files[feed_key] = {
                    "interfacial": csv_filename,
                    "envelope": envelope_filename
                }

            except Exception as e:
                print(f"Warning: Could not save CSV for {feed_key}: {e}")
                continue

        return saved_files


############# CLASS 8: PlottingEngine #############

class PlottingEngine:
    """Visualization and plot generation."""

    @staticmethod
    def _smooth_curve(T_raw, P_raw, Tc, Pc, n=1000, gap_tol=0.01):
        """
        Fit a shape-preserving PCHIP spline to raw curve data.

        Appends (Tc, Pc) as a hard anchor only when there is a meaningful
        gap between the last computed point and the critical point
        (> gap_tol * Tc K).
        """
        T = np.asarray(T_raw, dtype=float)
        P = np.asarray(P_raw, dtype=float)
        idx = np.argsort(T)
        T, P = T[idx], P[idx]
        if (Tc - T[-1]) > gap_tol * Tc:
            T = np.append(T, Tc)
            P = np.append(P, Pc)
        T_s = np.linspace(T.min(), T.max(), n)
        return T_s, PchipInterpolator(T, P)(T_s)

    @staticmethod
    def plot_phase_diagram(PT_results, feed_key, parameters):
        """
        Plot PT phase diagram (bubble + dew) for a given feed.
        Includes critical point.

        Parameters
        ----------
        PT_results : dict
            Results dictionary with PT data
        feed_key : str
            Feed composition key
        parameters : feos.Parameters
            Parameters object for component info

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object
        ax : matplotlib.axes.Axes
            Axes object
        """
        if feed_key not in PT_results:
            raise KeyError(f"{feed_key} not found in PT_results dict.")

        feed_data = PT_results[feed_key]
        comps = RegistryManager.get_component_names(parameters)

        # Extract curves
        T_bub = [pt["T_K"] for pt in feed_data["bubble"]]
        P_bub = [pt["P_bar"] for pt in feed_data["bubble"]]
        T_dew = [pt["T_K"] for pt in feed_data["dew"]]
        P_dew = [pt["P_bar"] for pt in feed_data["dew"]]

        # Critical point
        Tc = feed_data["TC_K"]
        Pc = feed_data["PC_bar"]
        z = feed_data["z"]

        fig, ax = ps.plot_init()

        T_bub_s, P_bub_s = PlottingEngine._smooth_curve(T_bub, P_bub, Tc, Pc)
        T_dew_s, P_dew_s = PlottingEngine._smooth_curve(T_dew, P_dew, Tc, Pc)

        ax.plot(T_bub_s, P_bub_s, label="Bubble curve", linestyle="-",
                linewidth=ps.linewidth, color="blue")
        ax.plot(T_dew_s, P_dew_s, label="Dew curve", linestyle="-",
                linewidth=ps.linewidth, color="red")
        ax.plot(Tc, Pc, 'o', markersize=4, zorder=4, markerfacecolor="grey",
                markeredgewidth=1, markeredgecolor="black", label="Critical point")

        ax.set_xlabel(r'$T \; / \; [\mathrm{K}]$', fontsize=ps.label_fontsize)
        ax.set_ylabel(r'$P \; / \; [\mathrm{bar}]$', fontsize=ps.label_fontsize)
        ax.set_xlim(left=200)

        z_str = ", ".join(f"{UtilityFunctions.latex_formula(c)}={val:g}"
                         for c, val in zip(comps, z))
        ax.set_title(f"{z_str}", fontsize=ps.label_fontsize)
        ax.minorticks_on()

        ps.style_legend(ax, loc='best', ncol=1, borderaxespad=1.0, frame=False)
        ax.grid(False)
        fig.tight_layout()

        return fig, ax

    @staticmethod
    def plot_interfacial_tension_map(PT_results, VLE_DFT, feed_key, parameters):
        """
        Create a colormap plot of interfacial tension in PT space.

        Parameters
        ----------
        PT_results : dict
            PT-space results
        VLE_DFT : dict
            Full VLE-DFT dictionary
        feed_key : str
            Feed identifier
        parameters : feos.Parameters
            Parameters object

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure object or None if insufficient data
        """
        if feed_key not in PT_results:
            raise KeyError(f"{feed_key} not found in PT_results dict.")

        feed_data = PT_results[feed_key]
        comps = RegistryManager.get_component_names(parameters)

        # Extract phase envelope
        bubble_data = VLE_DFT[feed_key]["phase_envelope"]["bubble_curve"]
        dew_data = VLE_DFT[feed_key]["phase_envelope"]["dew_curve"]
        T_bubble = [pt["T_K"] for pt in bubble_data]
        P_bubble = [pt["P_bar"] for pt in bubble_data]
        T_dew = [pt["T_K"] for pt in dew_data]
        P_dew = [pt["P_bar"] for pt in dew_data]

        Tc = feed_data["TC_K"]
        Pc = feed_data["PC_bar"]
        z = feed_data["z"]

        # Extract interfacial tension data
        T_gamma, P_gamma, gamma_data = [], [], []

        for T_K, data_list in VLE_DFT[feed_key]["interfacial_data"].items():
            for point in data_list:
                if "gamma" in point and not np.isnan(point["gamma"]):
                    T_gamma.append(point["temperature"])
                    P_gamma.append(point["pressure"])
                    gamma_data.append(point["gamma"])

        if len(gamma_data) < 4:
            print(f"Warning: Insufficient data for {feed_key} to create colormap "
                  f"(need at least 4 points).")
            return None

        # Create figure
        fig, ax = ps.plot_init()

        # Grid generation
        grid_size = 100
        T_grid = np.linspace(min(T_gamma), max(T_gamma), grid_size)
        P_grid = np.linspace(min(P_gamma), max(P_gamma), grid_size)
        T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

        # Anchor gamma=0 at (Tc, Pc)
        T_gamma_aug = T_gamma + [Tc]
        P_gamma_aug = P_gamma + [Pc]
        gamma_aug = gamma_data + [0.0]

        # Cubic interpolation with critical point anchor
        gamma_mesh = griddata((T_gamma_aug, P_gamma_aug), gamma_aug,
                             (T_mesh, P_mesh), method='cubic', fill_value=np.nan)

        # Nearest-neighbour fallback for any remaining NaN
        nan_mask = np.isnan(gamma_mesh)
        if nan_mask.any():
            gamma_fill = griddata((T_gamma_aug, P_gamma_aug), gamma_aug,
                                 (T_mesh, P_mesh), method='nearest')
            gamma_mesh[nan_mask] = gamma_fill[nan_mask]

        # Mask regions outside phase envelope
        T_bub_s, P_bub_s = PlottingEngine._smooth_curve(T_bubble, P_bubble, Tc, Pc)
        T_dew_s, P_dew_s = PlottingEngine._smooth_curve(T_dew, P_dew, Tc, Pc)
        envelope_T = list(T_bub_s) + list(T_dew_s[::-1])
        envelope_P = list(P_bub_s) + list(P_dew_s[::-1])
        envelope_path = MplPath(np.column_stack([envelope_T, envelope_P]))

        grid_points = np.column_stack([T_mesh.ravel(), P_mesh.ravel()])
        inside = envelope_path.contains_points(grid_points).reshape(T_mesh.shape)
        gamma_masked = np.ma.masked_where(~inside, gamma_mesh)
        gamma_min = np.nanmin(gamma_masked)
        gamma_max = np.nanmax(gamma_masked)

        # Isoline levels
        levels_lines = [2*n + 1 for n in range(10)]
        levels_lines = [lev for lev in levels_lines if gamma_min <= lev <= gamma_max]

        # Color bands
        levels = np.linspace(gamma_min, gamma_max, 35)
        contour = ax.contourf(T_mesh, P_mesh, gamma_masked, levels=levels,
                             cmap=plt.cm.Blues, extend='both')

        # Contour lines
        inside_eroded = binary_erosion(inside, iterations=2)
        gamma_masked = np.ma.masked_where(~inside_eroded, gamma_mesh)
        contour_lines = ax.contour(T_mesh, P_mesh, gamma_masked, levels=levels_lines,
                                   colors='black', linewidths=ps.linewidth/2, alpha=0.6)

        ax.clabel(contour_lines, levels=levels_lines, inline=True,
                 fontsize=ps.label_fontsize/2, fmt='%.0f',
                 inline_spacing=15, rightside_up=True)

        # Colorbar legend
        cbar = fig.colorbar(contour, ax=ax, pad=0.02, format='%.0f')
        cbar.set_ticks(levels_lines)
        cbar.set_label(r'$\gamma \; / \; [\mathrm{mN \, m^{-1}}]$',
                      fontsize=ps.label_fontsize)
        cbar.ax.tick_params(labelsize=10)

        # Phase envelope curves
        ax.plot(T_bub_s, P_bub_s, 'b-', linewidth=ps.linewidth,
               label='Bubble curve', zorder=10)
        ax.plot(T_dew_s, P_dew_s, 'r-', linewidth=ps.linewidth,
               label='Dew curve', zorder=10)
        ax.plot(Tc, Pc, 'o', markersize=4, zorder=15, markerfacecolor="grey",
               markeredgewidth=1, markeredgecolor="black", label="Critical point")

        ax.set_xlabel(r'$T \; / \; [\mathrm{K}]$', fontsize=ps.label_fontsize)
        ax.set_ylabel(r'$P \; / \; [\mathrm{bar}]$', fontsize=ps.label_fontsize)
        ps.style_legend(ax, fontsize=ps.legend_fontsize, loc='best', framealpha=0)

        z_str = ", ".join(f"{UtilityFunctions.latex_formula(c)}={val:g}"
                         for c, val in zip(comps, z))
        ax.set_title(f"{z_str}", fontsize=ps.label_fontsize/2)
        ax.set_xlim(left=200)
        ax.minorticks_on()

        plt.tight_layout()

        return fig

    @staticmethod
    def save_figure(fig, filename, dpi=1200):
        """
        Save figure to filename at specified DPI.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename : str
            Output file path
        dpi : int
            DPI resolution (default 1200)
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')

    @staticmethod
    def save_plots(fig, filename_base, folder="PLOTS"):
        """
        Save a figure as both PNG and PDF in separate folders.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        filename_base : str
            Base filename without extension
        folder : str
            Output root folder
        """
        try:
            png_folder = os.path.join(folder, "PNG")
            pdf_folder = os.path.join(folder, "PDF")

            os.makedirs(png_folder, exist_ok=True)
            os.makedirs(pdf_folder, exist_ok=True)

            png_path = os.path.join(png_folder, f"{filename_base}.png")
            pdf_path = os.path.join(pdf_folder, f"{filename_base}.pdf")

            PlottingEngine.save_figure(fig, png_path)
            PlottingEngine.save_figure(fig, pdf_path)

        except Exception as e:
            print(f"Warning: Could not save plots to '{folder}': {e}")
            return None


############# CLASS 9: UtilityFunctions #############

class UtilityFunctions:
    """Formatting and miscellaneous utilities."""

    @staticmethod
    def latex_formula(formula: str) -> str:
        """
        Convert a chemical formula to LaTeX format.

        Example: 'CO2' -> 'CO$_2$'
                 'Ar'  -> 'Ar'

        Parameters
        ----------
        formula : str
            Chemical formula

        Returns
        -------
        str
            LaTeX-formatted formula
        """
        return re.sub(r"(\d+)", r"$_{\1}$", formula)

    @staticmethod
    def get_random_generator(seed=None, rng_type="PCG64"):
        """
        Get a numpy random Generator with specified bit generator.

        Parameters
        ----------
        seed : int, optional
            Seed for the random number generator. If None, uses entropy from OS.
        rng_type : str, default="PCG64"
            Type of random number generator: "PCG64", "MT19937", "SFC64", or "Philox"

        Returns
        -------
        np.random.Generator
            A numpy Generator with the specified bit generator.

        See Also
        --------
        describe_available_rngs : Print available RNG options
        """
        return get_rng(seed, rng_type)

    @staticmethod
    def describe_available_rngs():
        """Print a description of available random number generators."""
        describe_rngs()


############# BACKWARD COMPATIBILITY WRAPPERS #############
# These module-level functions maintain backward compatibility with the old API

# Feed generation
def generate_feeds(CO2, n_points):
    """Backward compatibility wrapper."""
    return CompositionHandler.generate_feeds(CO2, n_points)

def make_feeds(*compositions):
    """Backward compatibility wrapper."""
    return CompositionHandler.make_feeds(*compositions)

def load_compositions(csv_path, components=None, n=None):
    """Backward compatibility wrapper."""
    return CompositionHandler.load_compositions(csv_path, components, n=n)

def normalize_z(z):
    """Backward compatibility wrapper."""
    return CompositionHandler.normalize_z(z)

def compute_feed_moles(z):
    """Backward compatibility wrapper."""
    return CompositionHandler.compute_feed_moles(z)

def reduce_components(z, components, verbose=False):
    """Backward compatibility wrapper."""
    return CompositionHandler.reduce_components(z, components, verbose)

def print_array2d(feeds, decimals=4):
    """Backward compatibility wrapper."""
    return CompositionHandler.print_array2d(feeds, decimals)

# Component and parameters
def components(parameters):
    """Backward compatibility wrapper."""
    return RegistryManager.get_component_names(parameters)

def _kij_labels(component_names):
    """Backward compatibility wrapper for private function."""
    return RegistryManager.kij_labels_from_names(component_names)

def molar_masses(parameters):
    """Backward compatibility wrapper."""
    return PropertyCalculator.molar_masses(parameters)

def PARAMETERS(component_names, T_K=None, kij_builder=None, model_map=None):
    """Backward compatibility wrapper."""
    return ParameterBuilder.build_parameters(component_names, T_K, kij_builder, model_map)

def reduce_kij_map(active_components, kij_map):
    """Backward compatibility wrapper."""
    return ParameterBuilder.reduce_kij_map(active_components, kij_map)

# Unit conversions
def value_in(q, unit):
    """Backward compatibility wrapper."""
    return PropertyCalculator.value_in(q, unit)

def molar_density_mol_m3(rho):
    """Backward compatibility wrapper."""
    return PropertyCalculator.molar_density_mol_m3(rho)

def density_kg_m3(rho_molar, z, molar_masses, molar_mass_unit="g/mol"):
    """Backward compatibility wrapper."""
    return PropertyCalculator.density_kg_m3(rho_molar, z, molar_masses, molar_mass_unit)

# VLE calculations
def compute_bubble_curve(eos, T_vals, feed, verbose):
    """Backward compatibility wrapper."""
    return VLECalculator.compute_bubble_curve(eos, T_vals, feed, verbose)

def compute_dew_curve(eos, T_vals, feed, verbose):
    """Backward compatibility wrapper."""
    return VLECalculator.compute_dew_curve(eos, T_vals, feed, verbose)

def compute_CT(feos_module, eos, z, T_guess):
    """Backward compatibility wrapper."""
    return VLECalculator.compute_critical_point(eos, z, T_guess)

def tp_flash(feos_module, eos, T, P, feed, molar_masses):
    """Backward compatibility wrapper."""
    return VLECalculator.tp_flash(eos, T, P, feed, molar_masses)

# Interfacial tension
def build_planar_interface(eq, critical_temperature, n_grid, l_grid):
    """Backward compatibility wrapper."""
    return InterfacialTensionCalculator.build_planar_interface(
        eq, critical_temperature, n_grid, l_grid)

def solve_interface_properties(interface, si_module):
    """Backward compatibility wrapper."""
    return InterfacialTensionCalculator.solve_interface_properties(interface)

# Data processing
def row_append(T_K, P_bar, z, Tc, Pc, P_bubble, P_dew, liquid_density, vapor_density,
               x, y, gamma_mN_m, interfacial_thickness_nm, components, enrichment, ML_mode=True):
    """Backward compatibility wrapper."""
    return DataProcessor.assemble_row(
        T_K, P_bar, z, Tc, Pc, P_bubble, P_dew, liquid_density, vapor_density,
        x, y, gamma_mN_m, interfacial_thickness_nm, components, enrichment, ML_mode)

def VLE_DFT_summary(VLE_DFT):
    """Backward compatibility wrapper."""
    return DataProcessor.summarize_vle_dft(VLE_DFT)

def VLE_DFT_to_csv(VLE_DFT, folder="CSV", verbose=False):
    """Backward compatibility wrapper."""
    return DataProcessor.export_to_csv(VLE_DFT, folder, verbose)

# Plotting
def plot_PT(PT_results, feed_key, parameters):
    """Backward compatibility wrapper."""
    return PlottingEngine.plot_phase_diagram(PT_results, feed_key, parameters)

def plot_gamma_colormap(PT_results, VLE_DFT, feed_key, parameters):
    """Backward compatibility wrapper."""
    return PlottingEngine.plot_interfacial_tension_map(PT_results, VLE_DFT, feed_key, parameters)

def save_figure(fig, filename):
    """Backward compatibility wrapper."""
    return PlottingEngine.save_figure(fig, filename)

def save_plot(fig, filename_base, folder="PLOTS"):
    """Backward compatibility wrapper."""
    return PlottingEngine.save_plots(fig, filename_base, folder)

# Utilities
def latex_formula(formula: str) -> str:
    """Backward compatibility wrapper."""
    return UtilityFunctions.latex_formula(formula)


############# Module Initialization #############
# Registry is NOT auto-loaded at import time.
# Users must explicitly call RegistryManager.load_registry(json_path="path/to/parameters.json")
# in their scripts before using other functions.
