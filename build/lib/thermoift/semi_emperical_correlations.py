#!/usr/bin/env python

# Created by Darshan on 2025-03-15
# UPDATE: 2025-03-15 by Darshan (Added the Parachor, WSD, RK model)
# UPDATE: 2025-03-17 by Darshan (Added the PT colormap plotting function, saturation pressure functionality)
# UPDATE: 2025-03-18 by Darshan (Added the critical pressure functionality, added more error handling and warnings)
# UPDATE: 2025-03-23 by Darshan (Test and made the examples to compute the interfacial tension of the semi emperical models)
# UPDATE: 2025-03-23 by Darshan (Add the PT colormap plotting and compute the with semi-emperical correlations).
# UPDATE: 2026-03-24 by Darshan (Added compute_saturation_line_IFT for IFT along bubble/dew curves only)
# UPDATE: 2026-03-25 by Darshan (Added the functionality of compute_saturation_line_IFT in the FEOS Plugin to compare (added but not as a function but as a useage code))
# UPDATE: 2026-03-25 by Darshan Saved the computed IFT as Dataframes seperate function specific to ML was added
# UPDATE: 2026-03-25 by claude  Added T-dependent kij_builder/model_map support to compute_PT_colormap_data and compute_saturation_line_IFT
# UPDATE: 2026-03-26 by claude  Added ColormapDifference class for computing and plotting IFT difference colormaps
# TODO (LATER): Add the IFT from the Pxy diagram for the binary mixtures - similar to old paper/package as a new file
# TODO (LATER): Add the metastable limits from the CNT similar to the old paper/parachorpy package as a new files

"""
semi_emperical_correlations.py
==============================m
Mixture interfacial tension via semi-empirical methods, fully consistent
with the feos/PC-SAFT framework.

Pure-component interfacial tension and saturation densities are obtained
from PC-SAFT EoS + cDFT (planar interface).

Classes
-------
semi_emperical_correlations
    Provides three semi-empirical methods for computing mixture IFT:

    - **Parachor** : uses parachor numbers and phase densities.
    - **WSD** : Winterfeld-Scriven-Davis method using pure-component IFTs.
    - **RK** : Redlich-Kister polynomial expansion.

    Additional functionality includes PT-colormap sweeps over the two-phase
    region, saturation-line IFT calculations, and DataFrame/CSV export
    utilities for ML training pipelines.
"""

import os
import json
import feos
import numpy as np
import pandas as pd
import warnings
import si_units as si
import matplotlib.pyplot as plt

from scipy.interpolate import griddata, PchipInterpolator
from scipy.ndimage import binary_erosion
from matplotlib.path import Path as MplPath
from .FeosPlugin import PARAMETERS, molar_density_mol_m3, components, latex_formula
from . import PLOT_SETTINGS as ps


class semi_emperical_correlations:
    """
    Mixture interfacial tension via semi-empirical correlations, fully consistent
    with the feos/PC-SAFT framework.

    Pure-component interfacial tensions and saturation densities are computed
    from PC-SAFT EoS + cDFT (planar interface) and cached internally so that
    expensive DFT solves are not repeated unnecessarily.

    Three semi-empirical methods are available for computing mixture IFT:

    - **Parachor** : uses parachor numbers and mixture phase densities.
    - **WSD** : Winterfeld-Scriven-Davis method using pure-component IFTs
      (DOI:10.1002/aic.690240610).
    - **RK** : Redlich-Kister polynomial expansion for the excess IFT.

    Typical workflow
    ----------------
    1. Instantiate the class (configure cDFT grid).
    2. Populate the pure-component cache with ``batch_pure_component_cDFT``
       (full cDFT) or ``batch_pure_component_VLE`` (fast VLE + scaling).
    3. Compute mixture IFT at a single state point via ``parachor_mixture_IFT``,
       ``wsd_mixture_IFT``, or ``RK_mixture_IFT``.
    4. For multi-point sweeps use ``compute_PT_colormap_data`` (full T-P grid)
       or ``compute_saturation_line_IFT`` (bubble/dew curves only).
    5. Export results to pandas DataFrames / CSV with ``data_to_dataframe``,
       ``saturation_line_to_dataframe``, ``save_saturation_dataframes``, or
       ``save_PT_colormap_dataframes``.
    6. Visualise with ``plot_PT_colormap``.

    Parameters
    ----------
    n_grid : int, optional
        Number of DFT grid points for the planar-interface solve. Default is 1024.
    l_grid : float, optional
        cDFT domain length in Angstroms. Default is 100.0 A.
    dynamic_lgrid : bool, optional
        When ``True``, the domain length is scaled as
        ``l_grid * (1 - T/Tc)^{-0.5}`` for each component so that the box
        grows near the critical point where the interface becomes diffuse.
        This improves cDFT convergence close to Tc at the cost of a larger
        (and slower) solve for those temperatures.  Default is ``False``.
    dynamic_ngrid : bool, optional
        When ``True``, the number of grid points is scaled by the same factor
        as ``dynamic_lgrid``, i.e. ``n_grid * (1 - T/Tc)^{-0.5}``, so that
        the grid spacing (Å/point) stays constant as the domain expands near
        Tc.  Has no effect unless ``dynamic_lgrid`` is also ``True``.
        Default is ``False``.

    Attributes
    ----------
    component_names : list[str] or None
        Component names set by the last call to ``batch_pure_component_cDFT``
        or ``batch_pure_component_VLE``.
    T_K_cached : float or None
        Temperature [K] for which pure-component data are currently cached.
    gamma0 : np.ndarray or None
        Cached pure IFTs [mN/m], shape (N,).
    rhoL0 : np.ndarray or None
        Cached pure saturated-liquid densities [mol/cm3], shape (N,).
    rhoV0 : np.ndarray or None
        Cached pure saturated-vapour densities [mol/cm3], shape (N,).
    Psat : np.ndarray or None
        Cached pure saturation pressures [bar], shape (N,).
    Tc : np.ndarray or None
        Cached PC-SAFT critical temperatures [K], shape (N,).
    Pc : np.ndarray or None
        Cached PC-SAFT critical pressures [bar], shape (N,).

    Methods
    -------
    batch_pure_component_cDFT(component_names, T_K)
        Full cDFT solve for every pure component; populates the cache.
    batch_pure_component_VLE(component_names, T_K, ...)
        Fast VLE-only alternative (~100x faster, uses scaling for gamma0).
    batch_parachor_numbers(component_names, T_K, ...)
        Parachor numbers from cached (or freshly computed) pure data.
    parachor_mixture_IFT(x, y, rho_l, rho_v, parachor_numbers, ...)
        Single-point mixture IFT via the Parachor method.
    wsd_mixture_IFT(T_K, x, y, rho_l, rho_v, ...)
        Single-point mixture IFT via the WSD method.
    RK_mixture_IFT(T_K, x, rk_coeffs)
        Single-point mixture IFT via the Redlich-Kister method.
    compute_PT_colormap_data(...)
        Sweep a (T, P) grid inside the two-phase envelope.
    compute_saturation_line_IFT(...)
        IFT along the bubble and/or dew curves only.
    data_to_dataframe(data, component_names)
        Convert PT-colormap dict to a flat DataFrame.
    saturation_line_to_dataframe(data, component_names, ...)
        Convert saturation-line dict to a DataFrame.
    prepare_method_df(df, line_name, method_suffix)
        Filter and suffix-rename a saturation DataFrame for merging.
    save_saturation_dataframes(...)
        Export saturation-line results to CSV (ML-ready or minimal).
    save_PT_colormap_dataframes(...)
        Export PT-colormap results to CSV (ML-ready or minimal).
    plot_PT_colormap(...)
        Filled-contour visualisation of IFT over the two-phase region.

    Examples
    --------
    >>> sec = semi_emperical_correlations(n_grid=1024, l_grid=100.0)
    >>> sec.batch_pure_component_cDFT(["methane", "ethane"], T_K=300.0)
    >>> parachors = sec.batch_parachor_numbers(["methane", "ethane"], T_K=300.0)
    """

    def __init__(
        self,
        n_grid: int          = 1024,
        l_grid: float        = 100.0,
        dynamic_lgrid: bool  = False,
        dynamic_ngrid: bool  = False,
        max_scale: float     = 10.0,
    ) -> None:

        self._parameters_fn  = PARAMETERS
        self._density_fn     = molar_density_mol_m3
        self._n_grid         = n_grid
        self._l_grid         = l_grid
        self._dynamic_lgrid  = dynamic_lgrid
        self._dynamic_ngrid  = dynamic_ngrid
        self._max_scale      = max_scale

        self.component_names: list[str] | None = None
        self.T_K_cached: float | None = None
        self.gamma0: np.ndarray | None = None
        self.rhoL0: np.ndarray | None = None
        self.rhoV0: np.ndarray | None = None
        self.Psat: np.ndarray | None = None
        self.Tc: np.ndarray | None = None
        self.Pc: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Pure-component DFT (single component)
    # ------------------------------------------------------------------

    def _pure_component_cDFT(
        self,
        component_name: str,
        T_K: float,
        dynamic_lgrid: bool | None = None,
        dynamic_ngrid: bool | None = None,
    ) -> tuple[float, float, float, float, float, float]:
        """
        Run a PC-SAFT EoS + planar cDFT solve for one pure component.

        Parameters
        ----------
        component_name : str
            Name accepted by FeosPlugin.PARAMETERS.
        T_K : float
            Temperature [K].
        dynamic_lgrid : bool or None, optional
            Override the instance-level ``dynamic_lgrid`` setting for this
            call only.  When ``True``, the cDFT domain length is scaled as
            ``l_grid * (1 - T/Tc)^{-0.5}`` so that the box grows near the
            critical point where the interface becomes diffuse.  Defaults to
            ``None``, which inherits from ``self._dynamic_lgrid``.
        dynamic_ngrid : bool or None, optional
            Override the instance-level ``dynamic_ngrid`` setting for this
            call only.  When ``True``, the grid point count is scaled by the
            same factor as the domain so that grid spacing (Å/point) remains
            constant.  Defaults to ``None``, which inherits from
            ``self._dynamic_ngrid``.

        Returns
        -------
        gamma0 : float
            Pure IFT [mN/m]. NaN if SC or cDFT fails.
        rhoL0 : float
            Liquid density [mol/cm3]. NaN if SC.
        rhoV0 : float
            Vapour density [mol/cm3]. NaN if SC.
        Tc_K : float
            PC-SAFT critical temperature [K].
        Pc_bar : float
            PC-SAFT critical pressure [bar].
        Psat : float
            Saturation pressure [bar]. NaN if SC.
        """
        use_dynamic_l = self._dynamic_lgrid if dynamic_lgrid is None else dynamic_lgrid
        use_dynamic_n = self._dynamic_ngrid if dynamic_ngrid is None else dynamic_ngrid

        params  = self._parameters_fn([component_name])
        eos     = feos.HelmholtzEnergyFunctional.pcsaft(params)

        cp      = feos.State.critical_point(eos)
        Tc_K    = float(cp.temperature / si.KELVIN)
        Pc_bar  = float(cp.pressure() / si.BAR)

        if T_K >= Tc_K:
            return np.nan, np.nan, np.nan, Tc_K, Pc_bar, np.nan

        # Dynamic domain/grid: scale with (1 - T/Tc)^{-0.5} near the critical
        # point so the box always contains the full diffuse interface.
        if use_dynamic_l:
            T_red  = T_K / Tc_K
            scale  = min(max(1.0, (1.0 - T_red) ** (-0.5)), self._max_scale)
            l_grid = self._l_grid * scale
            n_grid = int(self._n_grid * scale) if use_dynamic_n else self._n_grid
        else:
            l_grid = self._l_grid
            n_grid = self._n_grid

        try:
            vle = feos.PhaseEquilibrium.pure(eos, T_K * si.KELVIN)

        except Exception as exc:
            warnings.warn(
                f"cDFT (planar-interface) failed for pure component '{component_name}' at T={T_K} K: {exc} (From semi_emperical_correlations)"
            )
            return np.nan, np.nan, np.nan, Tc_K, Pc_bar, np.nan

        rhoL0   = self._density_fn(vle.liquid.density) * 1e-6
        rhoV0   = self._density_fn(vle.vapor.density) * 1e-6
        Psat    = float(vle.liquid.pressure() / si.BAR)

        # Guard against silent EOS failures (e.g. P ~ 1e-237 bar): positive and
        # finite but physically impossible for any real component below Tc.
        if not (np.isfinite(Psat) and Psat > 1e-6):
            return np.nan, np.nan, np.nan, Tc_K, Pc_bar, np.nan

        try:
            interface = feos.PlanarInterface.from_tanh(
                vle=vle,
                n_grid=n_grid,
                l_grid=l_grid * si.ANGSTROM,
                critical_temperature=cp.temperature)

            sol    = interface.solve()
            gamma0 = float(sol.surface_tension * 1e3 / si.NEWTON * si.METER)

        except Exception as exc:
            if not use_dynamic_l:
                try:
                    T_red        = T_K / Tc_K
                    retry_scale  = min(max(1.0, (1.0 - T_red) ** (-0.5)), self._max_scale)
                    retry_l_grid = self._l_grid * retry_scale
                    retry_n_grid = int(self._n_grid * retry_scale)

                    retry_interface = feos.PlanarInterface.from_tanh(
                        vle=vle,
                        n_grid=retry_n_grid,
                        l_grid=retry_l_grid * si.ANGSTROM,
                        critical_temperature=cp.temperature)

                    sol    = retry_interface.solve()
                    gamma0 = float(sol.surface_tension * 1e3 / si.NEWTON * si.METER)
                    return gamma0, rhoL0, rhoV0, Tc_K, Pc_bar, Psat

                except Exception as retry_exc:
                    warnings.warn(
                        f"cDFT solve failed for pure component '{component_name} (From semi_emperical_correlations)' "
                        f"at T={T_K} K with base grid: {exc}. Dynamic-grid retry also failed: {retry_exc}"
                    )
                    return np.nan, rhoL0, rhoV0, Tc_K, Pc_bar, np.nan

            warnings.warn(
                f"cDFT solve failed for pure component '{component_name} (From semi_emperical_correlations)' "
                f"at T={T_K} K: {exc}"
            )
            return np.nan, rhoL0, rhoV0, Tc_K, Pc_bar, np.nan

        return gamma0, rhoL0, rhoV0, Tc_K, Pc_bar, Psat

    def batch_pure_component_cDFT(
        self,
        component_names: list[str],
        T_K: float,
        dynamic_lgrid: bool | None = None,
        dynamic_ngrid: bool | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute and cache pure-component IFT and saturation densities for a list
        of components at a given temperature.

        Parameters
        ----------
        component_names : list[str]
            List of component names accepted by FeosPlugin.PARAMETERS.
        T_K : float
            Temperature [K].
        dynamic_lgrid : bool or None, optional
            Override the instance-level ``dynamic_lgrid`` setting for this
            batch.  When ``True``, each component's cDFT domain is scaled
            with ``(1 - T/Tc)^{-0.5}`` to capture the diffuse interface near
            the critical point.  Defaults to ``None`` (inherits from
            ``self._dynamic_lgrid``).
        dynamic_ngrid : bool or None, optional
            Override the instance-level ``dynamic_ngrid`` setting for this
            batch.  When ``True``, grid points scale by the same factor as the
            domain so grid spacing stays constant.  Defaults to ``None``
            (inherits from ``self._dynamic_ngrid``).

        Returns
        -------
        gamma0_arr : np.ndarray
            Pure IFTs [mN/m], shape (N,).
        rhoL0_arr : np.ndarray
            Liquid densities [mol/cm3], shape (N,).
        rhoV0_arr : np.ndarray
            Vapour densities [mol/cm3], shape (N,).
        Tc_arr : np.ndarray
            Critical temperatures [K], shape (N,).
        Pc_arr : np.ndarray
            Critical pressures [bar], shape (N,).
        Psat_arr : np.ndarray
            Saturation pressures [bar], shape (N,).
        """

        N = len(component_names)
        gamma0_arr = np.full(N, np.nan)
        rhoL0_arr = np.full(N, np.nan)
        rhoV0_arr = np.full(N, np.nan)
        Tc_arr = np.full(N, np.nan)
        Pc_arr = np.full(N, np.nan)
        Psat_arr = np.full(N, np.nan)

        for i, name in enumerate(component_names):
            gamma0_arr[i], rhoL0_arr[i], rhoV0_arr[i], Tc_arr[i], Pc_arr[i], Psat_arr[i] = (
                self._pure_component_cDFT(name, T_K, dynamic_lgrid=dynamic_lgrid, dynamic_ngrid=dynamic_ngrid)
            )

        self.component_names = list(component_names)
        self.T_K_cached = T_K
        self.gamma0 = gamma0_arr
        self.rhoL0 = rhoL0_arr
        self.rhoV0 = rhoV0_arr
        self.Tc = Tc_arr
        self.Pc = Pc_arr
        self.Psat = Psat_arr

        return gamma0_arr, rhoL0_arr, rhoV0_arr, Tc_arr, Pc_arr, Psat_arr

    def batch_pure_component_VLE(
        self,
        component_names: list[str],
        T_K: float,
        gamma0_ref: np.ndarray | None = None,
        scaling_exp: float = 1.26,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast pure-component VLE calculation WITHOUT cDFT.

        Uses pure VLE for density data and a scaling correlation for gamma0:
            gamma0(T) = gamma0_ref * (1 - T/Tc)^scaling_exp

        This is ~100x faster than batch_pure_component_cDFT because it skips
        the expensive PlanarInterface DFT solve.

        Parameters
        ----------
        component_names : list[str]
            List of component names.
        T_K : float
            Temperature [K].
        gamma0_ref : np.ndarray or None
            Reference pure IFT values [mN/m] at some reference state (typically
            computed once via cDFT at a low Tr). If None, uses Brock-Bird correlation.
        scaling_exp : float
            Exponent for the (1 - Tr)^n scaling. Default 1.26 (van der Waals).
            Use 11/9 ≈ 1.222 for Brock-Bird.

        Returns
        -------
        Same as batch_pure_component_cDFT.
        """
        N = len(component_names)
        gamma0_arr = np.full(N, np.nan)
        rhoL0_arr = np.full(N, np.nan)
        rhoV0_arr = np.full(N, np.nan)
        Tc_arr = np.full(N, np.nan)
        Pc_arr = np.full(N, np.nan)
        Psat_arr = np.full(N, np.nan)

        for i, name in enumerate(component_names):
            params = self._parameters_fn([name])
            eos = feos.HelmholtzEnergyFunctional.pcsaft(params)

            # Critical point (fast)
            cp = feos.State.critical_point(eos)
            Tc_K = float(cp.temperature / si.KELVIN)
            Pc_bar = float(cp.pressure() / si.BAR)
            Tc_arr[i] = Tc_K
            Pc_arr[i] = Pc_bar

            if T_K >= Tc_K:
                # Supercritical - no VLE
                continue

            # Pure VLE (fast - no cDFT)
            try:
                vle = feos.PhaseEquilibrium.pure(eos, T_K * si.KELVIN)
                rhoL0_arr[i] = self._density_fn(vle.liquid.density) * 1e-6
                rhoV0_arr[i] = self._density_fn(vle.vapor.density) * 1e-6
                Psat_arr[i]  = float(vle.liquid.pressure() / si.BAR)
                # Guard against silent EOS failures (e.g. P ~ 1e-237 bar).
                if not (np.isfinite(Psat_arr[i]) and Psat_arr[i] > 1e-6):
                    rhoL0_arr[i] = np.nan
                    rhoV0_arr[i] = np.nan
                    Psat_arr[i]  = np.nan
                    continue
            except Exception:
                continue

            # Estimate gamma0 using scaling correlation
            Tr = T_K / Tc_K
            if gamma0_ref is not None and i < len(gamma0_ref) and not np.isnan(gamma0_ref[i]):
                # Use provided reference value with scaling
                gamma0_arr[i] = gamma0_ref[i] * (1.0 - Tr) ** scaling_exp
            else:
                # Brock-Bird correlation (no reference needed)
                # gamma0 ≈ 0.073 * Pc^(2/3) * Tc^(1/3) * (1 - Tr)^1.26
                # Pc in bar -> convert to mN/m units
                gamma0_arr[i] = 0.073 * (Pc_bar * 0.1) ** (2/3) * Tc_K ** (1/3) * (1.0 - Tr) ** scaling_exp

        # Cache results
        self.component_names = list(component_names)
        self.T_K_cached = T_K
        self.gamma0 = gamma0_arr
        self.rhoL0 = rhoL0_arr
        self.rhoV0 = rhoV0_arr
        self.Tc = Tc_arr
        self.Pc = Pc_arr
        self.Psat = Psat_arr

        return gamma0_arr, rhoL0_arr, rhoV0_arr, Tc_arr, Pc_arr, Psat_arr

    # ---------------------------------------------------------------------------
    # Parachor method
    # ---------------------------------------------------------------------------

    @staticmethod
    def parachor_number(
        rho_l: float,
        rho_v: float,
        gamma: float,
        n_exp: float = 3.87,
    ) -> float:
        """
        Calculate the parachor number for a pure component.

        The parachor number P is defined such that:
            gamma^(1/n) = P * (rho_l - rho_v)

        Rearranging:
            P = gamma^(1/n) / (rho_l - rho_v)

        Parameters
        ----------
        rho_l : float
            Saturated liquid molar density [mol/cm3].
        rho_v : float
            Saturated vapour molar density [mol/cm3].
        gamma : float
            Pure-component interfacial tension [mN/m].
        n_exp : float, optional
            Parachor exponent (default 3.87, REFPROP v10).

        Returns
        -------
        parachor : float
            Parachor number [(mN/m)^(1/n) / (mol/cm3)].

        Raises
        ------
        ValueError
            If any input is non-positive or if rho_l <= rho_v.
        """
        if rho_l <= 0 or rho_v < 0 or gamma <= 0:
            raise ValueError("rho_l, gamma must be positive; rho_v must be non-negative.")

        delta_rho = rho_l - rho_v
        if delta_rho <= 0:
            raise ValueError("rho_l must be greater than rho_v.")

        parachor = gamma ** (1.0 / n_exp) / delta_rho
        return parachor

    def batch_parachor_numbers(
        self,
        component_names: list[str],
        T_K: float,
        n_exp: float = 3.87,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Compute parachor numbers for multiple components at a given temperature.

        For components where T_K > Tc (supercritical), the method automatically
        uses a reference temperature T_ref = 0.9 * Tc to compute a valid
        parachor number.

        Parameters
        ----------
        component_names : list[str]
            List of component names accepted by FeosPlugin.PARAMETERS.
        T_K : float
            Temperature [K].
        n_exp : float, optional
            Parachor exponent (default 3.87, REFPROP v10).
        verbose : bool, optional
            If True, print info about each component (default True).

        Returns
        -------
        parachor_numbers : np.ndarray
            Array of parachor numbers, shape (N,).
        """
        # Get pure-component data at T_K
        gamma0, rhoL0, rhoV0, Tc0, _, _ = self.batch_pure_component_cDFT(
            component_names, T_K
        )

        parachor_numbers = []
        for i, comp in enumerate(component_names):
            if np.isnan(gamma0[i]):
                # Component is supercritical at T_K, use T = 0.9*Tc instead
                # Use _pure_component_cDFT directly to avoid overwriting cached arrays
                T_ref = 0.9 * Tc0[i]
                gamma_ref, rhoL_ref, rhoV_ref, _, _, _ = self._pure_component_cDFT(
                    comp, T_ref
                )
                P_i = self.parachor_number(rhoL_ref, rhoV_ref, gamma_ref, n_exp)
                if verbose:
                    print(
                        f"{comp}: T_K={T_K:.2f} > Tc={Tc0[i]:.2f}, "
                        f"using T_ref={T_ref:.2f} K -> Parachor = {P_i:.4f}"
                    )
            else:
                P_i = self.parachor_number(rhoL0[i], rhoV0[i], gamma0[i], n_exp)
                if verbose:
                    print(f"{comp}: Parachor = {P_i:.4f}")
            parachor_numbers.append(P_i)

        return np.array(parachor_numbers, dtype=float)

    @staticmethod
    def parachor_mixture_IFT(
        x: np.ndarray,
        y: np.ndarray,
        rho_l: float,
        rho_v: float,
        parachor_numbers: np.ndarray,
        kij: float = 0.0,
        n_exp: float = 3.87,
    ) -> float:
        """
        Mixture IFT computed using the parachor method.

        Parameters
        ----------
        x, y : array
            Liquid / vapour mole fractions.
        rho_l, rho_v : float
            Mixture molar densities [mol/cm3].
        parachor_numbers : array
            Pure-component Parachor numbers.
        kij : float or 2D array-like, optional
            Binary interaction parameter(s) for the Parachor model.
        n_exp : float, optional
            Parachor exponent (default 3.87, REFPROP v10).

        Returns
        -------
        gamma_parachor : float
            Mixture IFT [mN/m]. NaN if delta <= 0.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        P = np.asarray(parachor_numbers, dtype=float)

        if not (len(x) == len(y) == len(P)):
            raise ValueError("x, y, and parachor_numbers must have the same length.")

        N = len(x)

        def _kij(i: int, j: int) -> float:
            return float(kij[i][j]) if np.ndim(kij) > 0 else float(kij)

        def _Pij(i: int, j: int) -> float:
            return P[i] if i == j else (1.0 - _kij(i, j)) * 0.5 * (P[i] + P[j])

        P_l = sum(x[i] * x[j] * _Pij(i, j) for i in range(N) for j in range(N))
        P_v = sum(y[i] * y[j] * _Pij(i, j) for i in range(N) for j in range(N))

        delta = rho_l * P_l - rho_v * P_v
        return np.nan if delta <= 0.0 else delta**n_exp

    # ---------------------------------------------------------------------------
    # Winterfeld - Scriven - Davis (WSD) method
    # ---------------------------------------------------------------------------

    def wsd_mixture_IFT(
        self,
        T_K: float,
        x: np.ndarray,
        y: np.ndarray,
        rho_l: float,
        rho_v: float,
        phi: float = 1.0,
        correction: bool = True,
    ) -> tuple[float, float]:
        """
        Mixture IFT computed using the Winterfeld-Scriven-Davis (WSD) method.
        DOI:10.1002/aic.690240610

        Matrix form:
            gamma_mix = a^T G a

        where
            a_i    = (x_i*rho_l - y_i*rho_v) / (rhoL0_i - rhoV0_i)
            G_ii   = gamma0_i
            G_ij   = phi_ij * sqrt(gamma0_i * gamma0_j), i != j

        Components with T > Tc_i are treated as supercritical: a_i = 0 and
        their G row/column is zeroed out, so they do not contribute.

        Requires batch_pure_component_cDFT() to have been called first.

        Parameters
        ----------
        T_K : float
            Temperature [K].
        x, y : array
            Liquid / vapour mole fractions, length N.
        rho_l, rho_v : float
            Mixture molar densities [mol/cm3].
        phi : float or 2-D array-like, optional
            Mixing parameter(s). Scalar means same value for all off-diagonal
            pairs. 2-D array uses phi[i][j].
        correction : bool, optional
            Whether to apply supercritical correction.

        Returns
        -------
        gamma_wsd : float
            Mixture IFT [mN/m].
        mixcorr : float
            Correction factor applied (1.0 if none).
        """
        if any(v is None for v in (self.gamma0, self.rhoL0, self.rhoV0, self.Tc)):
            raise RuntimeError(
                "Pure-component data not available. "
                "Call batch_pure_component_cDFT() before wsd_mixture_IFT()."
            )

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        gamma0 = self.gamma0
        rhoL0 = self.rhoL0
        rhoV0 = self.rhoV0
        Tc = self.Tc
        N = len(x)

        if not (len(y) == len(gamma0) == len(rhoL0) == len(rhoV0) == len(Tc) == N):
            raise ValueError("x, y and cached pure-component arrays must all have length N.")

        active = [T_K <= Tc_i for Tc_i in Tc]

        if not any(active):
            return 0.0, 1.0

        n0 = np.where(
            active,
            rhoL0 - rhoV0,
            1.0,
        )

        for i in range(N):
            if active[i] and abs(n0[i]) < 1e-30:
                warnings.warn(
                    f"[wsd_mixture_IFT] n0[{i}] ≈ 0 (near critical?). "
                    "Treating component as supercritical."
                )
                active[i] = False
                n0[i] = 1.0

        a = (x * rho_l - y * rho_v) / n0
        for i in range(N):
            if not active[i]:
                a[i] = 0.0

        def _phi(i: int, j: int) -> float:
            if hasattr(phi, "__len__"):
                return float(phi[i][j])
            return float(phi)

        G = np.zeros((N, N), dtype=float)
        for i in range(N):
            if active[i]:
                G[i, i] = gamma0[i]
            for j in range(i + 1, N):
                gij = 0.0
                if active[i] and active[j]:
                    gij = _phi(i, j) * np.sqrt(gamma0[i] * gamma0[j])
                G[i, j] = G[j, i] = gij

        gamma_mix = float(a @ G @ a)

        mixcorr = 1.0
        if correction and not all(active):
            sc_indices = [i for i in range(N) if not active[i]]
            x_inactive_sum = float(np.sum(x[sc_indices]))
            mixcorr = max(0.0, 1.0 - x_inactive_sum)
            gamma_mix *= mixcorr

        return gamma_mix, mixcorr

    # ---------------------------------------------------------------------------
    # Redlich-Kister (RK)
    # ---------------------------------------------------------------------------

    def RK_mixture_IFT(
        self,
        T_K: float,
        x: np.ndarray,
        rk_coeffs: np.ndarray,
    ) -> float:
        """
        Mixture IFT via the Redlich-Kister (RK) empirical method.

        Formula:
            sigma_mix = sum_i x_i * sigma_i
                      + sum_i sum_{j>i} x_i * x_j * Q_ij(x_i, x_j)

        where the pairwise excess term is a power series in (x_j - x_i):
            Q_ij = sum_{k=0}^{M-1} A_ij[k] * (x_j - x_i)^k

        M can be 2, 3, 4, or 5.

        Components with T > Tc_i are treated as supercritical (sigma0_i = 0).

        Requires batch_pure_component_cDFT() to have been called first.

        Parameters
        ----------
        T_K : float
            Temperature [K].
        x : array
            Liquid mole fractions, shape (N,).
        rk_coeffs : array
            Fitted RK coefficients [A, B, C, ...].

        Returns
        -------
        gamma_mix : float
            Mixture IFT [mN/m]. 0.0 if all components are supercritical.
        """
        if self.gamma0 is None or self.Tc is None:
            raise RuntimeError(
                "Pure-component data not available. "
                "Call batch_pure_component_cDFT() before RK_mixture_IFT()."
            )

        x = np.asarray(x, dtype=float)
        coeffs_arr = np.asarray(rk_coeffs, dtype=float)
        gamma0 = self.gamma0
        Tc = self.Tc
        N = len(x)

        if len(gamma0) != N or len(Tc) != N:
            raise ValueError("x and cached pure-component arrays must all have the same length.")

        M = len(coeffs_arr)
        if M not in (2, 3, 4, 5):
            raise ValueError(f"rk_coeffs has {M} coefficients; must be 2, 3, 4, or 5.")

        active = [T_K <= Tc_i for Tc_i in Tc]

        if not any(active):
            return 0.0

        sigma = np.where(active, gamma0, 0.0)
        sigma_linear = float(np.dot(x, sigma))

        x1, x2 = x[0], x[1]
        delta = x2 - x1
        powers = delta ** np.arange(M)
        Q = float(np.dot(coeffs_arr, powers))

        sigma_excess = x1 * x2 * Q
        sigma_mix = sigma_linear + sigma_excess

        return float(sigma_mix)

    # ---------------------------------------------------------------------------
    # PT sweep - collect (T, P, gamma) over the two-phase region
    # ---------------------------------------------------------------------------

    def compute_PT_colormap_data(
        self,
        component_names: list[str],
        feed_z: np.ndarray,
        eos,
        T_bub: list[float],
        P_bub: list[float],
        T_dew: list[float],
        P_dew: list[float],
        parachor_numbers: np.ndarray | None = None,
        kij_parachor: float = 0.0,
        n_exp: float = 3.87,
        phi_wsd: float = 1.0,
        wsd_correction: bool = True,
        rk_coeffs: np.ndarray | None = None,
        method: str = "parachor",
        n_T: int = 30,
        n_P: int = 30,
        recompute_parachor: bool = True,
        kij_builder=None,
        model_map: dict | None = None,
        T_grid: np.ndarray | None = None,
    ) -> dict:
        """
        Sweep a (T, P) grid inside the two-phase envelope, run a TP flash at
        each point, and compute mixture IFT using the chosen semi-empirical method.

        Parameters
        ----------
        component_names : list[str]
            Names of the components in the mixture (e.g., ["methane", "ethane"]).
            Must match parameter file entries.
        feed_z : np.ndarray
            Overall feed composition as mole fractions, shape (N,). Must sum to 1.0.
        eos : feos.EquationOfState
            The equation of state object (e.g., PC-SAFT) used for TP flash calculations.
        T_bub : list[float]
            Bubble-point temperatures [K] defining the phase envelope boundary.
        P_bub : list[float]
            Bubble-point pressures [bar] corresponding to T_bub.
        T_dew : list[float]
            Dew-point temperatures [K] defining the phase envelope boundary.
        P_dew : list[float]
            Dew-point pressures [bar] corresponding to T_dew.
        parachor_numbers : np.ndarray or None, optional
            Parachor numbers for each component, shape (N,). Required when
            method='parachor' and recompute_parachor=False. Default is None.
        kij_parachor : float, optional
            Binary interaction parameter for the parachor mixing rule. Default is 0.0.
        n_exp : float, optional
            Exponent in the parachor IFT equation. Default is 3.87.
        phi_wsd : float, optional
            Correction factor (phi) for the WSD method. Default is 1.0.
        wsd_correction : bool, optional
            Whether to apply correction in the WSD method. Default is True.
        rk_coeffs : np.ndarray or None, optional
            Redlich-Kister coefficients for the RK method. Required when method='rk'.
            Default is None.
        method : str, optional
            Semi-empirical method to use. Options are:
            - 'parachor': Parachor-based correlation (default)
            - 'wsd': Winterfeld-Scriven-Davis method
            - 'rk': Redlich-Kister polynomial expansion
        n_T : int, optional
            Number of temperature grid points in the sweep. Default is 30.
        n_P : int, optional
            Number of pressure grid points at each temperature. Default is 30.
        recompute_parachor : bool, optional
            If True (default), recompute parachor numbers at each temperature
            using batch_parachor_numbers. If False, use the fixed parachor_numbers
            array for all temperatures.
        kij_builder : KIJMatrixBuilder or None, optional
            If provided, rebuilds the PC-SAFT eos with T-dependent kij at each
            temperature in the sweep. Requires model_map. Default is None (use
            the single eos for all temperatures).
        model_map : dict or None, optional
            Maps component pairs to kij model type (e.g., {("CO2","Ar"): "constant"}).
            Only used when kij_builder is not None.
        T_grid : np.ndarray or None, optional
            Explicit temperature grid [K] to sweep.  When provided, *n_T* is
            ignored and these exact temperatures are used.  Pass the same array
            to the cDFT sweep to guarantee identical (T, P) grids.  Default is
            None (auto-generate from bubble/dew bounds).

        Returns
        -------
        dict
            Dictionary containing arrays of results with keys:
            - 'T': Temperature [K]
            - 'P': Pressure [bar]
            - 'gamma': Computed mixture IFT [mN/m]
            - 'rho_l': Liquid phase density [mol/cm3]
            - 'rho_v': Vapor phase density [mol/cm3]
            - 'x': Liquid mole fractions, shape (N,)
            - 'y': Vapor mole fractions, shape (N,)
            - 'gamma0': Pure component IFTs [mN/m]
            - 'rhoL0': Pure liquid densities [mol/cm3]
            - 'rhoV0': Pure vapor densities [mol/cm3]
            - 'Psat0': Pure saturation pressures [bar]
            - 'Tc': Critical temperatures [K]
            - 'Pc': Critical pressures [bar]
            - 'parachor': Parachor numbers used (or None)

        Raises
        ------
        ValueError
            If method is not one of 'parachor', 'wsd', or 'rk'.
            If parachor_numbers is None when method='parachor' and recompute_parachor=False.
            If rk_coeffs is None when method='rk'.

        Notes
        -----
        - The grid is constructed between the bubble and dew curves.
        - Points outside the two-phase region or where flash fails are skipped.
        - Warnings are issued for individual point failures without stopping the sweep.
        """
        method = method.lower().strip()

        if method not in ("parachor", "wsd", "rk"):
            raise ValueError(f"method must be 'parachor', 'wsd', or 'rk'; got '{method}'.")

        if method == "parachor" and parachor_numbers is None and not recompute_parachor:
            raise ValueError("parachor_numbers must be provided when method='parachor' and recompute_parachor=False.")

        if method == "rk" and rk_coeffs is None:
            raise ValueError("rk_coeffs must be provided when method='rk'.")

        feed_z = np.asarray(feed_z, dtype=float)
        feed_si = feed_z * si.MOL

        data = {
            k: []
            for k in (
                "T",
                "P",
                "gamma",
                "rho_l",
                "rho_v",
                "x",
                "y",
                "gamma0",
                "rhoL0",
                "rhoV0",
                "Psat0",
                "Tc",
                "Pc",
                "parachor",
            )
        }

        T_bub_arr, P_bub_arr = zip(*sorted(zip(T_bub, P_bub)))
        T_dew_arr, P_dew_arr = zip(*sorted(zip(T_dew, P_dew)))

        if T_grid is not None:
            T_range = np.asarray(T_grid, dtype=float)
        else:
            T_min = max(min(T_bub_arr), min(T_dew_arr))
            T_max = min(max(T_bub_arr), max(T_dew_arr)) * 0.999
            T_range = np.linspace(T_min, T_max, n_T)
        _cached_T = None
        _current_parachor = parachor_numbers  # start with user-supplied or None
        _current_eos = eos  # may be rebuilt per T when kij_builder is provided

        for T_K in T_range:

            P_bub_T = float(np.interp(T_K, T_bub_arr, P_bub_arr))
            P_dew_T = float(np.interp(T_K, T_dew_arr, P_dew_arr))

            if P_bub_T <= P_dew_T:
                continue

            P_range = np.linspace(P_dew_T * 1.01, P_bub_T * 0.99, n_P)

            if T_K != _cached_T:
                try:
                    self.batch_pure_component_cDFT(component_names, T_K)
                    # Recompute parachor numbers at this temperature if requested
                    if method == "parachor" and recompute_parachor:
                        _current_parachor = self.batch_parachor_numbers(
                            component_names, T_K, n_exp=n_exp, verbose=False
                        )
                    # Rebuild eos with T-dependent kij if builder provided
                    if kij_builder is not None:
                        params_T = PARAMETERS(
                            component_names, T_K=float(T_K),
                            kij_builder=kij_builder, model_map=model_map,
                        )
                        _current_eos = feos.HelmholtzEnergyFunctional.pcsaft(params_T)
                    _cached_T = T_K
                except Exception as exc:
                    warnings.warn(
                        f"[compute_PT_colormap_data] pure cache failed at T={T_K:.1f} K: {exc}"
                    )
                    continue

            for P_bar in P_range:

                T_si = T_K * si.KELVIN
                P_si = P_bar * si.BAR

                try:
                    eq = feos.PhaseEquilibrium.tp_flash(_current_eos, T_si, P_si, feed_si)
                except Exception:
                    continue

                liq = eq.liquid
                vap = eq.vapor
                x = np.array(liq.molefracs, dtype=float)
                y = np.array(vap.molefracs, dtype=float)

                rho_l = self._density_fn(liq.density) * 1e-6
                rho_v = self._density_fn(vap.density) * 1e-6

                try:
                    if method == "parachor":
                        gamma = self.parachor_mixture_IFT(
                            x,
                            y,
                            rho_l,
                            rho_v,
                            parachor_numbers=_current_parachor,
                            kij=kij_parachor,
                            n_exp=n_exp,
                        )

                    elif method == "wsd":
                        gamma, _ = self.wsd_mixture_IFT(
                            T_K,
                            x,
                            y,
                            rho_l,
                            rho_v,
                            phi=phi_wsd,
                            correction=wsd_correction,
                        )

                    elif method == "rk":
                        gamma = self.RK_mixture_IFT(T_K, x, rk_coeffs)

                    else:
                        raise ValueError(f"Unknown method '{method}'.")

                except Exception as exc:
                    warnings.warn(
                        f"[compute_PT_colormap_data] IFT failed at "
                        f"T={T_K:.1f} K, P={P_bar:.1f} bar: {exc}"
                    )
                    continue

                if np.isnan(gamma) or gamma <= 0.0:
                    continue

                data["T"].append(T_K)
                data["P"].append(P_bar)
                data["gamma"].append(gamma)
                data["rho_l"].append(rho_l)
                data["rho_v"].append(rho_v)
                data["x"].append(x.copy())
                data["y"].append(y.copy())
                data["gamma0"].append(self.gamma0.copy())
                data["rhoL0"].append(self.rhoL0.copy())
                data["rhoV0"].append(self.rhoV0.copy())
                data["Psat0"].append(self.Psat.copy())
                data["Tc"].append(self.Tc.copy())
                data["Pc"].append(self.Pc.copy())
                data["parachor"].append(_current_parachor.copy() if _current_parachor is not None else None)

        return data

    # ---------------------------------------------------------------------------
    # Saturation line IFT - compute gamma only at bubble and dew points
    # ---------------------------------------------------------------------------

    def compute_saturation_line_IFT(
        self,
        component_names: list[str],
        feed_z: np.ndarray,
        eos,
        T_bub: list[float],
        P_bub: list[float],
        T_dew: list[float],
        P_dew: list[float],
        parachor_numbers: np.ndarray | None = None,
        kij_parachor: float = 0.0,
        n_exp: float = 3.87,
        phi_wsd: float = 1.0,
        rk_coeffs: np.ndarray | None = None,
        method: str = "parachor",
        recompute_parachor: bool = True,
        line: str = "both",
        n_points: int | None = None,
        P_offset_frac: float = 0.005,
        kij_builder=None,
        model_map: dict | None = None,
    ) -> dict:
        """
        Compute mixture IFT only along the saturation lines (bubble and/or dew curves).

        Unlike compute_PT_colormap_data which sweeps the entire two-phase region,
        this function computes IFT only at the (T, P) points on the phase envelope.

        Parameters
        ----------
        component_names : list[str]
            Names of the components in the mixture (e.g., ["methane", "ethane"]).
        feed_z : np.ndarray
            Overall feed composition as mole fractions, shape (N,). Must sum to 1.0.
        eos : feos.EquationOfState
            The equation of state object (e.g., PC-SAFT) used for TP flash calculations.
        T_bub : list[float]
            Bubble-point temperatures [K] defining the bubble curve.
        P_bub : list[float]
            Bubble-point pressures [bar] corresponding to T_bub.
        T_dew : list[float]
            Dew-point temperatures [K] defining the dew curve.
        P_dew : list[float]
            Dew-point pressures [bar] corresponding to T_dew.
        parachor_numbers : np.ndarray or None, optional
            Parachor numbers for each component, shape (N,). Required when
            method='parachor' and recompute_parachor=False. Default is None.
        kij_parachor : float, optional
            Binary interaction parameter for the parachor mixing rule. Default is 0.0.
        n_exp : float, optional
            Exponent in the parachor IFT equation. Default is 3.87.
        phi_wsd : float, optional
            Correction factor (phi) for the WSD method. Default is 1.0.
        rk_coeffs : np.ndarray or None, optional
            Redlich-Kister coefficients for the RK method. Required when method='rk'.
        method : str, optional
            Semi-empirical method to use: 'parachor', 'wsd', or 'rk'. Default is 'parachor'.
        recompute_parachor : bool, optional
            If True (default), recompute parachor numbers at each temperature.
        line : str, optional
            Which saturation line(s) to compute: 'bubble', 'dew', or 'both'. Default is 'both'.
        n_points : int or None, optional
            Number of temperature grid points to use. If None (default), uses all provided
            saturation points. Set to e.g. 50 for faster computation with interpolation.
        P_offset_frac : float, optional
            Fractional pressure offset to stay inside two-phase region. Default is 0.005
            (0.5%). For bubble line uses P * (1 - offset), for dew line uses P * (1 + offset).
        kij_builder : KIJMatrixBuilder or None, optional
            If provided, rebuilds the PC-SAFT eos with T-dependent kij at each
            temperature. Requires model_map. Default is None.
        model_map : dict or None, optional
            Maps component pairs to kij model type. Only used when kij_builder is not None.

        Returns
        -------
        dict
            Dictionary with keys:
            - 'bubble': dict with T, P, gamma, gamma_wsd_corrected, rho_l, rho_v, x, y, etc.
            - 'dew': dict with T, P, gamma, gamma_wsd_corrected, rho_l, rho_v, x, y, etc.
            For WSD method, both 'gamma' (uncorrected) and 'gamma_wsd_corrected' are computed
            in a single pass to save computation. Empty dict for a line if line parameter excludes it.
        """
        method = method.lower().strip()
        line = line.lower().strip()

        if method not in ("parachor", "wsd", "rk"):
            raise ValueError(f"method must be 'parachor', 'wsd', or 'rk'; got '{method}'.")

        if line not in ("bubble", "dew", "both"):
            raise ValueError(f"line must be 'bubble', 'dew', or 'both'; got '{line}'.")

        if method == "parachor" and parachor_numbers is None and not recompute_parachor:
            raise ValueError("parachor_numbers must be provided when method='parachor' and recompute_parachor=False.")

        if method == "rk" and rk_coeffs is None:
            raise ValueError("rk_coeffs must be provided when method='rk'.")

        feed_z = np.asarray(feed_z, dtype=float)
        feed_si = feed_z * si.MOL

        def _empty_data() -> dict:
            return {
                k: []
                for k in (
                    "T", "P", "gamma", "gamma_wsd_corrected", "rho_l", "rho_v", "x", "y",
                    "gamma0", "rhoL0", "rhoV0", "Psat0", "Tc", "Pc", "parachor",
                )
            }

        def _compute_line(T_arr: list[float], P_arr: list[float], is_bubble: bool) -> dict:
            """Compute IFT along a single saturation line."""
            data = _empty_data()
            _cached_T = None
            _current_parachor = parachor_numbers
            _current_eos = eos  # may be rebuilt per T when kij_builder is provided
            _pure_cache_ok = False  # Track if pure component cache is valid

            # Sort by temperature and create interpolation arrays
            sorted_pairs = sorted(zip(T_arr, P_arr))
            T_sorted = np.array([p[0] for p in sorted_pairs])
            P_sorted = np.array([p[1] for p in sorted_pairs])

            # Use coarser grid if n_points specified (for speed)
            if n_points is not None and len(T_sorted) > n_points:
                T_grid = np.linspace(T_sorted.min(), T_sorted.max() * 0.999, n_points)
                P_grid = np.interp(T_grid, T_sorted, P_sorted)
            else:
                T_grid = T_sorted
                P_grid = P_sorted

            for T_K, P_bar in zip(T_grid, P_grid):
                # Apply pressure offset to stay inside two-phase region
                # Bubble: P slightly lower, Dew: P slightly higher
                if is_bubble:
                    P_bar_flash = P_bar * (1.0 - P_offset_frac)
                else:
                    P_bar_flash = P_bar * (1.0 + P_offset_frac)

                T_si = T_K * si.KELVIN
                P_si = P_bar_flash * si.BAR

                # Update pure component cache and parachor if temperature changed
                if T_K != _cached_T:
                    _pure_cache_ok = False
                    try:
                        self.batch_pure_component_cDFT(component_names, T_K)
                        _pure_cache_ok = True
                    except Exception as exc:
                        warnings.warn(
                            f"[compute_saturation_line_IFT] pure cache failed at T={T_K:.1f} K: {exc}"
                        )
                        # For parachor method, pure component data is required
                        if method == "parachor":
                            continue

                    if _pure_cache_ok and method == "parachor" and recompute_parachor:
                        try:
                            _current_parachor = self.batch_parachor_numbers(
                                component_names, T_K, n_exp=n_exp, verbose=False
                            )
                        except Exception as exc:
                            warnings.warn(
                                f"[compute_saturation_line_IFT] parachor computation failed at T={T_K:.1f} K: {exc}"
                            )
                            continue

                    # Rebuild eos with T-dependent kij if builder provided
                    if kij_builder is not None:
                        params_T = PARAMETERS(
                            component_names, T_K=float(T_K),
                            kij_builder=kij_builder, model_map=model_map,
                        )
                        _current_eos = feos.HelmholtzEnergyFunctional.pcsaft(params_T)

                    _cached_T = T_K

                # TP flash at saturation point
                try:
                    eq = feos.PhaseEquilibrium.tp_flash(_current_eos, T_si, P_si, feed_si)
                except Exception:
                    continue

                liq = eq.liquid
                vap = eq.vapor
                x = np.array(liq.molefracs, dtype=float)
                y = np.array(vap.molefracs, dtype=float)

                rho_l = self._density_fn(liq.density) * 1e-6
                rho_v = self._density_fn(vap.density) * 1e-6

                # Compute IFT using selected method
                gamma_wsd_corrected = np.nan  # Only used for WSD method
                try:
                    if method == "parachor":
                        gamma = self.parachor_mixture_IFT(
                            x, y, rho_l, rho_v,
                            parachor_numbers=_current_parachor,
                            kij=kij_parachor,
                            n_exp=n_exp,
                        )
                    elif method == "wsd":
                        # Compute both uncorrected and corrected WSD in one pass
                        # (saves the expensive TP flash and cDFT, only wsd_mixture_IFT is called twice)
                        gamma, _ = self.wsd_mixture_IFT(
                            T_K, x, y, rho_l, rho_v,
                            phi=phi_wsd,
                            correction=False,
                        )
                        gamma_wsd_corrected, _ = self.wsd_mixture_IFT(
                            T_K, x, y, rho_l, rho_v,
                            phi=phi_wsd,
                            correction=True,
                        )
                    elif method == "rk":
                        gamma = self.RK_mixture_IFT(T_K, x, rk_coeffs)
                    else:
                        raise ValueError(f"Unknown method '{method}'.")

                except Exception as exc:
                    warnings.warn(
                        f"[compute_saturation_line_IFT] IFT failed at "
                        f"T={T_K:.1f} K, P={P_bar:.1f} bar: {exc}"
                    )
                    continue

                if np.isnan(gamma) or gamma <= 0.0:
                    continue

                data["T"].append(T_K)
                data["P"].append(P_bar)
                data["gamma"].append(gamma)
                data["gamma_wsd_corrected"].append(gamma_wsd_corrected)
                data["rho_l"].append(rho_l)
                data["rho_v"].append(rho_v)
                data["x"].append(x.copy())
                data["y"].append(y.copy())
                # Pure component data (may be NaN if cDFT failed for non-parachor methods)
                if _pure_cache_ok:
                    data["gamma0"].append(self.gamma0.copy())
                    data["rhoL0"].append(self.rhoL0.copy())
                    data["rhoV0"].append(self.rhoV0.copy())
                    data["Psat0"].append(self.Psat.copy())
                    data["Tc"].append(self.Tc.copy())
                    data["Pc"].append(self.Pc.copy())
                else:
                    n_comps = len(component_names)
                    data["gamma0"].append(np.full(n_comps, np.nan))
                    data["rhoL0"].append(np.full(n_comps, np.nan))
                    data["rhoV0"].append(np.full(n_comps, np.nan))
                    data["Psat0"].append(np.full(n_comps, np.nan))
                    data["Tc"].append(np.full(n_comps, np.nan))
                    data["Pc"].append(np.full(n_comps, np.nan))
                data["parachor"].append(_current_parachor.copy() if _current_parachor is not None else None)

            return data

        result = {"bubble": {}, "dew": {}}

        if line in ("bubble", "both"):
            result["bubble"] = _compute_line(T_bub, P_bub, is_bubble=True)

        if line in ("dew", "both"):
            result["dew"] = _compute_line(T_dew, P_dew, is_bubble=False)

        return result

    @staticmethod
    def saturation_line_to_dataframe(
        data: dict,
        component_names: list[str],
        line: str = "both",
    ) -> pd.DataFrame:
        """
        Convert saturation line data to a pandas DataFrame.

        Parameters
        ----------
        data : dict
            Output from compute_saturation_line_IFT.
        component_names : list[str]
            Names of the components.
        line : str, optional
            Which line to convert: 'bubble', 'dew', or 'both'. Default is 'both'.

        Returns
        -------
        pd.DataFrame
            DataFrame with saturation line IFT data. If 'both', includes a 'line' column.
        """
        line = line.lower().strip()

        def _to_rows(line_data: dict, line_name: str) -> dict:
            if not line_data or not line_data.get("T"):
                return {}

            rows = {
                "T": line_data["T"],
                "P": line_data["P"],
                "gamma_mix": line_data["gamma"],
                "gamma_wsd_corrected": line_data.get("gamma_wsd_corrected", [np.nan] * len(line_data["T"])),
                "rho_l": line_data["rho_l"],
                "rho_v": line_data["rho_v"],
                "line": [line_name] * len(line_data["T"]),
            }

            for i, name in enumerate(component_names):
                rows[f"x_{name}"] = [v[i] for v in line_data["x"]]
                rows[f"y_{name}"] = [v[i] for v in line_data["y"]]
                # Add partial densities for all components: rhoLi = x[i] * rho_l, rhoVi = y[i] * rho_v
                # Use i+1 for indexing since 0 is reserved for pure properties (rhoL0, rhoV0)
                rows[f"rhoL{i+1}_{name}"] = [v[i] * rho for v, rho in zip(line_data["x"], line_data["rho_l"])]
                rows[f"rhoV{i+1}_{name}"] = [v[i] * rho for v, rho in zip(line_data["y"], line_data["rho_v"])]
                rows[f"gamma0_{name}"] = [v[i] for v in line_data["gamma0"]]
                rows[f"rhoL0_{name}"] = [v[i] for v in line_data["rhoL0"]]
                rows[f"rhoV0_{name}"] = [v[i] for v in line_data["rhoV0"]]
                rows[f"Psat0_{name}"] = [v[i] for v in line_data["Psat0"]]
                rows[f"Tc_{name}"] = [v[i] for v in line_data["Tc"]]
                rows[f"Pc_{name}"] = [v[i] for v in line_data["Pc"]]
                rows[f"parachor_{name}"] = [v[i] if v is not None else None for v in line_data["parachor"]]

            return rows

        if line == "bubble":
            rows = _to_rows(data.get("bubble", {}), "bubble")
            if not rows:
                return pd.DataFrame()
            del rows["line"]
            return pd.DataFrame(rows)

        elif line == "dew":
            rows = _to_rows(data.get("dew", {}), "dew")
            if not rows:
                return pd.DataFrame()
            del rows["line"]
            return pd.DataFrame(rows)

        else:  # both
            bub_rows = _to_rows(data.get("bubble", {}), "bubble")
            dew_rows = _to_rows(data.get("dew", {}), "dew")

            if not bub_rows and not dew_rows:
                return pd.DataFrame()

            df_bub = pd.DataFrame(bub_rows) if bub_rows else pd.DataFrame()
            df_dew = pd.DataFrame(dew_rows) if dew_rows else pd.DataFrame()

            return pd.concat([df_bub, df_dew], ignore_index=True)

    @staticmethod
    def data_to_dataframe(data: dict, component_names: list[str]) -> pd.DataFrame:
        """
        Convert the dict returned by ``compute_PT_colormap_data`` into a flat
        pandas DataFrame.

        Parameters
        ----------
        data : dict
            Output dictionary from ``compute_PT_colormap_data`` containing lists
            keyed by 'T', 'P', 'gamma', 'rho_l', 'rho_v', 'x', 'y', 'gamma0',
            'rhoL0', 'rhoV0', 'Psat0', 'Tc', 'Pc', and 'parachor'.
        component_names : list[str]
            Names of the components, used to expand per-component array entries
            into individual columns (e.g. ``x_methane``, ``gamma0_ethane``).

        Returns
        -------
        pd.DataFrame
            Flat DataFrame with scalar columns for T, P, gamma_mix, rho_l, rho_v
            and per-component columns for mole fractions, pure-component IFTs,
            densities, saturation pressures, critical properties, and parachor
            numbers.
        """
        rows: dict[str, list] = {
            "T": data["T"],
            "P": data["P"],
            "gamma_mix": data["gamma"],
            "rho_l": data["rho_l"],
            "rho_v": data["rho_v"],
        }

        for i, name in enumerate(component_names):
            rows[f"x_{name}"] = [v[i] for v in data["x"]]
            rows[f"y_{name}"] = [v[i] for v in data["y"]]
            rows[f"gamma0_{name}"] = [v[i] for v in data["gamma0"]]
            rows[f"rhoL0_{name}"] = [v[i] for v in data["rhoL0"]]
            rows[f"rhoV0_{name}"] = [v[i] for v in data["rhoV0"]]
            rows[f"Psat0_{name}"] = [v[i] for v in data["Psat0"]]
            rows[f"Tc_{name}"] = [v[i] for v in data["Tc"]]
            rows[f"Pc_{name}"] = [v[i] for v in data["Pc"]]
            rows[f"parachor_{name}"] = [v[i] if v is not None else None for v in data["parachor"]]

        return pd.DataFrame(rows)
    
    @staticmethod
    def prepare_method_df(df: pd.DataFrame, line_name: str, method_suffix: str) -> pd.DataFrame:
        """
        Filter a saturation-line DataFrame by line type and suffix-rename columns.

        Selects rows where ``df["line"] == line_name`` and appends
        ``_<method_suffix>`` to every column except 'T' and 'line', so that
        DataFrames from different methods can be merged on 'T' without
        column-name collisions.

        Parameters
        ----------
        df : pd.DataFrame
            Combined saturation-line DataFrame (must contain a 'line' column),
            typically from ``saturation_line_to_dataframe(..., line='both')``.
        line_name : str
            Saturation line to keep: ``'bubble'`` or ``'dew'``.
        method_suffix : str
            Suffix appended to renamed columns (e.g. ``'parachor'``, ``'wsd'``).

        Returns
        -------
        pd.DataFrame
            Filtered and renamed copy of the input DataFrame.
        """
        out = df[df["line"] == line_name].copy()

        rename_map = {
            col: f"{col}_{method_suffix}"
            for col in out.columns
            if col not in ["T", "line"]
        }

        return out.rename(columns=rename_map)

    @staticmethod
    def save_saturation_dataframes(
        df_sat_parachor: pd.DataFrame,
        df_sat_wsd: pd.DataFrame,
        output_dir: str,
        component_names: list[str],
        feed_z: np.ndarray,
        n_exp: float = 3.87,
        kij_parachor: float = 0.0,
        ML_mode: bool = True,
        main_component_index: int = 0,
        all_components: list[str] | None = None,
        verbose: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Save saturation IFT DataFrames to separate CSV files by method and line.

        Since parachor and WSD methods may have different row counts, this saves
        them as separate files rather than merging.
 
        Parameters
        ----------
        df_sat_parachor : pd.DataFrame
            DataFrame from parachor method (from saturation_line_to_dataframe).
        df_sat_wsd : pd.DataFrame
            DataFrame from WSD method (contains gamma_mix and gamma_wsd_corrected).
        output_dir : str
            Directory to save CSV files. Creates if doesn't exist.
        component_names : list[str]
            Names of the active components in this mixture.
        feed_z : np.ndarray
            Overall feed composition as mole fractions, shape (N,). Must sum to 1.0.
        n_exp : float, optional
            Parachor exponent (default 3.87). Added as a column in parachor CSV.
        ML_mode : bool, optional
            If True (default), includes all columns needed for ML training:
            - T, P, rho_l, rho_v (mixture properties)
            - z_{comp}, x_{comp}, y_{comp} (compositions for all_components, zeros for inactive)
            - Pure properties only for main component (others are often supercritical)
            - parachor_{comp}, n_exp (for parachor method)
            If False, includes only T, P, and gamma columns.
        main_component_index : int, optional
            Index of the main component (default 0, typically CO2) for which
            pure properties (gamma0, rhoL0, rhoV0, Tc, Pc, Psat0) are included.
            Impurities are often supercritical and would have NaN values.
        all_components : list[str], optional
            Full list of all possible components for ML training. If provided,
            pads z, x, y, parachor columns with zeros for inactive components.
            This ensures consistent column structure across different mixtures.
            If None, only active component_names are included.
 
        Returns
        -------
        dict[str, pd.DataFrame]
            Dictionary with keys: 'bubble_parachor', 'bubble_wsd',
            'dew_parachor', 'dew_wsd'.
        """
        os.makedirs(output_dir, exist_ok=True)
        result = {}
        feed_z = np.asarray(feed_z, dtype=float)
 
        # Use all_components if provided, otherwise just active components
        output_components = all_components if all_components is not None else component_names
 
        # Main component for pure properties (impurities often supercritical)
        main_comp = component_names[main_component_index]
 
        # Build dictionaries for active components
        z_dict = {c: float(feed_z[i]) for i, c in enumerate(component_names)}
 
        # Define column groups for ML mode
        base_cols = ["T", "P", "rho_l", "rho_v"]
        # Only include pure properties for the main component
        pure_prop_cols = [
            f"gamma0_{main_comp}", f"rhoL0_{main_comp}", f"rhoV0_{main_comp}",
            f"Tc_{main_comp}", f"Pc_{main_comp}", f"Psat0_{main_comp}"
        ]
 
        def _add_padded_compositions(df: pd.DataFrame, include_parachor: bool) -> pd.DataFrame:
            """Add z, x, y (and optionally parachor) columns, padded with zeros for inactive components."""
            n_rows = len(df)
 
            # Feed composition (z) - constant for all rows
            for c in output_components:
                df[f"z_{c}"] = z_dict.get(c, 0.0)
 
            # Liquid mole fractions (x) - from DataFrame or zero
            for c in output_components:
                src_col = f"x_{c}"
                if src_col in df.columns:
                    pass  # Already present
                else:
                    df[f"x_{c}"] = 0.0
 
            # Vapor mole fractions (y) - from DataFrame or zero
            for c in output_components:
                src_col = f"y_{c}"
                if src_col in df.columns:
                    pass  # Already present
                else:
                    df[f"y_{c}"] = 0.0
 
            # Parachor values (if applicable)
            if include_parachor:
                for c in output_components:
                    src_col = f"parachor_{c}"
                    if src_col in df.columns:
                        pass  # Already present
                    else:
                        df[f"parachor_{c}"] = 0.0
 
            return df
 
        # Map each component name -> its source density column name (uses component_names index)
        _comp_to_src_rhoL = {c: f"rhoL{i+1}_{c}" for i, c in enumerate(component_names)}
        _comp_to_src_rhoV = {c: f"rhoV{i+1}_{c}" for i, c in enumerate(component_names)}
 
        def _add_padded_densities(df: pd.DataFrame) -> pd.DataFrame:
            """
            Add rhoL{j+1}_{c} and rhoV{j+1}_{c} columns for every component in
            output_components (indexed by position in output_components).
            Active components copy from the existing source column; inactive ones are 0.
            """
            for j, c in enumerate(output_components):
                dst_L = f"rhoL{j+1}_{c}"
                dst_V = f"rhoV{j+1}_{c}"
                src_L = _comp_to_src_rhoL.get(c)
                src_V = _comp_to_src_rhoV.get(c)
                # Liquid component density
                if src_L and src_L in df.columns:
                    df[dst_L] = df[src_L]
                    if dst_L != src_L:          # rename completed; drop the old name
                        df.drop(columns=[src_L], inplace=True)
                elif dst_L not in df.columns:
                    df[dst_L] = 0.0
                # Vapour component density
                if src_V and src_V in df.columns:
                    df[dst_V] = df[src_V]
                    if dst_V != src_V:
                        df.drop(columns=[src_V], inplace=True)
                elif dst_V not in df.columns:
                    df[dst_V] = 0.0
            return df
 
        def _reorder_columns_parachor(df: pd.DataFrame) -> pd.DataFrame:
            """Reorder columns for parachor method: T, P, z, rho_l, rho_v, x, y, rhoL_i, rhoV_i, pure_props, parachor, n_exp, gamma."""
            ordered = ["T", "P"]
            ordered += [f"z_{c}" for c in output_components]
            ordered += ["rho_l", "rho_v"]
            ordered += [f"x_{c}" for c in output_components]
            ordered += [f"y_{c}" for c in output_components]
            ordered += [c for c in pure_prop_cols if c in df.columns]
            ordered += [f"parachor_{c}" for c in output_components]
            ordered += ["n_exp", "gamma_parachor"]
            # Only include columns that exist
            ordered = [c for c in ordered if c in df.columns]
            return df[ordered]
 
        def _reorder_columns_wsd(df: pd.DataFrame) -> pd.DataFrame:
            """Reorder columns for WSD method: T, P, z, rho_l, rho_v, x, y, rhoL_i, rhoV_i, pure_props, gamma."""
            ordered = ["T", "P"]
            ordered += [f"z_{c}" for c in output_components]
            ordered += ["rho_l", "rho_v"]
            ordered += [f"x_{c}" for c in output_components]
            ordered += [f"y_{c}" for c in output_components]
            ordered += [f"rhoL{j+1}_{c}" for j, c in enumerate(output_components)]
            ordered += [f"rhoV{j+1}_{c}" for j, c in enumerate(output_components)]
            ordered += [c for c in pure_prop_cols if c in df.columns]
            # Add gamma columns at the end
            gamma_cols = [c for c in df.columns if c.startswith("gamma") and c not in ordered]
            ordered += gamma_cols
            # Only include columns that exist
            ordered = [c for c in ordered if c in df.columns]
            return df[ordered]
 
        for line in ["bubble", "dew"]:
            # --- Parachor method ---
            # Variables: T, P, rho_l, rho_v, x, y, pure_props, parachor, n_exp, gamma
            df_para_full = df_sat_parachor[df_sat_parachor["line"] == line].copy()
 
            if ML_mode:
                # Select parachor-relevant columns including pure properties
                active_mole_cols = [f"x_{c}" for c in component_names] + [f"y_{c}" for c in component_names]
                active_parachor_cols = [f"parachor_{c}" for c in component_names]
                active_comp_density_cols = (
                    [f"rhoL{i+1}_{c}" for i, c in enumerate(component_names)] +
                    [f"rhoV{i+1}_{c}" for i, c in enumerate(component_names)]
                )
                para_cols = ["T", "P", "rho_l", "rho_v"] + active_mole_cols + active_comp_density_cols + pure_prop_cols + active_parachor_cols + ["gamma_mix"]
                para_cols = [c for c in para_cols if c in df_para_full.columns]
                df_para = df_para_full[para_cols].copy()
                # Pad z, x, y, rhoL_i, rhoV_i, parachor with zeros for inactive components
                for c in output_components:
                    df_para[f"z_{c}"] = z_dict.get(c, 0.0)
                    if f"x_{c}" not in df_para.columns:
                        df_para[f"x_{c}"] = 0.0
                    if f"y_{c}" not in df_para.columns:
                        df_para[f"y_{c}"] = 0.0
                    if f"parachor_{c}" not in df_para.columns:
                        df_para[f"parachor_{c}"] = 0.0
                df_para = _add_padded_densities(df_para)
                # Add parachor exponent
                df_para["n_exp"] = n_exp
                df_para = df_para.rename(columns={"gamma_mix": "gamma_parachor"})
                df_para = _reorder_columns_parachor(df_para)
            else:
                df_para = df_para_full[["T", "P", "gamma_mix"]].copy()
                df_para = df_para.rename(columns={"gamma_mix": "gamma_parachor"})
 
            df_para = df_para.sort_values("T").reset_index(drop=True)
 
            para_path = os.path.join(output_dir, f"{line}_parachor.csv")
            df_para.to_csv(para_path, index=False)
            result[f"{line}_parachor"] = df_para
            if verbose:
                print(f"Saved {para_path} ({len(df_para)} rows, {len(df_para.columns)} cols)")
 
            # --- WSD method ---
            df_wsd_full = df_sat_wsd[df_sat_wsd["line"] == line].copy()
 
            if ML_mode:
                # Select base columns and active component columns that exist (no parachor for WSD)
                active_mole_cols = [f"x_{c}" for c in component_names] + [f"y_{c}" for c in component_names]
                active_comp_density_cols = (
                    [f"rhoL{i+1}_{c}" for i, c in enumerate(component_names)] +
                    [f"rhoV{i+1}_{c}" for i, c in enumerate(component_names)]
                )
                wsd_cols = base_cols + active_mole_cols + active_comp_density_cols + pure_prop_cols + ["gamma_mix", "gamma_wsd_corrected"]
                wsd_cols = [c for c in wsd_cols if c in df_wsd_full.columns]
                df_wsd = df_wsd_full[wsd_cols].copy()
                df_wsd = _add_padded_compositions(df_wsd, include_parachor=False)
                df_wsd = _add_padded_densities(df_wsd)
                df_wsd = df_wsd.rename(columns={"gamma_mix": "gamma_wsd"})
                df_wsd = _reorder_columns_wsd(df_wsd)
            else:
                df_wsd = df_wsd_full[["T", "P", "gamma_mix", "gamma_wsd_corrected"]].copy()
                df_wsd = df_wsd.rename(columns={"gamma_mix": "gamma_wsd"})
 
            df_wsd = df_wsd.sort_values("T").reset_index(drop=True)
 
            wsd_path = os.path.join(output_dir, f"{line}_wsd.csv")
            df_wsd.to_csv(wsd_path, index=False)
            result[f"{line}_wsd"] = df_wsd
            if verbose:
                print(f"Saved {wsd_path} ({len(df_wsd)} rows, {len(df_wsd.columns)} cols)")
 
        return result

    @staticmethod
    def save_PT_colormap_dataframes(
        data_parachor: dict,
        data_wsd: dict,
        output_dir: str,
        component_names: list[str],
        feed_z: np.ndarray,
        n_exp: float = 3.87,
        ML_mode: bool = True,
        main_component_index: int = 0,
        all_components: list[str] | None = None,
        data_parachor_fixed: dict | None = None,
        T_ref_fixed: float | None = None,
        data_cdft: dict | None = None,
        data_wsd_nocorr: dict | None = None,
        filename_parachor: str = "PT_parachor",
        filename_parachor_fixed: str = "PT_parachor_fixed",
        filename_wsd: str = "PT_wsd",
        verbose: bool = False,
    ) -> dict[str, pd.DataFrame]:
        """
        Save PT colormap IFT DataFrames to CSV files, one per method.

        Mirrors save_saturation_dataframes but operates on the full two-phase
        region data from compute_PT_colormap_data (no bubble/dew split).

        Saved files
        -----------
        PT_parachor.csv          : T-dependent parachor
        PT_parachor_fixed.csv    : Fixed-T parachor (only if data_parachor_fixed given)
        PT_wsd.csv               : WSD method

        Column layout (ML_mode=True)
        ----------------------------
        T, P, z_i, rho_l, rho_v, x_i, y_i, rhoL_i, rhoV_i,
        pure_props (main comp only), [parachor_i, n_exp,] gamma

        When *data_cdft* is provided, difference columns are appended as the
        last column(s):

        - PT_parachor.csv  : ``gamma_cDFT_minus_parachor``
        - PT_wsd.csv       : ``gamma_cDFT_minus_wsd_corrected``, and if
          *data_wsd_nocorr* is also given, ``gamma_cDFT_minus_wsd_uncorrected``

        The cDFT gamma values are interpolated onto each SEC method's (T, P)
        points via cubic ``griddata``.

        Parameters
        ----------
        data_parachor : dict
            Output of compute_PT_colormap_data with method='parachor' and
            recompute_parachor=True.
        data_wsd : dict
            Output of compute_PT_colormap_data with method='wsd'.
        output_dir : str
            Directory for CSV files (created if absent).
        component_names : list[str]
            Active components in this mixture.
        feed_z : np.ndarray
            Feed mole fractions, shape (N,).
        n_exp : float, optional
            Parachor exponent written as a column. Default 3.87.
        ML_mode : bool, optional
            If True (default), write all features needed for ML training.
            If False, write only T, P, and gamma column(s).
        main_component_index : int, optional
            Index of the primary component (default 0) whose pure properties
            (gamma0, rhoL0, rhoV0, Tc, Pc, Psat0) are included.
            Impurity components are often supercritical and would be NaN.
        all_components : list[str] or None, optional
            Full component universe for consistent column layout across
            different mixture CSVs.  Inactive components get zero-filled
            z, x, y, rhoL, rhoV, and parachor columns.
        data_parachor_fixed : dict or None, optional
            Output of compute_PT_colormap_data with recompute_parachor=False
            (fixed-T parachor numbers).  Saved as PT_parachor_fixed.csv.
        T_ref_fixed : float or None, optional
            Reference temperature used for fixed parachor numbers; appended
            to the CSV filename suffix in the print statement only.
        data_cdft : dict or None, optional
            cDFT scattered data dict with 'T', 'P', 'gamma' keys.  When
            provided, difference columns (cDFT - SEC) are appended to each
            output CSV via cubic interpolation.
        data_wsd_nocorr : dict or None, optional
            Output of compute_PT_colormap_data with method='wsd' and
            wsd_correction=False.  When provided together with *data_cdft*,
            a ``gamma_cDFT_minus_wsd_uncorrected`` column is appended to
            PT_wsd.csv.

        Returns
        -------
        dict[str, pd.DataFrame]
            Keys: 'PT_parachor', 'PT_wsd', and optionally 'PT_parachor_fixed'.
        """
        os.makedirs(output_dir, exist_ok=True)
        result: dict[str, pd.DataFrame] = {}
        feed_z = np.asarray(feed_z, dtype=float)

        # Build cDFT arrays for interpolation onto each SEC (T, P) grid.
        _cdft_T = _cdft_P = _cdft_G = None
        if data_cdft is not None:
            _cdft_T = np.asarray(data_cdft["T"], dtype=float)
            _cdft_P = np.asarray(data_cdft["P"], dtype=float)
            _cdft_G = np.asarray(data_cdft["gamma"], dtype=float)

        def _interp_cdft(target_T: np.ndarray, target_P: np.ndarray) -> np.ndarray:
            """Interpolate cDFT gamma onto arbitrary (T, P) points via griddata."""
            points = np.column_stack([_cdft_T, _cdft_P])
            result = griddata(points, _cdft_G, (target_T, target_P), method="cubic")
            # Fill extrapolated NaNs with nearest-neighbour fallback
            nan_mask = np.isnan(result)
            if nan_mask.any():
                result[nan_mask] = griddata(
                    points, _cdft_G, (target_T[nan_mask], target_P[nan_mask]), method="nearest"
                )
            return result

        output_components = all_components if all_components is not None else component_names
        main_comp         = component_names[main_component_index]
        z_dict            = {c: float(feed_z[i]) for i, c in enumerate(component_names)}

        pure_prop_cols = [
            f"gamma0_{main_comp}", f"rhoL0_{main_comp}", f"rhoV0_{main_comp}",
            f"Tc_{main_comp}",     f"Pc_{main_comp}",    f"Psat0_{main_comp}",
        ]

        # ------------------------------------------------------------------
        # Build a flat DataFrame from a raw compute_PT_colormap_data dict,
        # including component-specific partial densities.
        # ------------------------------------------------------------------
        def _raw_to_df(data: dict, include_parachor: bool) -> pd.DataFrame:
            """Expand the nested lists in *data* into a flat DataFrame."""
            rows: dict[str, list] = {
                "T":     data["T"],
                "P":     data["P"],
                "rho_l": data["rho_l"],
                "rho_v": data["rho_v"],
                "gamma_mix": data["gamma"],
            }
            for i, name in enumerate(component_names):
                rows[f"x_{name}"]    = [v[i] for v in data["x"]]
                rows[f"y_{name}"]    = [v[i] for v in data["y"]]
                # Component-specific partial densities: rho_i^L = x_i * rho_l
                rows[f"rhoL{i+1}_{name}"] = [
                    v[i] * rho for v, rho in zip(data["x"], data["rho_l"])
                ]
                rows[f"rhoV{i+1}_{name}"] = [
                    v[i] * rho for v, rho in zip(data["y"], data["rho_v"])
                ]
                rows[f"gamma0_{name}"] = [v[i] for v in data["gamma0"]]
                rows[f"rhoL0_{name}"] = [v[i] for v in data["rhoL0"]]
                rows[f"rhoV0_{name}"] = [v[i] for v in data["rhoV0"]]
                rows[f"Psat0_{name}"] = [v[i] for v in data["Psat0"]]
                rows[f"Tc_{name}"]    = [v[i] for v in data["Tc"]]
                rows[f"Pc_{name}"]    = [v[i] for v in data["Pc"]]
                if include_parachor:
                    rows[f"parachor_{name}"] = [
                        v[i] if v is not None else None for v in data["parachor"]
                    ]
            return pd.DataFrame(rows)

        # ------------------------------------------------------------------
        # Pad inactive components (zeros) and add feed composition column.
        # ------------------------------------------------------------------
        def _pad_and_reorder(
            df: pd.DataFrame,
            include_parachor: bool,
            gamma_col_rename: str,
        ) -> pd.DataFrame:
            n_rows = len(df)

            # Feed composition
            for c in output_components:
                df[f"z_{c}"] = z_dict.get(c, 0.0)

            # x, y  – zero for inactive
            for c in output_components:
                if f"x_{c}" not in df.columns:
                    df[f"x_{c}"] = 0.0
                if f"y_{c}" not in df.columns:
                    df[f"y_{c}"] = 0.0

            # rhoL_i, rhoV_i – re-index to output_components position; zero for inactive
            _src_rhoL = {c: f"rhoL{i+1}_{c}" for i, c in enumerate(component_names)}
            _src_rhoV = {c: f"rhoV{i+1}_{c}" for i, c in enumerate(component_names)}
            for j, c in enumerate(output_components):
                dst_L, dst_V = f"rhoL{j+1}_{c}", f"rhoV{j+1}_{c}"
                src_L, src_V = _src_rhoL.get(c), _src_rhoV.get(c)
                if src_L and src_L in df.columns:
                    df[dst_L] = df[src_L]
                    if dst_L != src_L:
                        df.drop(columns=[src_L], inplace=True)
                elif dst_L not in df.columns:
                    df[dst_L] = 0.0
                if src_V and src_V in df.columns:
                    df[dst_V] = df[src_V]
                    if dst_V != src_V:
                        df.drop(columns=[src_V], inplace=True)
                elif dst_V not in df.columns:
                    df[dst_V] = 0.0

            # parachor – zero for inactive
            if include_parachor:
                for c in output_components:
                    if f"parachor_{c}" not in df.columns:
                        df[f"parachor_{c}"] = 0.0

            # Rename gamma_mix -> method-specific name
            df = df.rename(columns={"gamma_mix": gamma_col_rename})

            # Build final column order
            ordered  = ["T", "P"]
            ordered += [f"z_{c}"         for c in output_components]
            ordered += ["rho_l", "rho_v"]
            ordered += [f"x_{c}"         for c in output_components]
            ordered += [f"y_{c}"         for c in output_components]
            ordered += [f"rhoL{j+1}_{c}" for j, c in enumerate(output_components)]
            ordered += [f"rhoV{j+1}_{c}" for j, c in enumerate(output_components)]
            ordered += [col for col in pure_prop_cols if col in df.columns]
            if include_parachor:
                ordered += [f"parachor_{c}" for c in output_components]
            ordered += [gamma_col_rename]
            ordered  = [c for c in ordered if c in df.columns]
            return df[ordered]

        # ------------------------------------------------------------------
        # Parachor (T-dependent)
        # ------------------------------------------------------------------
        df_raw_para = _raw_to_df(data_parachor, include_parachor=True)

        if ML_mode:
            df_para = _pad_and_reorder(df_raw_para, include_parachor=True,
                                       gamma_col_rename="gamma_parachor")
        else:
            df_para = df_raw_para[["T", "P", "gamma_mix"]].rename(
                columns={"gamma_mix": "gamma_parachor"})

        # Append cDFT - Parachor difference column (interpolated onto parachor grid)
        if _cdft_T is not None:
            gamma_cdft_interp = _interp_cdft(df_para["T"].values, df_para["P"].values)
            df_para["gamma_cDFT_minus_parachor"] = gamma_cdft_interp - df_para["gamma_parachor"].values

        df_para = df_para.sort_values("T").reset_index(drop=True)
        path = os.path.join(output_dir, f"{filename_parachor}.csv")
        df_para.to_csv(path, index=False)
        result["PT_parachor"] = df_para
        if verbose:
            print(f"Saved {path} ({len(df_para)} rows, {len(df_para.columns)} cols)")

        # ------------------------------------------------------------------
        # Parachor (fixed T_ref) – optional
        # ------------------------------------------------------------------
        if data_parachor_fixed is not None:
            df_raw_fixed = _raw_to_df(data_parachor_fixed, include_parachor=True)

            if ML_mode:
                df_fixed = _pad_and_reorder(df_raw_fixed, include_parachor=True,
                                            gamma_col_rename="gamma_parachor_fixed")
                df_fixed["n_exp"] = n_exp
                gamma_col = "gamma_parachor_fixed"
                if "n_exp" in df_fixed.columns and gamma_col in df_fixed.columns:
                    cols = [c for c in df_fixed.columns if c not in ("n_exp", gamma_col)]
                    df_fixed = df_fixed[cols + ["n_exp", gamma_col]]
            else:
                df_fixed = df_raw_fixed[["T", "P", "gamma_mix"]].rename(
                    columns={"gamma_mix": "gamma_parachor_fixed"})

            df_fixed = df_fixed.sort_values("T").reset_index(drop=True)
            path = os.path.join(output_dir, f"{filename_parachor_fixed}.csv")
            df_fixed.to_csv(path, index=False)
            result["PT_parachor_fixed"] = df_fixed
            suffix = f" (T_ref={T_ref_fixed} K)" if T_ref_fixed is not None else ""
            if verbose:
                print(f"Saved {path} ({len(df_fixed)} rows, {len(df_fixed.columns)} cols){suffix}")

        # ------------------------------------------------------------------
        # WSD
        # ------------------------------------------------------------------
        df_raw_wsd = _raw_to_df(data_wsd, include_parachor=False)

        if ML_mode:
            df_wsd = _pad_and_reorder(df_raw_wsd, include_parachor=False,
                                      gamma_col_rename="gamma_wsd")
        else:
            df_wsd = df_raw_wsd[["T", "P", "gamma_mix"]].rename(
                columns={"gamma_mix": "gamma_wsd"})

        df_wsd["T"] = np.round(df_wsd["T"].values, 10)
        df_wsd["P"] = np.round(df_wsd["P"].values, 10)

        # Append uncorrected WSD column (always, when data_wsd_nocorr is provided)
        if data_wsd_nocorr is not None:
            df_nc = pd.DataFrame({
                "T": np.round(np.asarray(data_wsd_nocorr["T"]), 10),
                "P": np.round(np.asarray(data_wsd_nocorr["P"]), 10),
                "gamma_wsd_UC": np.asarray(data_wsd_nocorr["gamma"]),
            })
            df_wsd = df_wsd.merge(df_nc, on=["T", "P"], how="left")

        # Append cDFT difference columns (interpolated onto WSD grid)
        if _cdft_T is not None:
            gamma_cdft_interp = _interp_cdft(df_wsd["T"].values, df_wsd["P"].values)
            df_wsd["gamma_cDFT_minus_wsd_corrected"] = gamma_cdft_interp - df_wsd["gamma_wsd"].values
            if "gamma_wsd_UC" in df_wsd.columns:
                df_wsd["gamma_cDFT_minus_wsd_uncorrected"] = gamma_cdft_interp - df_wsd["gamma_wsd_UC"].values

        df_wsd = df_wsd.sort_values("T").reset_index(drop=True)
        path = os.path.join(output_dir, f"{filename_wsd}.csv")
        df_wsd.to_csv(path, index=False)
        result["PT_wsd"] = df_wsd
        if verbose:
            print(f"Saved {path} ({len(df_wsd)} rows, {len(df_wsd.columns)} cols)")

        return result

    @staticmethod
    def _smooth_curve(T_raw, P_raw, Tc, Pc, n=1000, gap_tol=0.01):
        """
        PCHIP spline through bubble/dew data, anchored at the critical point.

        Uses arc-length parameterisation so that the near-vertical tangent
        near Tc (where dP/dT → ∞) does not cause overshoots or kinks.
        T(s) and P(s) are each fitted independently as functions of the
        normalised arc length s ∈ [0, 1].
        """
        T = np.asarray(T_raw, dtype=float)
        P = np.asarray(P_raw, dtype=float)
        idx = np.argsort(T)
        T, P = T[idx], P[idx]
        if (Tc - T[-1]) > gap_tol * Tc:
            T = np.append(T, Tc)
            P = np.append(P, Pc)
        # Arc-length parameterisation
        seg = np.hypot(np.diff(T), np.diff(P))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        s /= s[-1]                              # normalise to [0, 1]
        s_fine = np.linspace(0.0, 1.0, n)
        T_s = PchipInterpolator(s, T)(s_fine)
        P_s = PchipInterpolator(s, P)(s_fine)
        return T_s, P_s

    @staticmethod
    def plot_PT_colormap(
        data: dict,
        T_bub: list[float],
        P_bub: list[float],
        T_dew: list[float],
        P_dew: list[float],
        Tc_K: float,
        Pc_bar: float,
        params,
        feed_z: np.ndarray,
        title_suffix: str = "",
        grid_size: int = 200,
        isoline_step: int = 2,
        filename: str | None = None,
        folder: str = "CSV",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot a filled-contour PT colormap of mixture IFT inside the two-phase
        envelope, with iso-IFT lines, bubble/dew curves, and the critical point.

        The scattered (T, P, gamma) data are interpolated onto a regular grid
        using cubic ``griddata``, masked outside the phase envelope, and rendered
        with ``contourf`` (colour fill) and ``contour`` (labelled iso-lines).

        Parameters
        ----------
        data : dict
            Output from ``compute_PT_colormap_data`` containing at least
            'T', 'P', and 'gamma' lists.
        T_bub, P_bub : list[float]
            Bubble-point temperatures [K] and pressures [bar].
        T_dew, P_dew : list[float]
            Dew-point temperatures [K] and pressures [bar].
        Tc_K : float
            Mixture critical temperature [K], plotted as a marker.
        Pc_bar : float
            Mixture critical pressure [bar], plotted as a marker.
        params : feos.PcSaftParameters
            PC-SAFT parameter record, used to extract component names and
            LaTeX formulae for the title.
        feed_z : np.ndarray
            Feed mole fractions, shape (N,), shown in the title.
        title_suffix : str, optional
            Extra text appended to the auto-generated title. Default is ``""``.
        grid_size : int, optional
            Number of points along each axis of the interpolation grid.
            Default is 200.
        isoline_step : int, optional
            Spacing (in mN/m) between labelled iso-IFT contour lines.
            Default is 2.

        Returns
        -------
        fig : matplotlib.figure.Figure or None
            The figure object. ``None`` if fewer than 4 valid data points.
        ax : matplotlib.axes.Axes or None
            The axes object. ``None`` if fewer than 4 valid data points.
        """

        T_pts = data["T"]
        P_pts = data["P"]
        gamma_pts = data["gamma"]

        if len(gamma_pts) < 4:
            warnings.warn("[plot_PT_colormap] Fewer than 4 valid points — cannot plot.")
            return None, None

        T_arr = np.array(T_pts)
        P_arr = np.array(P_pts)
        g_arr = np.array(gamma_pts)

        T_grid = np.linspace(T_arr.min(), T_arr.max(), grid_size)
        P_grid = np.linspace(P_arr.min(), P_arr.max(), grid_size)
        T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

        gamma_mesh = griddata(
            (T_arr, P_arr),
            g_arr,
            (T_mesh, P_mesh),
            method="cubic",
            fill_value=np.nan,
        )

        T_bub_s, P_bub_s = semi_emperical_correlations._smooth_curve(T_bub, P_bub, Tc_K, Pc_bar)
        T_dew_s, P_dew_s = semi_emperical_correlations._smooth_curve(T_dew, P_dew, Tc_K, Pc_bar)
        env_T = list(T_bub_s) + list(reversed(list(T_dew_s)))
        env_P = list(P_bub_s) + list(reversed(list(P_dew_s)))
        path = MplPath(np.column_stack([env_T, env_P]))
        pts = np.column_stack([T_mesh.ravel(), P_mesh.ravel()])
        inside = path.contains_points(pts).reshape(T_mesh.shape)

        gamma_masked = np.ma.masked_where(~inside, gamma_mesh)

        gamma_min = float(np.nanmin(gamma_masked))
        gamma_max = float(np.nanmax(gamma_masked))

        levels_lines = [
            n for n in range(1, 60, isoline_step) if gamma_min <= n <= gamma_max
        ]

        fig, ax = ps.plot_init()

        levels_fill = np.linspace(gamma_min, gamma_max, 35)
        cf = ax.contourf(
            T_mesh,
            P_mesh,
            gamma_masked,
            levels=levels_fill,
            cmap=plt.cm.Blues,
            extend="both",
        )

        inside_eroded = binary_erosion(inside, iterations=2)
        gamma_eroded = np.ma.masked_where(~inside_eroded, gamma_mesh)

        if levels_lines:
            cl = ax.contour(
                T_mesh,
                P_mesh,
                gamma_eroded,
                levels=levels_lines,
                colors="black",
                linewidths=ps.linewidth / 2,
                alpha=0.6,
            )
            ax.clabel(
                cl,
                levels=levels_lines,
                inline=True,
                fontsize=ps.label_fontsize / 2,
                fmt="%.0f",
                inline_spacing=15,
                rightside_up=True,
            )

        cbar = fig.colorbar(cf, ax=ax, pad=0.02, format="%.0f")
        cbar.set_ticks(levels_lines)
        cbar.set_label(
            r"$\gamma \; / \; [\mathrm{mN \, m^{-1}}]$",
            fontsize=ps.label_fontsize,
        )
        cbar.ax.tick_params(labelsize=10)

        ax.plot(
            T_bub_s,
            P_bub_s,
            "b-",
            linewidth=ps.linewidth,
            label="Bubble curve",
            zorder=10,
        )
        ax.plot(
            T_dew_s,
            P_dew_s,
            "r-",
            linewidth=ps.linewidth,
            label="Dew curve",
            zorder=10,
        )

        ax.plot(
            Tc_K,
            Pc_bar,
            "o",
            markersize=4,
            zorder=15,
            markerfacecolor="grey",
            markeredgewidth=1,
            markeredgecolor="black",
            label="Critical point",
        )

        ax.set_xlabel(r"$T \; / \; [\mathrm{K}]$", fontsize=ps.label_fontsize)
        ax.set_ylabel(r"$P \; / \; [\mathrm{bar}]$", fontsize=ps.label_fontsize)
        ax.set_xlim(left=200)
        ax.minorticks_on()

        comps = components(params)
        z_str = ", ".join(
            f"{latex_formula(c)}={v:g}" for c, v in zip(comps, np.asarray(feed_z))
        )
        suffix = f"  [{title_suffix}]" if title_suffix else ""
        ax.set_title(f"{z_str}{suffix}", fontsize=ps.label_fontsize / 2)

        ps.style_legend(ax, fontsize=ps.legend_fontsize, loc="best", framealpha=0)
        plt.tight_layout()

        if filename is not None:
            png_dir = os.path.join(folder, "PNG")
            pdf_dir = os.path.join(folder, "PDF")
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)
            fig.savefig(os.path.join(png_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(pdf_dir, f"{filename}.pdf"), bbox_inches="tight")

        return fig, ax


class ColormapDifference:
    """
    Compute and plot the difference between two PT-colormap IFT datasets.

    This class takes two datasets produced by
    ``semi_emperical_correlations.compute_PT_colormap_data`` (or any dict
    with 'T', 'P', 'gamma' lists) and computes the pointwise difference
    on a common interpolation grid masked to the two-phase envelope.

    The sign convention is  ``diff = data_A - data_B``.

    Parameters
    ----------
    data_A, data_B : dict
        Each must contain keys 'T', 'P', 'gamma' (lists or arrays).
    T_bub, P_bub : array-like
        Bubble-point temperatures [K] and pressures [bar].
    T_dew, P_dew : array-like
        Dew-point temperatures [K] and pressures [bar].
    grid_size : int, optional
        Resolution of the interpolation grid per axis. Default 200.
    """

    def __init__(
        self,
        data_A: dict,
        data_B: dict,
        T_bub,
        P_bub,
        T_dew,
        P_dew,
        grid_size: int = 200,
    ):
        T_A = np.asarray(data_A["T"])
        P_A = np.asarray(data_A["P"])
        G_A = np.asarray(data_A["gamma"])

        T_B = np.asarray(data_B["T"])
        P_B = np.asarray(data_B["P"])
        G_B = np.asarray(data_B["gamma"])

        # Common grid spanning both datasets
        T_all = np.concatenate([T_A, T_B])
        P_all = np.concatenate([P_A, P_B])

        T_lin = np.linspace(T_all.min(), T_all.max(), grid_size)
        P_lin = np.linspace(P_all.min(), P_all.max(), grid_size)
        self.T_mesh, self.P_mesh = np.meshgrid(T_lin, P_lin)

        G_A_mesh = griddata((T_A, P_A), G_A, (self.T_mesh, self.P_mesh),
                            method="cubic", fill_value=np.nan)
        G_B_mesh = griddata((T_B, P_B), G_B, (self.T_mesh, self.P_mesh),
                            method="cubic", fill_value=np.nan)

        # Phase-envelope mask
        env_T = list(T_bub) + list(reversed(list(T_dew)))
        env_P = list(P_bub) + list(reversed(list(P_dew)))
        envelope_path = MplPath(np.column_stack([env_T, env_P]))
        pts = np.column_stack([self.T_mesh.ravel(), self.P_mesh.ravel()])
        self.inside = envelope_path.contains_points(pts).reshape(self.T_mesh.shape)
        self.inside_eroded = binary_erosion(self.inside, iterations=2)

        diff = G_A_mesh - G_B_mesh
        self.diff_masked = np.ma.masked_where(
            ~self.inside | np.isnan(diff), diff
        )
        self.diff_eroded = np.ma.masked_where(
            ~self.inside_eroded | np.isnan(diff), diff
        )

    def plot(
        self,
        Tc_K: float,
        Pc_bar: float,
        T_bub=None,
        P_bub=None,
        T_dew=None,
        P_dew=None,
        label: str = "",
        isoline_step: int = 1,
        filename: str | None = None,
        folder: str = "CSV",
    ) -> tuple[plt.Figure, plt.Axes] | tuple[None, None]:
        """
        Plot the difference colormap with iso-difference lines and
        the phase envelope overlay.

        Parameters
        ----------
        Tc_K : float
            Mixture critical temperature [K].
        Pc_bar : float
            Mixture critical pressure [bar].
        T_bub, P_bub : array-like, optional
            Bubble curve for overlay. If *None*, the envelope is not drawn.
        T_dew, P_dew : array-like, optional
            Dew curve for overlay. If *None*, the envelope is not drawn.
        label : str, optional
            Title text, e.g. ``"cDFT - Parachor"``.
        isoline_step : int, optional
            Spacing [mN/m] between iso-difference contour lines. Default 1.

        Returns
        -------
        fig, ax or (None, None) if insufficient data.
        """
        if self.diff_masked.count() < 4:
            warnings.warn(
                f"[ColormapDifference.plot] '{label}' — fewer than 4 "
                "valid points, skipping."
            )
            return None, None

        vmin = float(np.nanmin(self.diff_masked))
        vmax = float(np.nanmax(self.diff_masked))
        vlim = max(abs(vmin), abs(vmax))

        fig, ax = ps.plot_init()

        levels_fill = np.linspace(-vlim, vlim, 35)
        cf = ax.contourf(
            self.T_mesh, self.P_mesh, self.diff_masked,
            levels=levels_fill, cmap="RdBu_r", extend="both",
        )

        levels_lines = [
            n for n in range(-int(vlim), int(vlim) + 1, isoline_step)
            if vmin <= n <= vmax
        ]

        if levels_lines:
            cl = ax.contour(
                self.T_mesh, self.P_mesh, self.diff_eroded,
                levels=levels_lines, colors="black",
                linestyles="solid", linewidths=ps.linewidth / 2, alpha=0.6,
            )
            ax.clabel(
                cl, levels=levels_lines, inline=True,
                fontsize=ps.label_fontsize / 2, fmt="%.0f",
                inline_spacing=15, rightside_up=True,
            )

        cbar = fig.colorbar(cf, ax=ax, pad=0.02, format="%.1f")
        cbar.set_label(
            r"$\Delta\gamma \; / \; [\mathrm{mN \, m^{-1}}]$",
            fontsize=ps.label_fontsize,
        )
        cbar.ax.tick_params(labelsize=10)

        if T_bub is not None:
            T_bub_s, P_bub_s = semi_emperical_correlations._smooth_curve(
                T_bub, P_bub, Tc_K, Pc_bar)
            ax.plot(T_bub_s, P_bub_s, "b-", linewidth=ps.linewidth,
                    label="Bubble curve", zorder=10)
        if T_dew is not None:
            T_dew_s, P_dew_s = semi_emperical_correlations._smooth_curve(
                T_dew, P_dew, Tc_K, Pc_bar)
            ax.plot(T_dew_s, P_dew_s, "r-", linewidth=ps.linewidth,
                    label="Dew curve", zorder=10)

        ax.plot(Tc_K, Pc_bar, "o", markersize=4, zorder=15,
                markerfacecolor="grey", markeredgewidth=1,
                markeredgecolor="black", label="Critical point")

        ax.set_xlabel(r"$T \; / \; [\mathrm{K}]$", fontsize=ps.label_fontsize)
        ax.set_ylabel(r"$P \; / \; [\mathrm{bar}]$", fontsize=ps.label_fontsize)
        ax.set_xlim(left=200)
        ax.minorticks_on()
        ps.style_legend(ax, fontsize=ps.legend_fontsize, loc="best", framealpha=0)

        if label:
            ax.set_title(f"IFT Difference: {label}", fontsize=ps.label_fontsize)

        fig.tight_layout()

        if filename is not None:
            png_dir = os.path.join(folder, "PNG")
            pdf_dir = os.path.join(folder, "PDF")
            os.makedirs(png_dir, exist_ok=True)
            os.makedirs(pdf_dir, exist_ok=True)
            fig.savefig(os.path.join(png_dir, f"{filename}.png"), dpi=300, bbox_inches="tight")
            fig.savefig(os.path.join(pdf_dir, f"{filename}.pdf"), bbox_inches="tight")

        return fig, ax

    def deviation_summary(
        self,
        label: str = "",
        filename: str | None = None,
        folder: str = "CSV",
    ) -> dict:
        """
        Compute max/min deviation statistics from the masked difference grid
        and optionally write them to a JSON file.

        The sign convention follows the class:  ``diff = data_A - data_B``
        (typically ``cDFT - SEC``).

        * **max_deviation**  — largest positive value  (data_A over-predicts data_B)
        * **min_deviation**  — largest negative value  (data_A under-predicts data_B)

        Parameters
        ----------
        label : str, optional
            Human-readable label stored in the JSON (e.g. ``"cDFT - Parachor"``).
        filename : str, optional
            Base filename (without extension) for the JSON output.
            If *None* the data are returned but not written to disk.
        folder : str, optional
            Root output folder; the JSON is placed directly in this folder.

        Returns
        -------
        dict
            ``{label, max_deviation_mNm, max_deviation_T_K, max_deviation_P_bar,
                min_deviation_mNm, min_deviation_T_K, min_deviation_P_bar,
                mean_deviation_mNm, rmse_mNm, n_points}``
        """
        if self.diff_masked.count() == 0:
            return {}

        flat = self.diff_masked.compressed()
        idx_flat_max = int(np.argmax(flat))
        idx_flat_min = int(np.argmin(flat))

        # Map back to 2-D indices to retrieve T and P at the extrema
        valid_idx = np.argwhere(~np.ma.getmaskarray(self.diff_masked))
        row_max, col_max = valid_idx[idx_flat_max]
        row_min, col_min = valid_idx[idx_flat_min]

        summary = {
            "label": label,
            "max_deviation_mNm": float(flat[idx_flat_max]),
            "max_deviation_T_K": float(self.T_mesh[row_max, col_max]),
            "max_deviation_P_bar": float(self.P_mesh[row_max, col_max]),
            "min_deviation_mNm": float(flat[idx_flat_min]),
            "min_deviation_T_K": float(self.T_mesh[row_min, col_min]),
            "min_deviation_P_bar": float(self.P_mesh[row_min, col_min]),
            "mean_deviation_mNm": float(np.mean(flat)),
            "rmse_mNm": float(np.sqrt(np.mean(flat ** 2))),
        }

        if filename is not None:
            os.makedirs(folder, exist_ok=True)
            json_path = os.path.join(folder, f"{filename}.json")
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=4)

        return summary
