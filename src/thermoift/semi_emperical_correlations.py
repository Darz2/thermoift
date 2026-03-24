#!/usr/bin/env python

# Created by Darshan on 2025-03-15
# UPDATE: 2025-03-15 by Darshan (Added the Parachor, WSD, RK model)
# UPDATE: 2025-03-17 by Darshan (Added the PT colormap plotting function, saturation pressure functionality)
# UPDATE: 2025-03-18 by Darshan (Added the critical pressure functionality, added more error handling and warnings)
# UPDATE: 2025-03-23 by Darshan (Test and made the examples to compute the interfacial tension of the semi emperical models)
# UPDATE: 2025-03-23 by Darshan (Add the PT colormap plotting and compute the with semi-emperical correlations).
# UPDATE: 2026-03-24 by Darshan (Added compute_saturation_line_IFT for IFT along bubble/dew curves only)
# TODO: Add the functinality of compute_saturation_line_IFT in the FEOS Plugin to compare
# TODO: Add the IFT from the Pxy diagram for the binary mixtures (Validation)
# TODO: Add the metastable limits from the CNT from the old parachorpy package

"""
semi-emperical_correlations.py
=====================
Mixture interfacial tension via the Parachor and Winterfeld-Scriven-Davis (WSD)
methods, fully consistent with the feos/PC-SAFT framework.

Pure-component Interfacial tension and saturation densities are
obtained from PC-SAFT EoS + cDFT (planar interface).

Modules
-----
- PURE_COMPONENTS: Interfacial tension and saturation densities for pure components.
"""

import feos
import numpy as np
import pandas as pd
import warnings
import si_units as si
import matplotlib.pyplot as plt

from scipy.interpolate import griddata
from scipy.ndimage import binary_erosion
from matplotlib.path import Path as MplPath
from .FeosPlugin import PARAMETERS, molar_density_mol_m3, components, latex_formula
from . import PLOT_SETTINGS as ps


class semi_emperical_correlations:
    """
    Mixture interfacial tension via the Parachor and Winterfeld-Scriven-Davis (WSD)
    methods, fully consistent with the feos/PC-SAFT framework.

    Pure-component Interfacial tension and saturation densities are computed from the
    PC-SAFT EoS + cDFT (planar interface).

    This class provides three semi-empirical methods for computing mixture IFT:
    - **Parachor**: Uses parachor numbers and phase densities
    - **WSD**: Winterfeld-Scriven-Davis method using pure-component IFTs
    - **RK**: Redlich-Kister polynomial expansion

    Parameters
    ----------
    n_grid : int, optional
        Number of DFT grid points for the planar-interface solve. Default is 1024.
    l_grid : float, optional
        cDFT domain length in Angstroms. Default is 100.0 A.

    Attributes
    ----------
    component_names : list[str] or None
        Component names set by the last call to batch_pure_component_cDFT().
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

    Examples
    --------
    >>> sec = semi_emperical_correlations(n_grid=1024, l_grid=100.0)
    >>> sec.batch_pure_component_cDFT(["methane", "ethane"], T_K=300.0)
    >>> parachors = sec.batch_parachor_numbers(["methane", "ethane"], T_K=300.0)
    """

    def __init__(
        self,
        n_grid: int     = 1024,
        l_grid: float   = 100.0,
    ) -> None:

        self._parameters_fn = PARAMETERS
        self._density_fn    = molar_density_mol_m3
        self._n_grid        = n_grid
        self._l_grid            = l_grid

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
    ) -> tuple[float, float, float, float, float, float]:
        """
        Run a PC-SAFT EoS + planar cDFT solve for one pure component.

        Parameters
        ----------
        component_name : str
            Name accepted by FeosPlugin.PARAMETERS.
        T_K : float
            Temperature [K].

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

        params  = self._parameters_fn([component_name])
        eos     = feos.HelmholtzEnergyFunctional.pcsaft(params)

        cp      = feos.State.critical_point(eos)
        Tc_K    = float(cp.temperature / si.KELVIN)
        Pc_bar  = float(cp.pressure() / si.BAR)

        if T_K >= Tc_K:
            return np.nan, np.nan, np.nan, Tc_K, Pc_bar, np.nan

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

        try:
            interface   = feos.PlanarInterface.from_tanh(
                vle     =vle,
                n_grid  =self._n_grid,
                l_grid  =self._l_grid * si.ANGSTROM,
                critical_temperature=cp.temperature)
            
            sol     = interface.solve()
            gamma0  = float(sol.surface_tension * 1e3 / si.NEWTON * si.METER)

        except Exception as exc:
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
                self._pure_component_cDFT(name, T_K)
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
                Psat_arr[i] = float(vle.liquid.pressure() / si.BAR)
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
        gamma0, rhoL0, rhoV0, Tc0, Pc0, Psat0 = self.batch_pure_component_cDFT(
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
        T_range = np.linspace(min(T_bub), max(T_bub) * 0.999, n_T)
        _cached_T = None
        _current_parachor = parachor_numbers  # start with user-supplied or None

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
                    eq = feos.PhaseEquilibrium.tp_flash(eos, T_si, P_si, feed_si)
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
        wsd_correction: bool = True,
        rk_coeffs: np.ndarray | None = None,
        method: str = "parachor",
        recompute_parachor: bool = True,
        line: str = "both",
        n_points: int | None = None,
        P_offset_frac: float = 0.005,
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
        wsd_correction : bool, optional
            Whether to apply correction in the WSD method. Default is True.
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

        Returns
        -------
        dict
            Dictionary with keys:
            - 'bubble': dict with T, P, gamma, rho_l, rho_v, x, y, etc. for bubble line
            - 'dew': dict with T, P, gamma, rho_l, rho_v, x, y, etc. for dew line
            Empty dict for a line if line parameter excludes it.
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
                    "T", "P", "gamma", "rho_l", "rho_v", "x", "y",
                    "gamma0", "rhoL0", "rhoV0", "Psat0", "Tc", "Pc", "parachor",
                )
            }

        def _compute_line(T_arr: list[float], P_arr: list[float], is_bubble: bool) -> dict:
            """Compute IFT along a single saturation line."""
            data = _empty_data()
            _cached_T = None
            _current_parachor = parachor_numbers
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

                    _cached_T = T_K

                # TP flash at saturation point
                try:
                    eq = feos.PhaseEquilibrium.tp_flash(eos, T_si, P_si, feed_si)
                except Exception:
                    continue

                liq = eq.liquid
                vap = eq.vapor
                x = np.array(liq.molefracs, dtype=float)
                y = np.array(vap.molefracs, dtype=float)

                rho_l = self._density_fn(liq.density) * 1e-6
                rho_v = self._density_fn(vap.density) * 1e-6

                # Compute IFT using selected method
                try:
                    if method == "parachor":
                        gamma = self.parachor_mixture_IFT(
                            x, y, rho_l, rho_v,
                            parachor_numbers=_current_parachor,
                            kij=kij_parachor,
                            n_exp=n_exp,
                        )
                    elif method == "wsd":
                        gamma, _ = self.wsd_mixture_IFT(
                            T_K, x, y, rho_l, rho_v,
                            phi=phi_wsd,
                            correction=wsd_correction,
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
                "rho_l": line_data["rho_l"],
                "rho_v": line_data["rho_v"],
                "line": [line_name] * len(line_data["T"]),
            }

            for i, name in enumerate(component_names):
                rows[f"x_{name}"] = [v[i] for v in line_data["x"]]
                rows[f"y_{name}"] = [v[i] for v in line_data["y"]]
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
        Convert the dict returned by compute_PT_colormap_data into a flat
        pandas DataFrame.
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
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the PT colormap of mixture IFT.
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

        env_T = list(T_bub) + list(reversed(T_dew))
        env_P = list(P_bub) + list(reversed(P_dew))
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
            T_bub,
            P_bub,
            "b-",
            linewidth=ps.linewidth,
            label="Bubble curve",
            zorder=10,
        )
        ax.plot(
            T_dew,
            P_dew,
            "r-",
            linewidth=ps.linewidth,
            label="Dew curve",
            zorder=10,
        )
        ax.plot(
            [T_dew[-1], Tc_K],
            [P_dew[-1], Pc_bar],
            "r-",
            linewidth=ps.linewidth,
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

        return fig, ax