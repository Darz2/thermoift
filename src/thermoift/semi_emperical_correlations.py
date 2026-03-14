#!/usr/bin/env python

# Created by Darshan on 2025-03-15
# UPDATE: 2025-03-15 by Darshan (Added the Parachor, WSD, RK model)
# TODO :  2025-03-15 Test and rmake the examples

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
import numpy        as np
import warnings
import si_units     as si
import matplotlib.pyplot as plt
 
from scipy.interpolate  import griddata
from scipy.ndimage      import binary_erosion
from matplotlib.path    import Path as MplPath
from .FeosPlugin        import PARAMETERS, molar_density_mol_m3, components, latex_formula
from .                  import PLOT_SETTINGS as ps
            
class semi_emperical_correlations:
    """
    Mixture interfacial tension via the Parachor and Winterfeld-Scriven-Davis (WSD)
    methods, fully consistent with the feos/PC-SAFT framework.
    
    Pure-component Interfacial tension and saturation densities are computed from the PC_SAFT EoS + cDFT (planar interface).
    parameters_fn : callable
    Reference to FeosPlugin.PARAMETERS
    n_grid : int
        Number of DFT grid points for the planar-interface solve (default 1024).
    l_grid : float
        cDFT domain length [A] (default 100 A).
    
    Attributes
    ----------
    component_names : list[str] or None
        Component names set by the last call to compute_pure().
    T_K_cached : float or None
        Temperature [K] for which pure-component data are currently cached.
    gamma0 : np.ndarray or None
        Cached pure IFTs [mN/m], shape (N,).
    rhoL0 : np.ndarray or None
        Cached pure saturated-liquid densities [mol/cm3], shape (N,).
    rhoV0 : np.ndarray or None
        Cached pure saturated-vapour densities [mol/cm3], shape (N,).
    Tc : np.ndarray or None
        Cached PC-SAFT critical temperatures [K], shape (N,).
    """
    
    def __init__(
        self, 
        n_grid: int = 1024, 
        l_grid: float = 100.0):
        
        self._parameters_fn = PARAMETERS
        self._density_fn    = molar_density_mol_m3
        self._n_grid        = n_grid
        self._l_grid        = l_grid

        self.component_names: list[str] | None = None
        self.T_K_cached:      float | None     = None
        self.gamma0:          np.ndarray | None = None
        self.rhoL0:           np.ndarray | None = None
        self.rhoV0:           np.ndarray | None = None
        self.Tc:              np.ndarray | None = None
        
    # ------------------------------------------------------------------
    # Pure-component DFT (single component)
    # ------------------------------------------------------------------
    
    def _pure_component_cDFT(
        self,
        component_name: str,
        T_K: float,
    ) -> tuple[float, float, float, float]:
        """
        Run a PC-SAFT EoS + planar cDFT solve for one pure component.

        Parameters
        ----------
        component_name : str    Name accepted by FeosPlugin.PARAMETERS.
        T_K            : float  Temperature [K].

        Returns
        -------
        gamma0 : float   Pure IFT [mN/m]. NaN if SC (SuperCritical) or cDFT fails.
        rhoL0  : float   Liquid density [mol/cm3]. NaN if SC.
        rhoV0  : float   Vapour density [mol/cm3]. NaN if SC.
        Tc_K   : float   PC-SAFT critical temperature [K].
        """

        # Build single-component EOS
        params      = self._parameters_fn([component_name])
        eos         = feos.HelmholtzEnergyFunctional.pcsaft(params)

        # PC-SAFT critical temperature
        cp          = feos.State.critical_point(eos)
        Tc_K        = float(cp.temperature / si.KELVIN)

        if T_K >= Tc_K:
            return np.nan, np.nan, np.nan, Tc_K

        # Saturation densities
        try:
            vle = feos.PhaseEquilibrium.pure(eos, T_K * si.KELVIN)
            
        except Exception as exc:
            warnings.warn(
                f"From semi_emperical_correlations: cDFT (planar-Interface failed ofr the pure compoent) '{component_name}' "
                f"at T={T_K} K: {exc}")
            
            return np.nan, np.nan, np.nan, Tc_K

        rhoL0 = self._density_fn(vle.liquid.density) * 1e-6   # mol/m3 -> mol/cm3
        rhoV0 = self._density_fn(vle.vapor.density)  * 1e-6

        # Pure IFT via planar cDFT
        try:
            interface = feos.PlanarInterface.from_tanh(
                vle   =vle,
                n_grid=self._n_grid,
                l_grid=self._l_grid * si.ANGSTROM,
                critical_temperature=cp.temperature,
            )
            sol    = interface.solve()
            gamma0 = float(sol.surface_tension * 1e3 / si.NEWTON * si.METER)
        except Exception as exc:
            warnings.warn(
                f"From semi_emperical_correlations: cDFT solve failed for '{component_name}' "
                f"at T={T_K} K: {exc}"
            )
            return np.nan, rhoL0, rhoV0, Tc_K

        return gamma0, rhoL0, rhoV0, Tc_K
    
    def batch_pure_component_cDFT(
        self,
        component_names: list[str],
        T_K: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute and cache pure-component IFT and saturation densities for a list of components at a given temperature.
        Call "_pure_component_cDFT" for each component and store results in the instance attributes for reuse across flash points.

        Parameters
        ----------
        component_names     : list[str]     List of component names accepted by FeosPlugin.PARAMETERS.
        T_K                 : float         Temperature [K].
        n_grid, l_grid      : int / float   cDFT grid parameters. (Optional)
        
        Returns
        -------
        gamma0_arr : np.ndarray  Pure IFTs [mN/m], shape (N,).
        rhoL0_arr  : np.ndarray  Liquid densities [mol/cm3], shape (N,).
        rhoV0_arr  : np.ndarray  Vapour densities [mol/cm3], shape (N,).
        Tc_arr     : np.ndarray  Critical temperatures [K], shape (N,).
        """
        
        N          = len(component_names)
        gamma0_arr = np.full(N, np.nan)
        rhoL0_arr  = np.full(N, np.nan)
        rhoV0_arr  = np.full(N, np.nan)
        Tc_arr     = np.full(N, np.nan)

        for i, name in enumerate(component_names):
            gamma0_arr[i], rhoL0_arr[i], rhoV0_arr[i], Tc_arr[i] = (self._pure_component_cDFT(name, T_K))
        
        # Cache results on the instance
        self.component_names = list(component_names)
        self.T_K_cached      = T_K
        self.gamma0          = gamma0_arr
        self.rhoL0           = rhoL0_arr
        self.rhoV0           = rhoV0_arr
        self.Tc              = Tc_arr
        
        return gamma0_arr, rhoL0_arr, rhoV0_arr, Tc_arr
    
    # ---------------------------------------------------------------------------
    # Parachor method
    # ---------------------------------------------------------------------------
    
    def parachor_mixture_IFT(
        self,
        x,
        y,
        rho_l: float,
        rho_v: float,
        parachor_numbers,
        kij: float = 0.0,
        n_exp: float = 3.87,
    ) -> float:
        """
        Mixture IFT computed using the parachor method.

        Parameters
        ----------
        x, y             : array        Liquid / vapour mole fractions.
        rho_l, rho_v     : float        Mixture molar densities [mol / cm3].
        parachor_numbers : array        Pure-component Parachor numbers.
        kij              : Scalar       Binary Interaction Paramter for the Parachor model.
        n_exp            : float        Parachor exponent (default 3.87, REFPROP v10).

        Returns
        -------
        gamma_parachor : float  Mixture IFT [mN/m]. NaN if any pure component is SC or if pure data are unavailable.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        P = np.asarray(parachor_numbers, dtype=float)

        if not (len(x) == len(y) == len(P)):
            raise ValueError("x, y, and parachor_numbers must have the same length.")

        N = len(x)

        def _kij(i: int , j: int ) -> float:
            return float(kij[i][j]) if np.ndim(kij) > 0 else float(kij)

        def _Pij(i: int, j: int) -> float:
            return P[i] if i == j else (1.0 - _kij(i, j)) * 0.5 * (P[i] + P[j])

        P_l = sum(x[i] * x[j] * _Pij(i, j) for i in range(N) for j in range(N))
        P_v = sum(y[i] * y[j] * _Pij(i, j) for i in range(N) for j in range(N))

        delta = rho_l * P_l - rho_v * P_v
        
        return np.nan if delta <= 0.0 else delta ** n_exp
    
    # ---------------------------------------------------------------------------
    # Winterfield - Scriven - Davis (WSD) method
    # ---------------------------------------------------------------------------

    def wsd_mixture_IFT(
        self,
        T_K: float,
        x,
        y,
        rho_l: float,
        rho_v: float,
        phi: float = 1.0,
        correction: bool = True,
    ) -> float:
        """
        Mixture IFT computed using the Winterfeld-Scriven-Davis (WSD) method.
        DOI:10.1002/aic.690240610

        Matrix form:  gamma_mix = a^T G a,  where
            a_i    = (x_i*rho_l - y_i*rho_v) / (rhoL0_i - rhoV0_i)
            G_ii   = gamma0_i
            G_ij   = phi_ij * sqrt(gamma0_i * gamma0_j),   i != j

        Components with T > Tc_i are treated as supercritical: a_i = 0 and
        their G row/column is zeroed out, so they do not contribute.

        Requires batch_pure_component_cDFT() to have been called first so
        that self.gamma0, self.rhoL0, self.rhoV0, and self.Tc are populated.
        
        Parameters
        ----------
        T_K          : float    Temperature [K].
        x, y         : array    Liquid / vapour mole fractions, length N.
        rho_l, rho_v : float    Mixture molar densities [mol / cm3].
        phi          : float or 2-D array-like
                               Mixing parameter(s). Scalar --> same value
                               for all i neq j pairs.  2-D array --> phi[i][j].
                               Default 1.0 (original WSD).
                               
        corrections  : bool     Whether to apply corrections.

        Returns
        -------
        gamma_wsd : float  Mixture IFT [mN/m]. NaN if any pure component is SC or if pure data are unavailable.
        mixcorr   : float   Correction factor applied (1.0 if none).
        """
        if any(v is None for v in (self.gamma0, self.rhoL0, self.rhoV0, self.Tc)):
            raise RuntimeError(
                "Pure-component data not available. "
                "Call batch_pure_component_cDFT() before wsd_mixture_IFT().")
        
        x      = np.asarray(x, dtype=float)
        y      = np.asarray(y, dtype=float)
        gamma0 = self.gamma0          # [mN/m]  shape (N,)
        rhoL0  = self.rhoL0           # [mol/cm³]
        rhoV0  = self.rhoV0           # [mol/cm³]
        Tc     = self.Tc              # [K]
        N      = len(x)

        if not (len(y) == len(gamma0) == len(rhoL0) == len(rhoV0) == len(Tc) == N):
            raise ValueError("x, y and cached pure-component arrays must all have length N.")

        # --- Step 0: get Tc_i from PC-SAFT (self.Tc) so we can decide which components are active ---
        active = [T_K <= Tc_i for Tc_i in Tc]

        if not any(active):
            return 0.0, 1.0   # nothing contributes above all critical temps

        # --- Step 1: Pure-component data at T (saturated liquid/vapor) ---

        # We need: n0_i = (rhoL0_i - rhoV0_i) [mol/cm3], and gamma0_i [mN/m]
        # Both come directly from the cached batch_pure_component_cDFT() results.
        n0 = np.where(
            active,
            rhoL0 - rhoV0,
            1.0,    # dummy to avoid division by zero; a_i will be forced to 0 below
        )

        # Catch near-critical components where n0 collapses to ~0
        for i in range(N):
            if active[i] and abs(n0[i]) < 1e-30: # Machine precision 1e-30 for densities in mol/cm3
                warnings.warn(
                    f"[wsd_mixture_IFT] n0[{i}] ≈ 0 (near critical?). "
                    "Treating component as supercritical."
                )
                active[i] = False
                n0[i]     = 1.0

        # --- Step 2: Mixture coefficients a_i (from mixture and pure densities) ---
        # a_i = (x_i*rho_l - y_i*rho_v) / (rhoL0_i - rhoV0_i)

        a = (x * rho_l - y * rho_v) / n0
        for i in range(N):
            if not active[i]:
                a[i] = 0.0   # force inactive components to zero

        # --- Step 3: Build symmetric interaction matrix G ---
        # G_ii = gamma0_i;  G_ij = phi_ij * np.sqrt(gamma0_i * gamma0_j),  i != j

        def _phi(i: int, j: int) -> float:
            if hasattr(phi, "__len__"):
                return float(phi[i][j])
            return float(phi)

        G = np.zeros((N, N), dtype=float)
        for i in range(N):
            if active[i]:
                G[i, i] = gamma0[i]
            for j in range(i + 1, N):
                gij = 0.0                    # ← always initialise
                if active[i] and active[j]:
                    gij = _phi(i, j) * np.sqrt(gamma0[i] * gamma0[j])
                G[i, j] = G[j, i] = gij

        # --- Step 4: Quadratic form ---
        gamma_mix = float(a @ G @ a)   # [mN/m]

        # --- Step 5: Supercritical correction ---
        mixcorr = 1.0
        if correction and not all(active):
            sc_indices     = [i for i in range(N) if not active[i]]
            x_inactive_sum = float(np.sum(x[sc_indices]))
            mixcorr        = max(0.0, 1.0 - x_inactive_sum)
            gamma_mix     *= mixcorr

        return gamma_mix, mixcorr
    
    # ---------------------------------------------------------------------------
    # Reedlich-Kister (RK) 
    # ---------------------------------------------------------------------------
    
    def RK_mixture_IFT(
        self,
        T_K: float,
        x,
        rk_coeffs,
    ) -> float:
        """
        Mixture IFT via the Redlich-Kister (RK) empirical method.

        Formula:
            sigma_mix = sum_i  x_i * sigma_i
                    + sum_i sum_{j>i}  x_i * x_j * Q_ij(x_i, x_j)
    
        where the pairwise excess term is a power series in (x_j - x_i):
            Q_ij = sum_{k=0}^{M-1}  A_ij[k] * (x_j - x_i)^k
    
        M can be 2, 3, 4, or 5 (i.e. 2-5 fitted coefficients per pair).
        Different pairs may use a different number of terms.

        Components with T > Tc_i are treated as supercritical (sigma0_i = 0).

        Requires batch_pure_component_cDFT() to have been called first so
        that self.gamma0 and self.Tc are populated.

        Parameters
        ----------
        T_K        : float     Temperature [K].
        x          : array     Liquid mole fractions, shape (N,)
        rk_coeffs  : array,    length M  where M in {2, 3, 4, 5}
        Fitted RK coefficients [A, B, C, ...] for the mixture.
        A  -- constant term
        B  -- coefficient of (x2 - x1)
        C  -- coefficient of (x2 - x1)^2
        D  -- coefficient of (x2 - x1)^3
        ...
 
        Returns
        -------
        gamma_mix : float   Mixture IFT [mN/m].  0.0 if all components are SC.
        """
        # Guard: pure-component data must already be cached
        if self.gamma0 is None or self.Tc is None:
            raise RuntimeError(
                "Pure-component data not available. "
                "Call batch_pure_component_cDFT() before rk_mixture_IFT()."
            )
 
        x          = np.asarray(x, dtype=float)
        coeffs_arr = np.asarray(rk_coeffs, dtype=float)
        gamma0     = self.gamma0      # [mN/m], shape (N,)
        Tc         = self.Tc          # [K],    shape (N,)
        N          = len(x)
 
        if len(gamma0) != N or len(Tc) != N:
            raise ValueError(
                "x and cached pure-component arrays must all have the same length."
            )
 
        M = len(coeffs_arr)
        if M not in (2, 3, 4, 5):
            raise ValueError(
                f"rk_coeffs has {M} coefficients; must be 2, 3, 4, or 5.")
 
        # --- Step 0: get Tc_i from PC-SAFT (self.Tc) so we can decide which components are active ---
        active = [T_K <= Tc_i for Tc_i in Tc]
 
        if not any(active):
            return 0.0   # nothing contributes above all critical temps
 
        # --- Step 1: Pure-component IFTs at T (saturated interface) ---
        sigma = np.where(active, gamma0, 0.0)
    
        # --- Step 2: Linear mixing term ---
        sigma_linear = float(np.dot(x, sigma))
 
        # --- Step 3: Redlich-Kister excess term ---
        x1, x2 = x[0], x[1]
        delta  = x2 - x1                              # (x2 - x1)
        powers = delta ** np.arange(M)                # [1, delta, delta^2, ...]
        Q      = float(np.dot(coeffs_arr, powers))
 
        sigma_excess = x1 * x2 * Q
    
        # --- Step 4: Mixture IFT ---
        sigma_mix = sigma_linear + sigma_excess
    
        return float(sigma_mix)
    
    # ---------------------------------------------------------------------------
    # PT sweep - collect (T, P, gamma) over the two-phase region 
    # ---------------------------------------------------------------------------
    
    def compute_PT_colormap_data(
        self,
        component_names: list[str],
        feed_z,
        eos,
        T_bub:  list[float],
        P_bub:  list[float],
        T_dew:  list[float],
        P_dew:  list[float],
        parachor_numbers=None,
        kij_parachor: float         = 0.0,
        n_exp:        float         = 3.87,
        phi_wsd:      float         = 1.0,
        wsd_correction: bool        = True,
        rk_coeffs                   = None,
        method:         str         = "parachor",
        n_T:            int         = 30,
        n_P:            int         = 30,
    ) -> tuple[list, list, list]:
        """
        Sweep a (T, P) grid inside the two-phase envelope, run a TP flash at
        each point, and compute mixture IFT using the chosen semi-empirical method.
 
        Parameters
        ----------
        component_names  : list[str]        Component names (feos identifiers).
        feed_z           : array-like       Overall feed mole fractions, shape (N,).
        eos              : feos functional  Pre-built PC-SAFT functional.
        T_bub, P_bub     : list[float]      Bubble-curve points [K] / [bar].
        T_dew, P_dew     : list[float]      Dew-curve points   [K] / [bar].
        parachor_numbers : array-like       Pure-component Parachor numbers.
                                            Required when method="parachor".
        kij_parachor     : float            Parachor binary interaction parameter.
        n_exp            : float            Parachor exponent (default 3.87).
        phi_wsd          : float or 2-D array
                                            WSD mixing parameter (default 1.0).
        wsd_correction   : bool             Apply supercritical correction in WSD.
        method           : str              "parachor", "wsd", and "rk" .
        n_T              : int              Number of temperature grid points.
        n_P              : int              Number of pressure grid points per T.
 
        Returns
        -------
        T_pts     : list[float]   Temperature values [K]    of valid points.
        P_pts     : list[float]   Pressure values    [bar]  of valid points.
        gamma_pts : list[float]   IFT values         [mN/m] of valid points.
        """
        method = method.lower().strip()
        
        if method not in ("parachor", "wsd", "rk"):
            raise ValueError(f"method must be 'parachor', 'wsd', or 'rk'; got '{method}'.")
        
        if method == "parachor" and parachor_numbers is None:
            raise ValueError("parachor_numbers must be provided when method='parachor'.")

        if method == "rk" and rk_coeffs is None:
            raise ValueError("rk_coeffs must be provided when method='rk'.")
        
        feed_z      = np.asarray(feed_z, dtype=float)
        feed_si     = feed_z * si.MOL
 
        T_pts, P_pts, gamma_pts = [], [], []
        T_bub_arr, P_bub_arr    = zip(*sorted(zip(T_bub, P_bub)))
        T_dew_arr, P_dew_arr    = zip(*sorted(zip(T_dew, P_dew)))
        T_range                 = np.linspace(min(T_bub), max(T_bub) * 0.999, n_T)
        _cached_T               = None
        
        for T_K in T_range:
 
            # Interpolate bubble / dew pressure at this temperature
            P_bub_T = float(np.interp(T_K, T_bub_arr, P_bub_arr))
            P_dew_T = float(np.interp(T_K, T_dew_arr, P_dew_arr))
 
            if P_bub_T <= P_dew_T:
                continue
 
            P_range = np.linspace(P_dew_T * 1.01, P_bub_T * 0.99, n_P)
 
            # WSD: refresh pure-component cache once per temperature row
            if method in ("wsd", "rk") and T_K != _cached_T:
                try:
                    self.batch_pure_component_cDFT(component_names, T_K)
                    _cached_T = T_K
                except Exception as exc:
                    warnings.warn(
                        f"[compute_PT_colormap_data] WSD pure cache failed "
                        f"at T={T_K:.1f} K: {exc}"
                    )
                    continue
 
            for P_bar in P_range:
 
                T_si = T_K  *  si.KELVIN
                P_si = P_bar * si.BAR
 
                # TP flash
                try:
                    eq = feos.PhaseEquilibrium.tp_flash(eos, T_si, P_si, feed_si)
                except Exception:
                    continue
 
                liq = eq.liquid
                vap = eq.vapor
                x   = np.array(liq.molefracs, dtype=float)
                y   = np.array(vap.molefracs, dtype=float)
 
                # Molar densities [mol/cm3]  (mol/m3 × 1e-6)
                rho_l = self._density_fn(liq.density) * 1e-6
                rho_v = self._density_fn(vap.density) * 1e-6
                
                # IFT
                try:
                    if method == "parachor":
                        gamma = self.parachor_mixture_IFT(
                            x, y, rho_l, rho_v,
                            parachor_numbers=parachor_numbers,
                            kij=kij_parachor,
                            n_exp=n_exp)
                    
                    elif method == "wsd":
                        gamma, _ = self.wsd_mixture_IFT(
                            T_K, x, y, rho_l, rho_v,
                            phi=phi_wsd,
                            correction=wsd_correction)
                        
                    elif method =="rk":
                        gamma = self.RK_mixture_IFT(T_K, x, rk_coeffs)
                        
                    else:
                        raise ValueError(f"Unknown method '{method}'.")
                        
                except Exception as exc:
                    warnings.warn(
                        f"[compute_PT_colormap_data] IFT failed at "
                        f"T={T_K:.1f} K, P={P_bar:.1f} bar: {exc}")
                    continue
 
                if np.isnan(gamma) or gamma <= 0.0:
                    continue
 
                T_pts.append(T_K)
                P_pts.append(P_bar)
                gamma_pts.append(gamma)
 
        return T_pts, P_pts, gamma_pts
    
    def plot_PT_colormap(
        self,
        T_pts:     list[float],
        P_pts:     list[float],
        gamma_pts: list[float],
        T_bub:     list[float],
        P_bub:     list[float],
        T_dew:     list[float],
        P_dew:     list[float],
        Tc_K:      float,
        Pc_bar:    float,
        params,
        feed_z,
        title_suffix: str  = "",
        grid_size:    int  = 200,
        isoline_step: int  = 2,
    ):
        """
        Plot the PT colormap of mixture IFT, reproducing the style of
        FeosPlugin.plot_gamma_colormap() exactly:
        Blues contourf + black contour isolines + bubble/dew envelope +
        critical point marker.

        Parameters
        ----------
        T_pts, P_pts, gamma_pts : list[float]
            Output of compute_PT_colormap_data().
        T_bub, P_bub            : list[float]   Bubble curve [K] / [bar].
        T_dew, P_dew            : list[float]   Dew curve    [K] / [bar].
        Tc_K, Pc_bar            : float         Critical point.
        params                  : feos.Parameters
            Used only for component labels in the title.
        feed_z                  : array-like    Feed mole fractions for the title.
        title_suffix            : str           Appended to title, e.g. "Parachor".
        grid_size               : int           Interpolation grid resolution.
        isoline_step            : int           Gap between labelled isolines [mN/m].

        Returns
        -------
        fig, ax : matplotlib Figure and Axes.
        """

        if len(gamma_pts) < 4:
            warnings.warn("[plot_PT_colormap] Fewer than 4 valid points — cannot plot.")
            return None, None

        T_arr = np.array(T_pts)
        P_arr = np.array(P_pts)
        g_arr = np.array(gamma_pts)

        # --- Interpolation grid ---
        T_grid         = np.linspace(T_arr.min(), T_arr.max(), grid_size)
        P_grid         = np.linspace(P_arr.min(), P_arr.max(), grid_size)
        T_mesh, P_mesh = np.meshgrid(T_grid, P_grid)

        gamma_mesh = griddata(
            (T_arr, P_arr), g_arr,
            (T_mesh, P_mesh),
            method='cubic', fill_value=np.nan,
        )

        # --- Mask outside phase envelope ---
        env_T  = list(T_bub) + list(reversed(T_dew))
        env_P  = list(P_bub) + list(reversed(P_dew))
        path   = MplPath(np.column_stack([env_T, env_P]))
        pts    = np.column_stack([T_mesh.ravel(), P_mesh.ravel()])
        inside = path.contains_points(pts).reshape(T_mesh.shape)

        gamma_masked = np.ma.masked_where(~inside, gamma_mesh)

        gamma_min = float(np.nanmin(gamma_masked))
        gamma_max = float(np.nanmax(gamma_masked))

        # Odd-integer isolines within the data range  (1, 3, 5, …)
        levels_lines = [
            n for n in range(1, 60, isoline_step)
            if gamma_min <= n <= gamma_max
        ]

        # --- Figure ---
        fig, ax = ps.plot_init()

        # Filled contours
        levels_fill = np.linspace(gamma_min, gamma_max, 35)
        cf = ax.contourf(
            T_mesh, P_mesh, gamma_masked,
            levels=levels_fill, cmap=plt.cm.Blues, extend='both',
        )

        # Contour lines — erode mask slightly to avoid boundary artefacts
        inside_eroded = binary_erosion(inside, iterations=2)
        gamma_eroded  = np.ma.masked_where(~inside_eroded, gamma_mesh)

        if levels_lines:
            cl = ax.contour(
                T_mesh, P_mesh, gamma_eroded,
                levels=levels_lines,
                colors='black', linewidths=ps.linewidth / 2, alpha=0.6,
            )
            ax.clabel(
                cl, levels=levels_lines, inline=True,
                fontsize=ps.label_fontsize / 2, fmt='%.0f',
                inline_spacing=15, rightside_up=True,
            )

        # Colorbar
        cbar = fig.colorbar(cf, ax=ax, pad=0.02, format='%.0f')
        cbar.set_ticks(levels_lines)
        cbar.set_label(
            r'$\gamma \; / \; [\mathrm{mN \, m^{-1}}]$',
            fontsize=ps.label_fontsize,
        )
        cbar.ax.tick_params(labelsize=10)

        # Phase envelope
        ax.plot(T_bub, P_bub, 'b-', linewidth=ps.linewidth,
                label='Bubble curve', zorder=10)
        ax.plot(T_dew, P_dew, 'r-', linewidth=ps.linewidth,
                label='Dew curve', zorder=10)
        ax.plot([T_dew[-1], Tc_K], [P_dew[-1], Pc_bar],
                'r-', linewidth=ps.linewidth, zorder=10)

        # Critical point
        ax.plot(Tc_K, Pc_bar, 'o', markersize=4, zorder=15,
                markerfacecolor='grey', markeredgewidth=1,
                markeredgecolor='black', label='Critical point')

        # Axis labels
        ax.set_xlabel(r'$T \; / \; [\mathrm{K}]$', fontsize=ps.label_fontsize)
        ax.set_ylabel(r'$P \; / \; [\mathrm{bar}]$', fontsize=ps.label_fontsize)
        ax.set_xlim(left=200)
        ax.minorticks_on()

        # Title  (e.g.  "CO2=0.95, H2=0.05, Ar=0  [Parachor]")
        comps = components(params)
        z_str = ", ".join(
            f"{latex_formula(c)}={v:g}"
            for c, v in zip(comps, np.asarray(feed_z))
        )
        suffix = f"  [{title_suffix}]" if title_suffix else ""
        ax.set_title(f"{z_str}{suffix}", fontsize=ps.label_fontsize / 2)

        ps.style_legend(ax, fontsize=ps.legend_fontsize, loc='best', framealpha=0)
        plt.tight_layout()

        return fig, ax