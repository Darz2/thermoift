#!/usr/bin/env python
"""
ML Preprocessing module specific to the SEC (cDFT Surface Energy Calculation) dataset.

Subclasses MLPreprocessing and remaps column names from the SEC CSV format
(T, P, rho_l, rho_v, gamma_wsd, ...) to publication-quality LaTeX labels.

Classes
-------
MLPreprocessingSEC
    SEC-specific preprocessing with correct column names and labels.

Example
-------
>>> from thermoift import MLPreprocessingSEC
>>> import pandas as pd
>>>
>>> df = pd.read_csv("CombinedDataset_A3_clean.csv")
>>> features = ["T", "P", "rho_l", "rho_v", "z_carbon dioxide", "z_hydrogen"]
>>> prep = MLPreprocessingSEC(df=df, features=features, target="gamma_wsd")
>>> prep.plot_target_vs_T(save_path="gamma_vs_T")
>>> prep.plot_target_vs_P(save_path="gamma_vs_P")
>>> prep.plot_target_distribution(save_path="gamma_wsd_dist")
"""

from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from .ML_preprocessing import MLPreprocessing
from . import PLOT_SETTINGS as ps


# SEC-specific label map (column name → LaTeX axis label)
_SEC_LABEL_MAP: dict = {
    # Thermodynamic state
    "t":                                    r"$T$ / [K]",
    "p":                                    r"$P$ / [bar]",
    # Bulk densities (mol/L units from PC-SAFT/cDFT)
    "rho_l":                                r"$\rho^{\mathrm{liq}}$ / [mol L$^{-1}$]",
    "rho_v":                                r"$\rho^{\mathrm{vap}}$ / [mol L$^{-1}$]",
    # Surface tension targets
    "gamma_wsd":                            r"$\gamma_{\mathrm{WSD}}$ / [mN m$^{-1}$]",
    "gamma_wsd_uc":                         r"$\gamma_{\mathrm{WSD,UC}}$ / [mN m$^{-1}$]",
    "gamma_cdft_minus_wsd_corrected":       r"$\gamma_{\mathrm{cDFT}}^{\mathrm{corr}}$ / [mN m$^{-1}$]",
    "gamma_cdft_minus_wsd_uncorrected":     r"$\gamma_{\mathrm{cDFT}}^{\mathrm{uncorr}}$ / [mN m$^{-1}$]",
    # Reference component properties
    "gamma0_carbon dioxide":                r"$\gamma^{0}_{\mathrm{CO_2}}$ / [mN m$^{-1}$]",
    "psat0_carbon dioxide":                 r"$P^{\mathrm{sat}}_{\mathrm{CO_2}}$ / [bar]",
    # Feed identifier
    "source_id":                            r"Feed ID / [-]",
}

# Merge with the parent PLOT_SETTINGS label_map (parent takes lower priority)
_FULL_LABEL_MAP: dict = {**ps.label_map, **_SEC_LABEL_MAP}


class MLPreprocessingSEC(MLPreprocessing):
    """
    SEC-dataset-specific ML preprocessing.

    Inherits all methods from MLPreprocessing but resolves axis labels
    against the SEC column names (T, P, rho_l, rho_v, gamma_wsd, ...)
    and provides SEC-specific convenience plot methods.

    Parameters
    ----------
    df : pd.DataFrame
        Combined SEC dataset (e.g. CombinedDataset_A3_clean.csv).
    features : List[str]
        Feature column names as they appear in the CSV.
    target : str
        Target column name, e.g. "gamma_wsd".

    Convenience plots (SEC-specific)
    ---------------------------------
    plot_target_vs_T(save_path)
        Target vs temperature (T) coloured by pressure (P).
    plot_target_vs_P(save_path)
        Target vs pressure (P) coloured by temperature (T).
    plot_target_vs_rho_l(save_path)
        Target vs liquid density (rho_l) coloured by T.
    plot_target_vs_rho_v(save_path)
        Target vs vapour density (rho_v) coloured by T.
    plot_target_distribution(bins, save_path)
        Histogram of the target column.
    plot_target_phase_envelope(save_path)
        Mean ± range of target along temperature axis.

    Example
    -------
    >>> from thermoift import MLPreprocessingSEC
    >>> import pandas as pd
    >>>
    >>> df = pd.read_csv("CombinedDataset_A3_clean.csv")
    >>> z_cols  = [c for c in df.columns if c.startswith("z_")]
    >>> features = ["T", "P", "rho_l", "rho_v"] + z_cols
    >>> prep = MLPreprocessingSEC(df=df, features=features, target="gamma_wsd")
    >>>
    >>> prep.plot_target_vs_T(save_path="gamma_vs_T")
    >>> prep.plot_target_vs_P(save_path="gamma_vs_P")
    >>> prep.plot_target_vs_rho_l(save_path="gamma_vs_rhoL")
    >>> prep.plot_target_vs_rho_v(save_path="gamma_vs_rhoV")
    >>> prep.plot_target_distribution(save_path="gamma_distribution")
    >>> prep.plot_target_phase_envelope(save_path="gamma_phase_envelope")
    """

    def _get_label(self, col: str) -> str:
        """Resolve a column name to a LaTeX axis label (SEC-aware)."""
        key = col.lower()
        if key in _FULL_LABEL_MAP:
            return _FULL_LABEL_MAP[key]
        return col

    # ------------------------------------------------------------------
    # SEC convenience plots
    # ------------------------------------------------------------------

    def plot_target_vs_T(
        self,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs temperature (T) coloured by pressure (P)."""
        return self.plot_scatter(
            x="T",
            y=self._target,
            color_by="P",
            xlabel=_FULL_LABEL_MAP["t"],
            ylabel=self._get_label(self._target),
            cbar_label=_FULL_LABEL_MAP["p"],
            title=rf"{self._get_label(self._target)} vs $T$",
            save_path=save_path,
        )

    def plot_target_vs_P(
        self,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs pressure (P) coloured by temperature (T)."""
        return self.plot_scatter(
            x="P",
            y=self._target,
            color_by="T",
            xlabel=_FULL_LABEL_MAP["p"],
            ylabel=self._get_label(self._target),
            cbar_label=_FULL_LABEL_MAP["t"],
            title=rf"{self._get_label(self._target)} vs $P$",
            save_path=save_path,
        )

    def plot_target_vs_rho_l(
        self,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs liquid density (rho_l) coloured by temperature (T)."""
        return self.plot_scatter(
            x="rho_l",
            y=self._target,
            color_by="T",
            xlabel=_FULL_LABEL_MAP["rho_l"],
            ylabel=self._get_label(self._target),
            cbar_label=_FULL_LABEL_MAP["t"],
            title=rf"{self._get_label(self._target)} vs $\rho^{{\mathrm{{liq}}}}$",
            save_path=save_path,
        )

    def plot_target_vs_rho_v(
        self,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs vapour density (rho_v) coloured by temperature (T)."""
        return self.plot_scatter(
            x="rho_v",
            y=self._target,
            color_by="T",
            xlabel=_FULL_LABEL_MAP["rho_v"],
            ylabel=self._get_label(self._target),
            cbar_label=_FULL_LABEL_MAP["t"],
            title=rf"{self._get_label(self._target)} vs $\rho^{{\mathrm{{vap}}}}$",
            save_path=save_path,
        )

    def plot_target_distribution(
        self,
        bins: int = 40,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Histogram of the target column with count on y-axis."""
        return self.plot_histogram(
            column=self._target,
            bins=bins,
            xlabel=self._get_label(self._target),
            title=rf"Distribution of {self._get_label(self._target)}",
            save_path=save_path,
        )

    def plot_target_phase_envelope(
        self,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Mean ± range of target along temperature (T) axis."""
        return self.plot_phase_envelope(
            group_by="T",
            value_col=self._target,
            xlabel=_FULL_LABEL_MAP["t"],
            ylabel=self._get_label(self._target),
            title=rf"{self._get_label(self._target)} along Phase Envelope",
            save_path=save_path,
        )
