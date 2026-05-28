#!/usr/bin/env python
"""
ML Preprocessing module for ThermoIFT.

Provides utilities for machine learning data preprocessing,
correlation analysis, and exploratory data visualization.

Classes
-------
MLPreprocessing
    Main class for ML preprocessing and visualization.

Example
-------
>>> from thermoift import MLPreprocessing
>>> prep = MLPreprocessing(df=data, features=["temperature", "pressure"], target="gamma")
>>> prep.plot_spearman_matrix(save_path="correlation_matrix")
>>> prep.plot_target_vs_temperature(save_path="gamma_vs_T")
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
from . import PLOT_SETTINGS as ps


class MLPreprocessing:
    """
    Machine Learning Preprocessing utilities for thermodynamic data.

    This class provides methods for analyzing and visualizing thermodynamic
    datasets, including correlation analysis and exploratory plots. All plots
    use styling from PLOT_SETTINGS for consistent, publication-quality figures.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data.
    features : List[str]
        List of feature column names.
    target : str
        Target column name (e.g., "gamma" for interfacial tension).

    Attributes
    ----------
    df : pd.DataFrame
        The input data.
    features : List[str]
        Feature column names.
    target : str
        Target column name.

    Methods
    -------
    Correlation Analysis:
        plot_spearman_matrix(label_map, save_path)
            Plot Spearman correlation matrix heatmap.
        feature_target_correlations()
            Compute feature-target correlations sorted by importance.
        drop_low_correlation_features(threshold, inplace)
            Identify/remove features with low correlation to target.

    Generic Plotting:
        plot_scatter(x, y, color_by, ...)
            Scatter plot with color mapping.
        plot_histogram(column, bins, color, ...)
            Histogram of a column.
        plot_phase_envelope(group_by, value_col, ...)
            Mean with min-max range along a grouping variable.

    Convenience Plots (for target visualization):
        plot_target_vs_temperature(save_path)
            Target vs temperature colored by pressure.
        plot_target_vs_pressure(save_path)
            Target vs pressure colored by temperature.
        plot_target_distribution(bins, save_path)
            Histogram of target distribution.
        plot_target_vs_liquid_density(save_path)
            Target vs liquid density colored by temperature.
        plot_target_vs_vapor_density(save_path)
            Target vs vapor density colored by temperature.
        plot_target_vs_interfacial_thickness(save_path)
            Target vs interfacial thickness colored by temperature.
        plot_target_phase_envelope(save_path)
            Target along phase envelope (mean +/- range per temperature).

    Example
    -------
    >>> from thermoift import MLPreprocessing
    >>>
    >>> # Define features and target
    >>> features = ["temperature", "pressure", "z_carbon dioxide", "z_hydrogen"]
    >>> target = "gamma"
    >>>
    >>> # Initialize
    >>> prep = MLPreprocessing(df=combined_df, features=features, target=target)
    >>>
    >>> # Correlation analysis
    >>> prep.plot_spearman_matrix(save_path="spearman_matrix")
    >>> corr_df = prep.feature_target_correlations()
    >>> low_corr = prep.drop_low_correlation_features(threshold=0.1)
    >>>
    >>> # Exploratory plots
    >>> prep.plot_target_vs_temperature(save_path="gamma_vs_T")
    >>> prep.plot_target_vs_pressure(save_path="gamma_vs_P")
    >>> prep.plot_target_distribution(save_path="gamma_distribution")
    >>> prep.plot_target_vs_liquid_density(save_path="gamma_vs_rhoL")
    >>> prep.plot_target_vs_vapor_density(save_path="gamma_vs_rhoV")
    >>> prep.plot_target_vs_interfacial_thickness(save_path="gamma_vs_thickness")
    >>> prep.plot_target_phase_envelope(save_path="gamma_phase_envelope")
    """

    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        target: str,
    ):
        """
        Initialize MLPreprocessing with data.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        features : List[str]
            List of feature column names.
        target : str
            Target column name.
        """
        self.df = df.copy()
        self._features = features
        self._target = target

        # Validate columns exist
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise KeyError(f"Features not found in data: {missing}")
        if target not in df.columns:
            raise KeyError(f"Target '{target}' not found in data columns.")

    @property
    def features(self) -> List[str]:
        """Get current feature columns."""
        return self._features

    @property
    def target(self) -> str:
        """Get current target column."""
        return self._target

    def plot_spearman_matrix(
        self,
        label_map: Optional[dict] = None,
        extra_cols: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot Spearman correlation matrix heatmap.

        Parameters
        ----------
        label_map : dict, optional
            Mapping from column names to LaTeX labels.
            If None, uses symbol_map from PLOT_SETTINGS.
        extra_cols : list of str, optional
            Additional target/output columns to include in the matrix,
            appended after features in the given order. If provided,
            self._target is not automatically included.
        save_path : str, optional
            Base filename to save the figure (without extension).

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        # Build column list
        if extra_cols is not None:
            numeric_cols = self._features + [c for c in extra_cols if c in self.df.columns]
        else:
            numeric_cols = self._features + [self._target]

        # Filter columns with more than one unique value
        numeric_cols = [col for col in numeric_cols if self.df[col].nunique() > 1]

        # Compute Spearman correlation
        corr = self.df[numeric_cols].corr(method="spearman")

        # Apply label mapping (use symbol_map from PLOT_SETTINGS by default)
        if label_map is None:
            raw_map = {k: f"${v}$" for k, v in ps.symbol_map.items()}
            label_map = {}
            for col in numeric_cols:
                if col in raw_map:
                    label_map[col] = raw_map[col]
                elif col.lower() in raw_map:
                    label_map[col] = raw_map[col.lower()]
        corr = corr.rename(index=label_map, columns=label_map)

        # Create upper triangle mask
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        # Scale figure to fit all squares + annotations
        n = len(numeric_cols)
        fig_size = max(4.0, n * 0.85)
        fig, ax = ps.plot_init(w=fig_size, h=fig_size)

        # Plot heatmap with seaborn
        cm = sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=ps.map,
            center=0,
            vmin=-1,
            vmax=1,
            mask=mask,
            square=True,
            linewidths=0.5,
            annot_kws={"size": max(6, int(fig_size * 0.75))},
            cbar_kws={"shrink": 1.0, "extend": "both"}
        )

        # Style colorbar
        cbar = cm.collections[0].colorbar
        ps.style_colorbar(cbar)
        cbar.set_label(
            r"Spearman $\rho_{\mathrm{s}}$ / $[-]$",
            fontsize=ps.tick_labelsize,
        )
        cbar.outline.set_linewidth(0.5)

        # Apply axis style
        ps.apply_axis_style(ax)
        ax.tick_params(
            axis="x", which="both", direction="out",
            length=ps.tick_length, width=ps.tick_width,
            bottom=True, top=False, labelbottom=True, labeltop=False,
        )
        ax.tick_params(
            axis="y", which="both", direction="out",
            length=ps.tick_length, width=ps.tick_width,
            left=True, right=False, labelleft=True, labelright=False,
        )

        # Rotate labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        plt.tight_layout()

        # Save if path provided
        if save_path:
            ps.save_plot(fig, save_path)

        plt.show()
        return fig, ax

    def feature_target_correlations(self) -> pd.DataFrame:
        """
        Compute Spearman correlations between features and target.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, correlation, p_value, abs_correlation
            Sorted by absolute correlation (descending).
        """
        results = []
        target_vals = self.df[self._target]

        for feat in self._features:
            feat_vals = self.df[feat]
            valid_mask = ~(feat_vals.isna() | target_vals.isna())

            if valid_mask.sum() < 3:
                corr, p_val = np.nan, np.nan
            else:
                corr, p_val = stats.spearmanr(
                    feat_vals[valid_mask],
                    target_vals[valid_mask]
                )

            results.append({
                "feature": feat,
                "correlation": corr,
                "p_value": p_val,
                "abs_correlation": abs(corr) if not np.isnan(corr) else 0.0,
            })

        result_df = pd.DataFrame(results)
        result_df = result_df.sort_values("abs_correlation", ascending=False)
        result_df = result_df.reset_index(drop=True)

        return result_df

    def drop_low_correlation_features(
        self,
        threshold: float = 0.1,
        inplace: bool = False,
    ) -> List[str]:
        """
        Identify features with low correlation to target.

        Parameters
        ----------
        threshold : float, default 0.1
            Minimum absolute correlation threshold.
        inplace : bool, default False
            If True, update self._features to exclude low-correlation features.

        Returns
        -------
        List[str]
            List of features to drop.
        """
        corr_df = self.feature_target_correlations()
        low_corr = corr_df[corr_df["abs_correlation"] < threshold]["feature"].tolist()

        if inplace:
            self._features = [f for f in self._features if f not in low_corr]

        return low_corr

    def _get_label(self, col: str) -> str:
        """Get LaTeX label for a column from PLOT_SETTINGS."""
        col_lower = col.lower()
        if col_lower in ps.label_map:
            return ps.label_map[col_lower]
        return col

    def plot_scatter(
        self,
        x: str,
        y: str,
        color_by: str,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        cbar_label: Optional[str] = None,
        title: Optional[str] = None,
        hline: Optional[float] = None,
        hline_kwargs: Optional[dict] = None,
        cmap: Optional[str] = None,
        alpha: float = 0.7,
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot scatter plot with color mapping.

        Parameters
        ----------
        x : str
            Column name for x-axis.
        y : str
            Column name for y-axis.
        color_by : str
            Column name for color mapping.
        xlabel, ylabel, cbar_label, title : str, optional
            Custom labels. If None, uses PLOT_SETTINGS labels.
        hline : float, optional
            y-value for a horizontal reference line (e.g. ``0.0``).
        hline_kwargs : dict, optional
            Extra keyword arguments forwarded to ``ax.axhline``.
            Defaults to ``{"color": "black", "linestyle": ":", "linewidth": 1.0}``.
        cmap : str, optional
            Colormap for the colour mapping. Defaults to ``PLOT_SETTINGS.map``
            (``"coolwarm"``) when not given.
        alpha : float, default 0.7
            Marker opacity. Use a higher value (e.g. ``0.95``) for fully
            saturated colours that match filled surface plots.
        save_path : str, optional
            Base filename to save the figure.
        folder : str, default "PLOTS"
            Output folder passed to ``ps.save_plot``.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig, ax = ps.plot_init()

        sc = ax.scatter(
            self.df[x], self.df[y],
            c=self.df[color_by],
            cmap=cmap or ps.map,
            s=15,
            alpha=alpha
        )

        cbar = plt.colorbar(sc, ax=ax, extend="both")
        cbar.set_label(cbar_label or self._get_label(color_by), fontsize=ps.label_fontsize)
        ps.style_colorbar(cbar)
        cbar.outline.set_linewidth(0.75)

        ax.set_xlabel(xlabel or self._get_label(x), fontsize=ps.label_fontsize)
        ax.set_ylabel(ylabel or self._get_label(y), fontsize=ps.label_fontsize)

        if title:
            ax.set_title(title, fontsize=ps.title_fontsize, fontweight="bold")

        if hline is not None:
            kw = {"color": "black", "linestyle": "--", "linewidth": 1.0}
            if hline_kwargs:
                kw.update(hline_kwargs)
            ax.axhline(hline, **kw)

        ps.apply_axis_style(ax)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path, folder=folder)

        plt.show()
        return fig, ax

    def plot_histogram(
        self,
        column: str,
        bins: int = 40,
        color: str = "red",
        xlabel: Optional[str] = None,
        ylabel: str = r"$\mathrm{Count} \, / \, [-]$",
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot histogram of a column.

        Parameters
        ----------
        column : str
            Column name to plot.
        bins : int, default 40
            Number of bins.
        color : str, default "red"
            Bar color.
        xlabel, ylabel, title : str, optional
            Custom labels.
        save_path : str, optional
            Base filename to save the figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig, ax = ps.plot_init()

        ax.hist(
            self.df[column],
            bins=bins,
            edgecolor="black",
            color=color,
            alpha=0.85
        )

        ax.set_xlabel(xlabel or self._get_label(column), fontsize=ps.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=ps.label_fontsize)

        if title:
            ax.set_title(title, fontsize=ps.title_fontsize, fontweight="bold")

        ps.apply_axis_style(ax)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path)

        plt.show()
        return fig, ax

    def plot_phase_envelope(
        self,
        group_by: str = "temperature",
        value_col: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        title: Optional[str] = None,
        color: str = "steelblue",
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot mean with min-max range along a grouping variable.

        Parameters
        ----------
        group_by : str, default "temperature"
            Column to group by.
        value_col : str, optional
            Column for values. Defaults to target.
        xlabel, ylabel, title : str, optional
            Custom labels.
        color : str, default "steelblue"
            Line and fill color.
        save_path : str, optional
            Base filename to save the figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        if value_col is None:
            value_col = self._target

        fig, ax = ps.plot_init()

        groups = self.df.groupby(group_by)[value_col]
        x_vals = sorted(self.df[group_by].unique())
        means = [groups.get_group(x).mean() for x in x_vals]
        mins = [groups.get_group(x).min() for x in x_vals]
        maxs = [groups.get_group(x).max() for x in x_vals]

        ax.plot(x_vals, means, color=color, linewidth=2, label=r"$\gamma_{\mathrm{mean}}$")
        ax.fill_between(x_vals, mins, maxs, alpha=0.2, color=color, label=r"$\gamma_{\mathrm{range}}$")

        ax.set_xlabel(xlabel or self._get_label(group_by), fontsize=ps.label_fontsize)
        ax.set_ylabel(ylabel or self._get_label(value_col), fontsize=ps.label_fontsize)

        if title:
            ax.set_title(title, fontsize=ps.title_fontsize, fontweight="bold")

        ps.apply_axis_style(ax)
        leg = ps.style_legend(ax, fontsize=max(getattr(ps, "legend_fontsize", 10), 10))
        if leg:
            leg.get_frame().set_visible(False)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path)

        plt.show()
        return fig, ax

    def plot_target_vs_temperature(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs temperature colored by pressure."""
        return self.plot_scatter(
            x="temperature",
            y=self._target,
            color_by="pressure",
            title=title,
            save_path=save_path,
        )

    def plot_target_vs_pressure(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs pressure colored by temperature."""
        return self.plot_scatter(
            x="pressure",
            y=self._target,
            color_by="temperature",
            title=title,
            save_path=save_path,
        )

    def plot_target_distribution(
        self,
        bins: int = 40,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot histogram of target distribution."""
        return self.plot_histogram(
            column=self._target,
            bins=bins,
            title=title,
            save_path=save_path,
        )

    def plot_target_vs_liquid_density(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs liquid density colored by temperature."""
        return self.plot_scatter(
            x="liquid_density",
            y=self._target,
            color_by="temperature",
            title=title,
            save_path=save_path,
        )

    def plot_target_vs_vapor_density(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs vapor density colored by temperature."""
        return self.plot_scatter(
            x="vapor_density",
            y=self._target,
            color_by="temperature",
            title=title,
            save_path=save_path,
        )

    def plot_target_vs_interfacial_thickness(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target vs interfacial thickness colored by temperature."""
        return self.plot_scatter(
            x="interfacial_thickness",
            y=self._target,
            color_by="temperature",
            title=title,
            save_path=save_path,
        )

    def plot_target_phase_envelope(
        self,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot target along phase envelope (mean +/- range per temperature)."""
        return self.plot_phase_envelope(
            group_by="temperature",
            value_col=self._target,
            title=title,
            save_path=save_path,
        )
