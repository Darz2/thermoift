#!/usr/bin/env python
"""
ML Postprocessing module for ThermoIFT.

Provides utilities for machine learning model evaluation,
including feature importance, parity plots, and residual analysis.

Classes
-------
MLPostprocessing
    Main class for ML model evaluation and visualization.

Example
-------
>>> from thermoift import MLPostprocessing
>>> post = MLPostprocessing(
...     y_true=y_test,
...     y_pred=y_pred,
...     target="gamma",
...     feature_importances=model.feature_importances_,
...     feature_names=features
... )
>>> post.plot_parity(save_path="parity_plot")
>>> post.plot_residual_distribution(save_path="residual_dist")
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union, Literal
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import r2_score, mean_squared_error

from . import PLOT_SETTINGS as ps


# Color scheme for different targets
TARGET_COLORS = {
    "gamma": {
        "facecolor": "lightgreen",
        "edgecolor": "darkgreen",
        "hist_color": "green",
    },
    "p_dew": {
        "facecolor": "#FF6B6B",
        "edgecolor": "darkred",
        "hist_color": "red",
    },
    "p_bubble": {
        "facecolor": "lightblue",
        "edgecolor": "darkblue",
        "hist_color": "blue",
    },
}

# Default colors (green for gamma)
DEFAULT_COLORS = TARGET_COLORS["gamma"]


class MLPostprocessing:
    """
    Machine Learning Postprocessing utilities for model evaluation.

    This class provides methods for evaluating and visualizing ML model
    performance, including feature importance analysis, parity plots,
    and residual diagnostics. All plots use styling from PLOT_SETTINGS
    for consistent, publication-quality figures.

    Parameters
    ----------
    y_true : array-like
        True target values.
    y_pred : array-like
        Predicted target values.
    target : str, default "gamma"
        Target variable name. Used for labels and color selection.
        Supported: "gamma", "P_dew", "P_bubble".
    feature_importances : array-like, optional
        Feature importance values (e.g., from RandomForest).
    feature_names : List[str], optional
        Names of features corresponding to importances.
    label_map : dict, optional
        Mapping from feature names to LaTeX labels.

    Attributes
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted target values.
    residuals : np.ndarray
        Residuals (y_true - y_pred).
    r2 : float
        R-squared score.
    rmse : float
        Root mean squared error.
    target : str
        Target variable name.
    colors : dict
        Color scheme for plots.

    Methods
    -------
    Feature Importance:
        plot_feature_importance(scale, save_path)
            Plot feature importance (linear or log scale).

    Model Evaluation:
        plot_parity(save_path)
            Parity plot (actual vs predicted).
        plot_residual_distribution(save_path)
            Histogram of residuals.
        plot_residual_vs_predicted(save_path)
            Residuals vs predicted values.

    Metrics:
        summary()
            Get summary of model performance metrics.

    Example
    -------
    >>> from thermoift import MLPostprocessing
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Train model
    >>> model = RandomForestRegressor()
    >>> model.fit(X_train, y_train)
    >>> y_pred = model.predict(X_test)
    >>>
    >>> # Initialize postprocessing
    >>> post = MLPostprocessing(
    ...     y_true=y_test,
    ...     y_pred=y_pred,
    ...     target="gamma",
    ...     feature_importances=model.feature_importances_,
    ...     feature_names=features
    ... )
    >>>
    >>> # Generate plots
    >>> post.plot_feature_importance(scale="linear", save_path="importance_linear")
    >>> post.plot_feature_importance(scale="log", save_path="importance_log")
    >>> post.plot_parity(save_path="parity_plot")
    >>> post.plot_residual_distribution(save_path="residual_dist")
    >>> post.plot_residual_vs_predicted(save_path="residual_vs_pred")
    >>>
    >>> # Get metrics
    >>> print(post.summary())
    """

    def __init__(
        self,
        y_true: Union[np.ndarray, pd.Series, List],
        y_pred: Union[np.ndarray, pd.Series, List],
        target: str = "gamma",
        feature_importances: Optional[Union[np.ndarray, List]] = None,
        feature_names: Optional[List[str]] = None,
        label_map: Optional[dict] = None,
    ):
        """Initialize MLPostprocessing with predictions and optional feature importances."""
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self._target = target.lower()

        # Compute residuals and metrics
        self.residuals = self.y_true - self.y_pred
        self.r2 = r2_score(self.y_true, self.y_pred)
        self.rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))

        # Feature importances
        self._feature_importances = None
        self._feature_names = None
        if feature_importances is not None:
            self._feature_importances = np.asarray(feature_importances)
            self._feature_names = feature_names or [f"Feature {i}" for i in range(len(feature_importances))]

        # Label map for feature names
        self._label_map = label_map or {}

        # Set colors based on target
        self.colors = TARGET_COLORS.get(self._target, DEFAULT_COLORS)

    @property
    def target(self) -> str:
        """Get target variable name."""
        return self._target

    def _get_target_label(self) -> str:
        """Get LaTeX label for target variable."""
        target_labels = {
            "gamma": r"$\gamma$ / [mN m$^{-1}$]",
            "p_dew": r"$P_{\mathrm{dew}}$ / [bar]",
            "p_bubble": r"$P_{\mathrm{bubble}}$ / [bar]",
        }
        return target_labels.get(self._target, self._target)

    def _get_target_symbol(self) -> str:
        """Get LaTeX symbol for target variable."""
        target_symbols = {
            "gamma": r"\gamma",
            "p_dew": r"P_{\mathrm{dew}}",
            "p_bubble": r"P_{\mathrm{bubble}}",
        }
        return target_symbols.get(self._target, self._target)

    def _get_feature_label(self, name: str) -> str:
        """Get LaTeX label for feature name."""
        name_lower = name.lower()
        if name_lower in self._label_map:
            return self._label_map[name_lower]
        if name_lower in ps.symbol_map:
            return f"${ps.symbol_map[name_lower]}$"
        return name

    def plot_feature_importance(
        self,
        scale: Literal["linear", "log"] = "linear",
        color: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot feature importance as horizontal bar chart.

        Parameters
        ----------
        scale : {"linear", "log"}, default "linear"
            Scale for x-axis.
        color : str, optional
            Bar color. Defaults to "red" for linear, "steelblue" for log.
        save_path : str, optional
            Base filename to save the figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.

        Raises
        ------
        ValueError
            If feature importances were not provided.
        """
        if self._feature_importances is None:
            raise ValueError("Feature importances not provided during initialization.")

        # Sort by importance
        sorted_idx = np.argsort(self._feature_importances)
        sorted_importances = self._feature_importances[sorted_idx]
        sorted_names = [self._feature_names[i] for i in sorted_idx]

        # Get labels
        feat_labels = [self._get_feature_label(name) for name in sorted_names]

        # Calculate percentages
        imp_percent = (sorted_importances / sorted_importances.sum()) * 100

        fig, ax = ps.plot_init()

        # Set color based on scale if not provided
        if color is None:
            color = "red" if scale == "linear" else "steelblue"

        bars = ax.barh(
            feat_labels,
            sorted_importances,
            color=color,
            edgecolor="black",
            linewidth=0.8
        )

        if scale == "log":
            ax.set_xscale("log")
            ax.set_xlabel("Importance Score (log scale)", fontsize=ps.label_fontsize)
            ax.set_title("Feature Importance (Log Scale)", fontsize=ps.title_fontsize * 0.9, fontweight="bold")
            # Add percentage labels
            for bar, val, pct in zip(bars, sorted_importances, imp_percent):
                ax.text(
                    val * 1.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.2f}%",
                    va="center",
                    ha="left",
                    fontsize=ps.label_fontsize * 0.85
                )
        else:
            ax.set_xlabel("Importance Score", fontsize=ps.label_fontsize)
            ax.set_title("Feature Importance (Linear Scale)", fontsize=ps.title_fontsize * 0.9, fontweight="bold")
            ax.set_xlim(0, sorted_importances.max() * 1.2)
            # Add percentage labels
            for bar, val, pct in zip(bars, sorted_importances, imp_percent):
                ax.text(
                    val + 0.01 * sorted_importances.max(),
                    bar.get_y() + bar.get_height() / 2,
                    f"{pct:.2f}%",
                    va="center",
                    ha="left",
                    fontsize=ps.label_fontsize * 0.85
                )

        ps.apply_axis_style(ax)
        ax.tick_params(axis="y", length=0)

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path)

        plt.show()
        return fig, ax

    def plot_parity(
        self,
        model_name: str = "Model",
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot parity plot (actual vs predicted).

        Parameters
        ----------
        model_name : str, default "Model"
            Name of the model for legend.
        save_path : str, optional
            Base filename to save the figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig, ax = ps.plot_init()

        # Calculate limits
        lims = [
            min(self.y_true.min(), self.y_pred.min()),
            max(self.y_true.max(), self.y_pred.max())
        ]
        lims = [lims[0] - 0.5, lims[1] + 0.5]

        # Scatter plot
        ax.scatter(
            self.y_true, self.y_pred,
            alpha=0.8, s=20,
            facecolors=self.colors["facecolor"],
            edgecolors=self.colors["edgecolor"],
            linewidths=0.6,
            label=model_name
        )

        # Perfect prediction line
        ax.plot(lims, lims, "k--", linewidth=1.4, label="Perfect prediction")

        ax.set_xlim(lims)
        ax.set_ylim(lims)

        target_label = self._get_target_label()
        ax.set_xlabel(f"Actual {target_label}", fontsize=ps.label_fontsize * 0.9)
        ax.set_ylabel(f"Predicted {target_label}", fontsize=ps.label_fontsize * 0.9)

        ps.apply_axis_style(ax)

        # Add R2 to legend
        dummy_r2 = Line2D([], [], linestyle="none", label=rf"$R^2$ = {self.r2:.2f}")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(dummy_r2)

        ax.legend(
            handles=handles,
            fontsize=ps.label_fontsize * 0.65,
            loc="upper left",
            edgecolor="black",
            framealpha=1.0
        )

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path)

        plt.show()
        return fig, ax

    def plot_residual_distribution(
        self,
        bins: int = 40,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot histogram of residuals.

        Parameters
        ----------
        bins : int, default 40
            Number of histogram bins.
        save_path : str, optional
            Base filename to save the figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig, ax = ps.plot_init()

        ax.hist(
            self.residuals,
            bins=bins,
            alpha=0.75,
            color=self.colors["hist_color"],
            edgecolor="black",
            linewidth=0.8,
            label="Residuals"
        )

        ax.axvline(0, color="black", linewidth=1.3, linestyle="--", label="Zero residual")

        mean_resid = self.residuals.mean()
        std_resid = self.residuals.std()

        target_label = self._get_target_label()
        ax.set_xlabel(f"Residual / {target_label.split('/')[-1].strip()}", fontsize=ps.label_fontsize * 0.9)
        ax.set_ylabel(r"Count", fontsize=ps.label_fontsize * 0.9)
        ax.set_title(r"Residual Distribution", fontsize=ps.title_fontsize * 0.75, fontweight="bold")

        # Add statistics text box
        unit = target_label.split('/')[-1].strip() if '/' in target_label else ""
        textstr = (
            f"Mean = {mean_resid:.4f} {unit}\n"
            f"Std Dev = {std_resid:.4f} {unit}\n"
            f"RMSE = {self.rmse:.3f} {unit}"
        )

        ax.text(
            0.02, 0.8, textstr,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ps.apply_axis_style(ax)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            fontsize=ps.label_fontsize * 0.5,
            loc="upper right",
            edgecolor="black",
            facecolor="white",
            framealpha=1.0
        )

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path)

        plt.show()
        return fig, ax

    def plot_residual_vs_predicted(
        self,
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot residuals vs predicted values.

        Parameters
        ----------
        save_path : str, optional
            Base filename to save the figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig, ax = ps.plot_init()

        ax.scatter(
            self.y_pred, self.residuals,
            alpha=0.8, s=20,
            facecolors=self.colors["facecolor"],
            edgecolors=self.colors["edgecolor"],
            linewidths=0.6,
            label="Residuals"
        )

        ax.axhline(0, color="black", linewidth=1.3, linestyle="--", label="Zero residual")
        ax.axhline(self.rmse, color="blue", linewidth=1.1, linestyle="-.", alpha=0.85,
                   label=rf"$\pm$RMSE = {self.rmse:.3f}")
        ax.axhline(-self.rmse, color="blue", linewidth=1.1, linestyle="-.", alpha=0.85)

        target_label = self._get_target_label()
        unit = target_label.split('/')[-1].strip() if '/' in target_label else ""
        ax.set_xlabel(f"Predicted {target_label}", fontsize=ps.label_fontsize * 0.9)
        ax.set_ylabel(f"Residual / {unit}", fontsize=ps.label_fontsize * 0.9)

        ps.apply_axis_style(ax)
        ps.style_legend(ax, fontsize=ps.label_fontsize * 0.5, loc="upper left")

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path)

        plt.show()
        return fig, ax

    def summary(self) -> dict:
        """
        Get summary of model performance metrics.

        Returns
        -------
        dict
            Dictionary containing performance metrics.
        """
        return {
            "target": self._target,
            "n_samples": len(self.y_true),
            "r2": self.r2,
            "rmse": self.rmse,
            "mae": np.mean(np.abs(self.residuals)),
            "mean_residual": self.residuals.mean(),
            "std_residual": self.residuals.std(),
            "min_residual": self.residuals.min(),
            "max_residual": self.residuals.max(),
        }

    def print_summary(self) -> None:
        """Print formatted summary of model performance."""
        s = self.summary()
        print("=" * 60)
        print(f"MODEL PERFORMANCE SUMMARY - {s['target'].upper()}")
        print("=" * 60)
        print(f"Samples:        {s['n_samples']}")
        print(f"R² Score:       {s['r2']:.6f}")
        print(f"RMSE:           {s['rmse']:.6f}")
        print(f"MAE:            {s['mae']:.6f}")
        print(f"Mean Residual:  {s['mean_residual']:.6f}")
        print(f"Std Residual:   {s['std_residual']:.6f}")
        print("=" * 60)

        if self._feature_importances is not None:
            print("\nFEATURE IMPORTANCES:")
            print("-" * 60)
            sorted_idx = np.argsort(self._feature_importances)[::-1]
            for i in sorted_idx:
                name = self._feature_names[i]
                imp = self._feature_importances[i]
                pct = (imp / self._feature_importances.sum()) * 100
                bar = "█" * int(pct / 2)
                print(f"  {name:25s} | {pct:6.2f}% | {bar}")
