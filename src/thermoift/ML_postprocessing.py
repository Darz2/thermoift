#!/usr/bin/env python
"""
ML Postprocessing module for ThermoIFT.

Provides standardised, publication-quality evaluation and visualisation
utilities for regression models trained on thermodynamic mixture data.
All plots use :mod:`thermoift.PLOT_SETTINGS` for consistent styling.

Classes
-------
MLPostprocessing
    Central class for model evaluation: feature importance, parity plots,
    residual diagnostics, GPR-specific uncertainty plots, 1-D response
    curves, and 2-D T-P response surface heatmaps.

Functions
---------
print_model_metrics
    Compute and print train / test / validation R², RMSE, and MAE for
    any scikit-learn-compatible regression model.

Supported targets
-----------------
``"gamma"``
    Interfacial tension gamma [mN m⁻¹].
``"interfacial_thickness"``
    Interfacial thickness L^{90}_{10} [nm].
``"delta_gamma"``
    Residual interfacial tension Δγ = γ_cDFT − γ_base [mN m⁻¹].
``"p_dew``
    Dew-point pressure P_dew [bar].
``"p_bubble"``
    Bubble-point pressure P_bubble [bar].

Example
-------
>>> from thermoift import MLPostprocessing
>>> post = MLPostprocessing(
...     y_true=y_test,
...     y_pred=y_pred,
...     target="delta_gamma",
...     feature_importances=importances,
...     feature_names=features,
...     datasets={
...         "train": (y_train, y_train_pred),
...         "test":  (y_test,  y_test_pred),
...     },
... )
>>> post.plot_parity(save_path="parity_plot", folder="OUTPUTS")
>>> post.plot_residual_distribution(save_path="residual_dist", folder="OUTPUTS")
>>> post.plot_2d_response_surface(
...     model=fitted_model, X_train=X_train,
...     feature_names=features, save_path="surface_T_P", folder="OUTPUTS",
... )
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union, Literal
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

from . import PLOT_SETTINGS as ps

# Color scheme for different targets
TARGET_COLORS = {
    "gamma": {
        "facecolor": "lightgreen",
        "edgecolor": "darkgreen",
        "hist_color": "green",
    },
    "delta_gamma": {
        "facecolor": "lightgreen",
        "edgecolor": "darkgreen",
        "hist_color": "green",
    },
    "interfacial_thickness": {
        "facecolor": "#FDB863",
        "edgecolor": "#B35806",
        "hist_color": "#E66101",
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

# Colors for train/test/validation splits in parity plots
SPLIT_COLORS = {
    "train": {"facecolor": "lightgray", "edgecolor": "gray"},
    "test": {"facecolor": "lightblue", "edgecolor": "darkblue"},
    "val": {"facecolor": "#FF6B6B", "edgecolor": "darkred"},
}

# Default colors (green for gamma)
DEFAULT_COLORS = TARGET_COLORS["gamma"]


class MLPostprocessing:
    """
    Publication-quality postprocessing for regression models on thermodynamic data.

    Wraps a trained model's predictions with a suite of evaluation plots and
    metrics.  All figures use :mod:`thermoift.PLOT_SETTINGS` for consistent
    styling and can be saved to PDF/PNG via ``save_path`` + ``folder``.

    Parameters
    ----------
    y_true : array-like, shape (n_samples,)
        Ground-truth target values (test set or whichever split is primary).
    y_pred : array-like, shape (n_samples,)
        Model predictions corresponding to ``y_true``.
    target : str, default ``"gamma"``
        Target variable key.  Controls axis labels, colour scheme, and LaTeX
        symbols.  Supported values:

        * ``"gamma"``       — interfacial tension γ [mN m⁻¹]
        * ``"interfacial_thickness"`` — interfacial thickness L^{90}_{10} [nm]
        * ``"delta_gamma"`` — residual Δγ = γ_cDFT − γ_base [mN m⁻¹]
        * ``"p_dew"``       — dew-point pressure [bar]
        * ``"p_bubble"``    — bubble-point pressure [bar]
    feature_importances : array-like, shape (n_features,), optional
        Feature importance scores (e.g. normalised inverse ARD length-scales
        for GPR, ``feature_importances_`` for tree-based models).
    feature_names : List[str], optional
        Feature names in the same order as ``feature_importances``.
        Required when ``feature_importances`` is provided.
    label_map : dict, optional
        Custom ``{feature_name: latex_label}`` mapping that overrides the
        default look-up in :data:`thermoift.PLOT_SETTINGS.symbol_map`.
    datasets : dict, optional
        Train / test / validation splits for multi-split parity plots::

            {
                "train": (y_train_true, y_train_pred),
                "test":  (y_test_true,  y_test_pred),
                "val":   (y_val_true,   y_val_pred),   # optional
            }

        When provided, :meth:`plot_parity` colours each split separately and
        shows per-split R² in the legend.

    Attributes
    ----------
    y_true : np.ndarray
        Ground-truth target values (primary split).
    y_pred : np.ndarray
        Predicted target values (primary split).
    residuals : np.ndarray
        Signed residuals ``y_true − y_pred``.
    r2 : float
        R² score on the primary split.
    rmse : float
        Root-mean-squared error on the primary split.
    target : str
        Target variable key (lower-cased).
    colors : dict
        Colour scheme dict ``{facecolor, edgecolor, hist_color}`` for the
        active target.

    Methods
    -------
    Feature importance
        .. autosummary::

           plot_feature_importance

    General evaluation
        .. autosummary::

           plot_parity
           plot_residual_distribution
           plot_residual_vs_predicted

    GPR / probabilistic models (require predictive std σ)
        .. autosummary::

           plot_std_histogram
           plot_parity_colored_by_std
           plot_error_vs_std
           plot_response_curves
           plot_2d_response_surface
           plot_reconstructed_parity
           plot_parity_with_residual_distribution

    Metrics
        .. autosummary::

           summary
           print_summary

    Example
    -------
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from thermoift import MLPostprocessing
    >>>
    >>> gpr = GaussianProcessRegressor().fit(X_train, y_train)
    >>> y_pred, y_std = gpr.predict(X_test, return_std=True)
    >>>
    >>> post = MLPostprocessing(
    ...     y_true=y_test,
    ...     y_pred=y_pred,
    ...     target="delta_gamma",
    ...     feature_importances=1.0 / gpr.kernel_.k1.k2.length_scale,
    ...     feature_names=features,
    ...     datasets={
    ...         "train": (y_train, gpr.predict(X_train)),
    ...         "test":  (y_test,  y_pred),
    ...     },
    ... )
    >>>
    >>> post.plot_parity(save_path="parity", folder="OUTPUTS")
    >>> post.plot_residual_distribution(save_path="residuals", folder="OUTPUTS")
    >>> post.plot_error_vs_std(y_std, save_path="calibration", folder="OUTPUTS")
    >>> post.plot_2d_response_surface(
    ...     model=gpr, X_train=X_train, feature_names=features,
    ...     save_path="surface_T_P", folder="OUTPUTS",
    ... )
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
        datasets: Optional[dict] = None,
    ):
        """
        Initialise MLPostprocessing.

        See class docstring for full parameter descriptions.
        Residuals, R², and RMSE are computed immediately on construction.
        """
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

        # Optional train/test/val datasets for parity plot
        self._datasets = datasets

        # Set colors based on target
        self.colors = TARGET_COLORS.get(self._target, DEFAULT_COLORS)

    @property
    def target(self) -> str:
        """Get target variable name."""
        return self._target

    def _get_target_label(self) -> str:
        r"""Return the full LaTeX axis label for the active target, e.g. ``r"$\Delta\gamma$ / [mN m$^{-1}$]"``."""
        target_labels = {
            "gamma": r"$\gamma$ / $[\mathrm{mN\,m^{-1}}]$",
            "interfacial_thickness": r"$L^{90}_{10}$ / $[\mathrm{nm}]$",
            "delta_gamma": r"$\Delta\gamma$ / $[\mathrm{mN\,m^{-1}}]$",
            "residual": r"$\Delta\gamma$ / $[\mathrm{mN\,m^{-1}}]$",
            "p_dew": r"$P_{\mathrm{dew}}$ / $[\mathrm{bar}]$",
            "p_bubble": r"$P_{\mathrm{bubble}}$ / $[\mathrm{bar}]$",
        }
        return target_labels.get(self._target, self._target)

    def _get_target_symbol(self) -> str:
        r"""Return the bare LaTeX math symbol for the active target, e.g. ``r"\Delta\gamma"``."""
        target_symbols = {
            "gamma": r"\gamma",
            "interfacial_thickness": r"L^{90}_{10}",
            "delta_gamma": r"\Delta\gamma",
            "residual": r"\Delta\gamma",
            "p_dew": r"P_{\mathrm{dew}}",
            "p_bubble": r"P_{\mathrm{bubble}}",
        }
        return target_symbols.get(self._target, self._target)

    def _get_feature_label(self, name: str) -> str:
        """Return a LaTeX label for *name*, consulting ``label_map`` then ``ps.label_map``."""
        name_lower = name.lower()
        if name_lower in self._label_map:
            return self._label_map[name_lower]
        if name_lower in ps.label_map:
            return ps.label_map[name_lower]
        return name

    def plot_feature_importance(
        self,
        scale: Literal["linear", "log"] = "linear",
        color: Optional[str] = None,
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Horizontal bar chart of feature importances with percentage labels.

        Bars are sorted from least to most important (bottom to top) so the
        dominant feature is always at the top of the chart.  Percentage
        contribution relative to the total importance sum is annotated on
        each bar.

        Parameters
        ----------
        scale : {"linear", "log"}, default ``"linear"``
            X-axis scale.  Use ``"log"`` when importances span several orders
            of magnitude (e.g. normalised ARD inverse length-scales).
        color : str, optional
            Bar fill colour.  Defaults to ``"red"`` (linear) or
            ``"steelblue"`` (log).
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.

        Raises
        ------
        ValueError
            If ``feature_importances`` was not supplied at construction time.
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
            ps.save_plot(fig, save_path, folder=folder)

        plt.show()
        return fig, ax

    def plot_parity(
        self,
        model_name: str = "Model",
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
        cv_r2: Optional[float] = None,
        cv_rmse: Optional[float] = None,
        cv_mae: Optional[float] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Actual vs predicted parity plot.

        When ``datasets`` was supplied at construction time, each split
        (train / test / val) is drawn with a distinct colour and its own R²
        appears in the legend.  Otherwise a single scatter of ``y_true`` vs
        ``y_pred`` is drawn.  A perfect-prediction diagonal (``y = x``) is
        always included.  Optional cross-validation scores can be annotated
        in a text box.

        Parameters
        ----------
        model_name : str, default ``"Model"``
            Legend label used only in single-split mode (no ``datasets``).
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.
        cv_r2 : float, optional
            Mean cross-validation R² to annotate in the lower-right corner.
        cv_rmse : float, optional
            Mean cross-validation RMSE to annotate alongside ``cv_r2``.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig, ax = ps.plot_init()

        if self._datasets is not None:
            # Collect all values for axis limits
            all_true, all_pred = [], []
            split_labels = {
                "train": r"$\mathrm{Train}$",
                "test":  r"$\mathrm{Test}$",
                "val":   r"$\mathrm{Validation}$",
            }

            for key in ["train", "test", "val"]:
                if key not in self._datasets:
                    continue
                y_t, y_p = np.asarray(self._datasets[key][0]), np.asarray(self._datasets[key][1])
                all_true.append(y_t)
                all_pred.append(y_p)

                colors = SPLIT_COLORS[key]
                r2 = r2_score(y_t, y_p)
                ax.scatter(
                    y_t, y_p,
                    alpha=0.8, s=20,
                    facecolors=colors["facecolor"],
                    edgecolors=colors["edgecolor"],
                    linewidths=0.6,
                    label=rf"{split_labels[key]} ($R^2$ = {r2:.2f})",
                )

            all_true = np.concatenate(all_true)
            all_pred = np.concatenate(all_pred)
            lims = [
                min(all_true.min(), all_pred.min()) - 0.5,
                max(all_true.max(), all_pred.max()) + 0.5,
            ]
        else:
            # Single scatter (backward-compatible)
            lims = [
                min(self.y_true.min(), self.y_pred.min()) - 0.5,
                max(self.y_true.max(), self.y_pred.max()) + 0.5,
            ]
            ax.scatter(
                self.y_true, self.y_pred,
                alpha=0.8, s=20,
                facecolors=self.colors["facecolor"],
                edgecolors=self.colors["edgecolor"],
                linewidths=0.6,
                label=rf"{model_name} ($R^2$ = {self.r2:.4f})",
            )

        # Perfect prediction line
        ax.plot(lims, lims, "k--", linewidth=1.4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        target_label = self._get_target_label()
        ax.set_xlabel(rf"$\mathrm{{Actual}}$ {target_label}", fontsize=ps.label_fontsize)
        ax.set_ylabel(rf"$\mathrm{{Predicted}}$ {target_label}", fontsize=ps.label_fontsize)

        ps.apply_axis_style(ax)

        ax.legend(
            fontsize=ps.label_fontsize * 0.65,
            loc="upper left",
            frameon=False,
        )

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        # Cross-validation annotation
        if cv_r2 is not None or cv_rmse is not None or cv_mae is not None:
            cv_lines = []
            if cv_r2 is not None:
                cv_lines.append(rf"CV $R^2$ = {cv_r2:.4f}")
            if cv_rmse is not None:
                cv_lines.append(rf"CV RMSE = {cv_rmse:.4f}")
            if cv_mae is not None:
                cv_lines.append(rf"CV MAE = {cv_mae:.4f}")
            cv_text = "\n".join(cv_lines)
            ax.text(
                0.95, 0.05, cv_text,
                transform=ax.transAxes,
                fontsize=ps.label_fontsize * 0.6,
                verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
            )

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path, folder=folder)

        plt.show()
        return fig, ax

    def plot_residual_distribution(
        self,
        bins: int = 40,
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
        cv_r2: Optional[float] = None,
        cv_rmse: Optional[float] = None,
        cv_mae: Optional[float] = None,
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Histogram of signed residuals (y_true − y_pred).

        A dashed zero-residual line is overlaid and a text box shows the mean,
        standard deviation, and RMSE.  Optional cross-validation scores can be
        appended to the annotation.

        Parameters
        ----------
        bins : int, default 40
            Number of histogram bins.
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.
        cv_r2 : float, optional
            Mean cross-validation R² to append to the annotation box.
        cv_rmse : float, optional
            Mean cross-validation RMSE to append to the annotation box.

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
            facecolor=self.colors["facecolor"],
            edgecolor=self.colors["edgecolor"],
            linewidth=0.8,
            label="Residuals"
        )

        ax.axvline(0, color="black", linewidth=1.3, linestyle="--", label="Zero residual")

        mean_resid = self.residuals.mean()
        std_resid = self.residuals.std()

        target_label = self._get_target_label()
        target_sym   = self._get_target_symbol()
        unit = target_label.split("/")[-1].strip() if "/" in target_label else ""
        ax.set_xlabel(rf"Residual ${target_sym}$ / {unit}", fontsize=ps.label_fontsize)
        ax.set_ylabel("Count / [-]", fontsize=ps.label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=ps.title_fontsize * 0.75, fontweight="bold")

        # Add statistics text box
        textstr = (
            f"Mean = {mean_resid:.2f} / {unit}\n"
            f"Std Dev = {std_resid:.2f} / {unit}\n"
            f"RMSE = {self.rmse:.2f} / {unit}"
        )
        if cv_r2 is not None:
            textstr += f"\nCV $R^2$ = {cv_r2:.4f}"
        if cv_rmse is not None:
            textstr += f"\nCV RMSE = {cv_rmse:.4f}"
        if cv_mae is not None:
            textstr += f"\nCV MAE = {cv_mae:.4f}"

        ax.text(
            0.025, 0.9, textstr,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        ps.apply_axis_style(ax)

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(
        #     handles=handles,
        #     fontsize=ps.label_fontsize * 0.5,
        #     loc="upper right",
        #     edgecolor="black",
        #     facecolor="white",
        #     framealpha=1.0
        # )

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path, folder=folder)

        plt.show()
        return fig, ax

    def plot_residual_vs_predicted(
        self,
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
        cv_r2: Optional[float] = None,
        cv_rmse: Optional[float] = None,
        cv_mae: Optional[float] = None,
        title: Optional[str] = None,
        y_range: Optional[Tuple[float, float]] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Scatter plot of residuals (y_true − y_pred) against predicted values.

        Horizontal reference lines are drawn at zero and at ±RMSE to provide
        a visual calibration band.  An optional cross-validation annotation
        is placed in the upper-right corner.

        Parameters
        ----------
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.
        cv_r2 : float, optional
            Mean cross-validation R² to annotate.
        cv_rmse : float, optional
            Mean cross-validation RMSE to annotate.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig, ax = ps.plot_init()

        ax.scatter(
            self.y_pred, self.residuals,
            alpha=0.8, s=20,
            facecolors="lightblue",
            edgecolors="darkblue",
            linewidths=0.6,
        )

        target_label = self._get_target_label()
        target_sym   = self._get_target_symbol()
        unit = target_label.split('/')[-1].strip() if '/' in target_label else ""

        # strip outer [$...$] to get the bare LaTeX unit content for the legend
        unit_inner = unit.strip("$")[1:-1] if (unit.startswith("$[") and unit.endswith("]$")) else unit
        ax.axhline(0, color="black", linewidth=1.3, linestyle="--")
        ax.axhline(self.rmse, color="red", linewidth=1.1, linestyle="-.", alpha=0.85,
                   label=rf"$\pm\mathrm{{RMSE}} = {self.rmse:.2f}\,/\,[{unit_inner}]$")
        ax.axhline(-self.rmse, color="red", linewidth=1.1, linestyle="-.", alpha=0.85)
        ax.set_xlabel(rf"$\mathrm{{Predicted}}$ {target_label}", fontsize=ps.label_fontsize)
        ax.set_ylabel(rf"$\mathrm{{Residual}}$ ${target_sym}$ / {unit}", fontsize=ps.label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=ps.title_fontsize * 0.75, fontweight="bold")

        ps.apply_axis_style(ax)
        ax.legend(
            fontsize=ps.label_fontsize * 0.6,
            loc="best",
            frameon=False,
        )

        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        if y_range is not None:
            ax.set_ylim(y_range)

        # Cross-validation annotation
        if cv_r2 is not None or cv_rmse is not None or cv_mae is not None:
            cv_lines = []
            if cv_r2 is not None:
                cv_lines.append(rf"CV $R^2$ = {cv_r2:.4f}")
            if cv_rmse is not None:
                cv_lines.append(rf"CV RMSE = {cv_rmse:.4f}")
            if cv_mae is not None:
                cv_lines.append(rf"CV MAE = {cv_mae:.4f}")
            cv_text = "\n".join(cv_lines)
            ax.text(
                0.95, 0.95, cv_text,
                transform=ax.transAxes,
                fontsize=ps.label_fontsize * 0.6,
                verticalalignment="top",
                horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.9),
            )

        plt.tight_layout()

        if save_path:
            ps.save_plot(fig, save_path, folder=folder)

        plt.show()
        return fig, ax

    # ------------------------------------------------------------------
    # GPR-specific plots
    # ------------------------------------------------------------------

    def plot_std_histogram(
        self,
        y_std: Union[np.ndarray, List],
        bins: int = 40,
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Histogram of GPR predictive standard deviations σ.

        Useful for inspecting the spread of the model's uncertainty — a
        narrow, low-σ distribution indicates high overall confidence; a
        heavy right tail highlights extrapolation regions.  A text box
        annotates mean, median, and maximum σ.

        Parameters
        ----------
        y_std : array-like, shape (n_samples,)
            Predictive standard deviations from
            ``model.predict(X, return_std=True)``.
        bins : int, default 40
            Number of histogram bins.
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        y_std = np.asarray(y_std)
        fig, ax = ps.plot_init()

        ax.hist(
            y_std,
            bins=bins,
            alpha=0.75,
            facecolor=self.colors["facecolor"],
            edgecolor=self.colors["edgecolor"],
            linewidth=0.8,
        )

        target_label = self._get_target_label()
        target_sym = self._get_target_symbol()
        unit = target_label.split("/")[-1].strip() if "/" in target_label else ""
        ax.set_xlabel(rf"$\sigma_{{{target_sym}}}$ / {unit}", fontsize=ps.label_fontsize)
        ax.set_ylabel("Count / [-]", fontsize=ps.label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=ps.title_fontsize * 0.75, fontweight="bold")

        textstr = (
            f"Mean = {y_std.mean():.2f} / {unit}\n"
            f"Median = {np.median(y_std):.2f} / {unit}\n"
            f"Max = {y_std.max():.2f} / {unit}"
        )
        ax.text(
            0.975, 0.9, textstr,
            transform=ax.transAxes, fontsize=6,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

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

    def plot_parity_colored_by_std(
        self,
        y_std: Union[np.ndarray, List],
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Parity plot with each point coloured by its predictive standard deviation σ.

        High-uncertainty predictions stand out immediately against the
        perfect-prediction diagonal.  The colorbar uses the ``"Reds"`` palette
        so larger σ values are darker.

        Parameters
        ----------
        y_std : array-like, shape (n_samples,)
            Predictive standard deviations from
            ``model.predict(X, return_std=True)``.
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        y_std = np.asarray(y_std)
        fig, ax = ps.plot_init()

        sc = ax.scatter(
            self.y_true, self.y_pred,
            c=y_std, cmap="Reds",
            alpha=0.85, s=20,
            linewidths=0.4, edgecolors="none",
        )

        target_label = self._get_target_label()
        unit = target_label.split("/")[-1].strip() if "/" in target_label else ""
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(rf"$\sigma$ / {unit}", fontsize=ps.label_fontsize * 0.8)
        ps.style_colorbar(cbar)

        lims = [
            min(self.y_true.min(), self.y_pred.min()) - 0.5,
            max(self.y_true.max(), self.y_pred.max()) + 0.5,
        ]
        
        ax.plot(lims, lims, "k--", linewidth=1.4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f"Actual {target_label}", fontsize=ps.label_fontsize)
        ax.set_ylabel(f"Predicted {target_label}", fontsize=ps.label_fontsize)

        ps.apply_axis_style(ax)
        ax.legend(fontsize=ps.label_fontsize * 0.6, loc="upper left",
                  edgecolor="black", framealpha=1.0)
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()
        if save_path:
            ps.save_plot(fig, save_path, folder=folder)
        plt.show()
        return fig, ax

    def plot_error_vs_std(
        self,
        y_std: Union[np.ndarray, List],
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Uncertainty calibration plot: absolute error |ε| vs predictive std σ.

        A well-calibrated probabilistic model has most test points below the
        ``|ε| = σ`` diagonal and nearly all below ``|ε| = 2σ``.  The
        percentage of samples within each band is annotated in a text box.

        Parameters
        ----------
        y_std : array-like, shape (n_samples,)
            Predictive standard deviations from
            ``model.predict(X, return_std=True)``.
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        y_std = np.asarray(y_std)
        abs_error = np.abs(self.residuals)

        fig, ax = ps.plot_init()

        ax.scatter(
            y_std, abs_error,
            alpha=0.6, s=15,
            facecolors=self.colors["facecolor"],
            edgecolors=self.colors["edgecolor"],
            linewidths=0.4,
            label="Test samples",
        )

        lim = max(y_std.max(), abs_error.max()) * 1.05
        ref_x = np.linspace(0, lim, 200)
        ax.plot(ref_x, ref_x, "k--", linewidth=1.3, label=r"$|\epsilon| = \sigma$ (ideal)")
        ax.plot(ref_x, 2 * ref_x, "b-.", linewidth=1.0, alpha=0.7, label=r"$|\epsilon| = 2\sigma$")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)

        target_label = self._get_target_label()
        target_sym   = self._get_target_symbol()
        unit = target_label.split("/")[-1].strip() if "/" in target_label else ""
        ax.set_xlabel(rf"$\sigma_{{{target_sym}}}$ / {unit}", fontsize=ps.label_fontsize)
        ax.set_ylabel(rf"$|{target_sym}|$ / {unit}", fontsize=ps.label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=ps.title_fontsize * 0.75, fontweight="bold")

        within_1s = np.mean(abs_error <= y_std) * 100
        within_2s = np.mean(abs_error <= 2 * y_std) * 100
        textstr = f"Within $1\\sigma$: {within_1s:.1f}%\nWithin $2\\sigma$: {within_2s:.1f}%"
        ax.text(
            0.975, 0.05, textstr,
            transform=ax.transAxes, fontsize=6,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        ps.apply_axis_style(ax)
        ps.style_legend(ax, fontsize=ps.label_fontsize * 0.55, loc="upper left")
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()
        if save_path:
            ps.save_plot(fig, save_path, folder=folder)
        plt.show()
        return fig, ax

    def plot_response_curves(
        self,
        model,
        X_ref: Union[np.ndarray, "pd.Series"],
        feature_names: List[str],
        X_train: Optional[Union[np.ndarray, "pd.DataFrame"]] = None,
        n_points: int = 100,
        return_std: bool = False,
        save_individually: bool = False,
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        One-at-a-time response curves: sweep each feature while holding
        all others fixed at ``X_ref``.

        Parameters
        ----------
        model : sklearn estimator
            Fitted model with a ``predict`` method (Pipeline supported).
        X_ref : array-like, shape (n_features,)
            Reference point (e.g. median of training data) to anchor
            all non-swept features.
        feature_names : List[str]
            Names of features in the same order as ``X_ref`` columns.
        X_train : array-like, shape (n_samples, n_features), optional
            Training data used to determine realistic sweep ranges via
            ``[min, max]`` per feature.  If ``None``, falls back to
            ±50 % around the reference value (0–1 for ``z_`` columns).
        n_points : int, default 100
            Resolution of each sweep.
        return_std : bool, default False
            If ``True``, call ``model.predict(return_std=True)`` and
            shade ±1σ and ±2σ bands around the mean curve.
        save_individually : bool, default False
            If ``True``, save each feature as a separate figure named
            ``{save_path}_{feat_name}``.  The combined overview figure
            is shown on screen but not saved.
            If ``False``, save the combined multi-panel figure.
        save_path : str, optional
            Base filename for saving.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Combined figure and flattened array of Axes.
        """
        X_ref = np.asarray(X_ref, dtype=float).ravel()
        n_features = len(feature_names)

        # Determine sweep bounds
        if X_train is not None:
            X_train_arr = np.asarray(X_train)
            feat_min = X_train_arr.min(axis=0)
            feat_max = X_train_arr.max(axis=0)
        else:
            feat_min = np.where(X_ref > 0, X_ref * 0.5, X_ref * 1.5)
            feat_max = np.where(X_ref > 0, X_ref * 1.5, X_ref * 0.5)
            for i, name in enumerate(feature_names):
                if name.startswith("z_") and 0.0 <= X_ref[i] <= 1.0:
                    feat_min[i], feat_max[i] = 0.0, 1.0

        target_label = self._get_target_label()

        # --- Run predictions once; cache for combined and individual figures ---
        cache = []  # list of (feat_name, x_sweep, y_mean, y_sigma_or_None)
        for i, feat_name in enumerate(feature_names):
            x_sweep = np.linspace(feat_min[i], feat_max[i], n_points)
            X_sweep_arr = np.tile(X_ref, (n_points, 1))
            X_sweep_arr[:, i] = x_sweep
            X_sweep = pd.DataFrame(X_sweep_arr, columns=feature_names)
            if return_std:
                y_mean, y_sigma = model.predict(X_sweep, return_std=True)
            else:
                y_mean = model.predict(X_sweep)
                y_sigma = None
            cache.append((feat_name, x_sweep, y_mean, y_sigma))

        # --- Helper: draw one response curve onto an existing axis ---
        def _draw_curve(ax, feat_name, x_sweep, y_mean, y_sigma, ref_val,
                        show_ylabel=True, collect_handles=False):
            feat_label = self._get_feature_label(feat_name)
            handles = []
            if y_sigma is not None:
                h_band2 = ax.fill_between(x_sweep, y_mean - 2 * y_sigma, y_mean + 2 * y_sigma,
                                          alpha=0.12, color=self.colors["hist_color"])
                h_band = ax.fill_between(x_sweep, y_mean - y_sigma, y_mean + y_sigma,
                                         alpha=0.25, color=self.colors["hist_color"])
                if collect_handles:
                    handles.append((h_band, r"$\pm\sigma$"))
                    handles.append((h_band2, r"$\pm 2\sigma$"))
            h_line, = ax.plot(x_sweep, y_mean, color=self.colors["edgecolor"], linewidth=1.5)
            h_ref = ax.axvline(ref_val, color="gray", linewidth=0.9, linestyle=":", alpha=0.7)
            if collect_handles:
                handles = [(h_line, "Mean"), (h_ref, "Reference")] + handles
            ax.set_xlabel(feat_label, fontsize=ps.label_fontsize * 0.8)
            if show_ylabel:
                ax.set_ylabel(target_label, fontsize=ps.label_fontsize * 0.8)
            else:
                ax.set_ylabel("")
            ps.apply_axis_style(ax)
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(axis="both", which="minor", length=3)
            return handles

        # --- Combined overview figure ---
        ncols = min(3, n_features)
        nrows = int(np.ceil(n_features / ncols))
        panel_w, panel_h = 4.0, 3.0
        fig, axes = ps.plot_init_multi(
            nrows=nrows, ncols=ncols,
            w=ncols * panel_w, h=nrows * panel_h,
        )
        axes_flat = np.asarray(axes).ravel()
        legend_handles = []

        for i, (feat_name, x_sweep, y_mean, y_sigma) in enumerate(cache):
            col = i % ncols
            hdls = _draw_curve(
                axes_flat[i], feat_name, x_sweep, y_mean, y_sigma,
                ref_val=X_ref[i],
                show_ylabel=(col == 0),
                collect_handles=(i == 0),
            )
            if i == 0:
                legend_handles = hdls

        for j in range(n_features, len(axes_flat)):
            axes_flat[j].set_visible(False)

        if legend_handles:
            handles, labels = zip(*legend_handles)
            fig.legend(
                handles, labels,
                loc="lower center",
                ncol=len(legend_handles),
                fontsize=ps.label_fontsize * 0.7,
                edgecolor="black",
                framealpha=1.0,
                bbox_to_anchor=(0.5, -0.02),
            )

        if title is not None:
            fig.suptitle(title, fontsize=ps.title_fontsize * 0.8, fontweight="bold", y=1.01)

        plt.tight_layout(rect=[0, 0.04, 1, 1])

        if save_path and not save_individually:
            ps.save_plot(fig, save_path, folder=folder)

        plt.show()

        # --- Individual figures (one per feature) ---
        if save_individually and save_path:
            for feat_name, x_sweep, y_mean, y_sigma in cache:
                fig_i, ax_i = ps.plot_init()
                _draw_curve(
                    ax_i, feat_name, x_sweep, y_mean, y_sigma,
                    ref_val=X_ref[feature_names.index(feat_name)],
                    show_ylabel=True,
                    collect_handles=False,
                )
                if y_sigma is not None or True:
                    # minimal inline legend for individual plots
                    from matplotlib.patches import Patch
                    from matplotlib.lines import Line2D as _L2D
                    legend_elems = [
                        _L2D([0], [0], color=self.colors["edgecolor"], lw=1.5, label="Mean"),
                        _L2D([0], [0], color="gray", lw=0.9, ls=":", label="Reference"),
                    ]
                    if y_sigma is not None:
                        legend_elems.append(
                            Patch(facecolor=self.colors["hist_color"], alpha=0.25, label=r"$\pm\sigma$")
                        )
                        legend_elems.append(
                            Patch(facecolor=self.colors["hist_color"], alpha=0.12, label=r"$\pm 2\sigma$")
                        )
                    ax_i.legend(handles=legend_elems, fontsize=ps.label_fontsize * 0.6,
                                edgecolor="black", framealpha=1.0, loc="best")
                plt.tight_layout()
                ps.save_plot(fig_i, f"{save_path}_{feat_name}", folder=folder)
                plt.close(fig_i)

        return fig, axes_flat

    def plot_2d_response_surface(
        self,
        model,
        X_train: Union[np.ndarray, "pd.DataFrame"],
        feature_names: List[str],
        y_train: Optional[np.ndarray] = None,
        T_name: str = "T",
        P_name: str = "P",
        n_grid: int = 80,
        return_std: bool = False,
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        2D T–P response surface heatmap (all other features fixed at median).

        Produces a side-by-side figure:
        - **Left panel**: predicted mean coloured by ``RdBu_r`` with contour
          lines and training-data scatter overlaid.
        - **Right panel** (only when ``return_std=True``): predictive
          standard deviation coloured by ``viridis``.

        All features other than ``T_name`` and ``P_name`` are held fixed at
        their column-wise training-set median.

        Parameters
        ----------
        model : estimator
            Fitted model with a ``predict(X, return_std=False)`` method.
            If ``return_std=True`` the model must support
            ``predict(X, return_std=True)``.
        X_train : array-like, shape (n_samples, n_features)
            Training data.  Used to derive sweep ranges (min/max per column)
            and the median reference point; also scattered on the mean panel.
        y_train : array-like, shape (n_samples,), optional
            Training-set target values used to colour the scatter overlay.
            When ``None`` the scatter is drawn with a neutral grey colour
            (data-coverage indicator only).
        feature_names : List[str]
            Feature names in the same order as columns of ``X_train``.
        T_name : str, default "T"
            Name of the temperature feature.
        P_name : str, default "P"
            Name of the pressure feature.
        n_grid : int, default 80
            Number of grid points along each axis.
        return_std : bool, default True
            Whether to add a second panel showing predictive uncertainty.
        save_path : str, optional
            Base filename to save the figure.
        folder : str, default "PLOTS"
            Output folder.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Figure and array of Axes.

        Raises
        ------
        ValueError
            If ``T_name`` or ``P_name`` are not found in ``feature_names``.
        """
        X_train_df = (
            pd.DataFrame(X_train, columns=feature_names)
            if not isinstance(X_train, pd.DataFrame)
            else X_train.copy()
        )

        if T_name not in feature_names:
            raise ValueError(f"T_name={T_name!r} not found in feature_names.")
        if P_name not in feature_names:
            raise ValueError(f"P_name={P_name!r} not found in feature_names.")

        T_idx = feature_names.index(T_name)
        P_idx = feature_names.index(P_name)

        # Grid bounds from training data
        T_vals = np.linspace(X_train_df[T_name].min(), X_train_df[T_name].max(), n_grid)
        P_vals = np.linspace(X_train_df[P_name].min(), X_train_df[P_name].max(), n_grid)
        TT, PP = np.meshgrid(T_vals, P_vals)  # shape (n_grid, n_grid)

        # Reference row: column-wise median
        X_ref = X_train_df.median().values.copy()

        # Build full grid matrix (n_grid² × n_features)
        X_grid = np.tile(X_ref, (n_grid * n_grid, 1))
        X_grid[:, T_idx] = TT.ravel()
        X_grid[:, P_idx] = PP.ravel()
        X_grid_df = pd.DataFrame(X_grid, columns=feature_names)

        # Predict
        if return_std:
            mean_flat, std_flat = model.predict(X_grid_df, return_std=True)
            std_grid = std_flat.reshape(n_grid, n_grid)
        else:
            mean_flat = model.predict(X_grid_df)
            std_grid = None
        mean_grid = mean_flat.reshape(n_grid, n_grid)

        target_label = self._get_target_label()
        target_sym   = self._get_target_symbol()
        unit = target_label.split("/")[-1].strip() if "/" in target_label else ""

        T_label = self._get_feature_label(T_name)
        P_label = self._get_feature_label(P_name)

        ncols = 2 if return_std else 1
        fig, axes = ps.plot_init_multi(nrows=1, ncols=ncols, w=ncols * 5.0, h=4.0)
        axes_flat = np.asarray(axes).ravel()
        ax_mean = axes_flat[0]

        # ── Mean panel ─────────────────────────────────────────────────────
        _abs_max = np.percentile(np.abs(mean_grid), 98)
        _vmin, _vmax = -_abs_max, _abs_max
        cf_mean = ax_mean.contourf(
            TT, PP, mean_grid,
            levels=np.linspace(_vmin, _vmax, 21), cmap="RdBu_r", alpha=0.92,
            vmin=_vmin, vmax=_vmax, extend="both",
        )
        cs_mean = ax_mean.contour(
            TT, PP, mean_grid,
            levels=np.linspace(_vmin, _vmax, 11),
            colors="black", linewidths=0.8, linestyles="solid",
        )
        ax_mean.clabel(
            cs_mean, inline=True, fontsize=ps.label_fontsize * 0.65,
            fmt="%.2g", inline_spacing=4,
        )

        # Training scatter: coloured by y_train when provided, otherwise neutral
        if y_train is not None:
            ax_mean.scatter(
                X_train_df[T_name], X_train_df[P_name],
                c=np.asarray(y_train), cmap="RdBu_r",
                vmin=_vmin, vmax=_vmax,
                s=22, alpha=0.85, linewidths=0.5, edgecolors="white",
                zorder=3,
            )
        else:
            ax_mean.scatter(
                X_train_df[T_name], X_train_df[P_name],
                color="white", s=22, alpha=0.75, linewidths=0.5, edgecolors="grey",
                zorder=3,
            )

        cbar_mean = fig.colorbar(cf_mean, ax=ax_mean, pad=0.02)
        cbar_mean.set_label(
            rf"$\langle {target_sym} \rangle$ / {unit}",
            fontsize=ps.label_fontsize,
        )
        ps.style_colorbar(cbar_mean)

        ax_mean.set_xlabel(T_label, fontsize=ps.label_fontsize)
        ax_mean.set_ylabel(P_label, fontsize=ps.label_fontsize)
        if title is not None:
            ax_mean.set_title(title, fontsize=ps.title_fontsize, fontweight="bold")
        ps.apply_axis_style(ax_mean)
        ax_mean.minorticks_on()
        ax_mean.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_mean.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax_mean.tick_params(axis="both", which="minor", length=3)

        # ── Std panel (optional) ────────────────────────────────────────────
        if return_std and std_grid is not None:
            ax_std = axes_flat[1]
            cf_std = ax_std.contourf(
                TT, PP, std_grid,
                levels=20, cmap="viridis", alpha=0.92,
            )
            cs_std = ax_std.contour(
                TT, PP, std_grid,
                levels=10,
                colors="white", linewidths=0.8, linestyles="solid",
            )
            ax_std.clabel(
                cs_std, inline=True, fontsize=ps.label_fontsize * 0.65,
                fmt="%.2g", inline_spacing=4,
            )
            ax_std.scatter(
                X_train_df[T_name], X_train_df[P_name],
                c="white", s=22, alpha=0.75, linewidths=0.5, edgecolors="grey", zorder=3,
            )
            cbar_std = fig.colorbar(cf_std, ax=ax_std, pad=0.02)
            cbar_std.set_label(
                rf"$\sigma({target_sym})$ / {unit}",
                fontsize=ps.label_fontsize * 0.8,
            )
            ps.style_colorbar(cbar_std)
            ax_std.set_xlabel(T_label, fontsize=ps.label_fontsize * 0.9)
            ax_std.set_ylabel(P_label, fontsize=ps.label_fontsize * 0.9)
            if title is not None:
                ax_std.set_title(title, fontsize=ps.title_fontsize * 0.75, fontweight="bold")
            ps.apply_axis_style(ax_std)
            ax_std.minorticks_on()
            ax_std.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax_std.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax_std.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()
        if save_path:
            ps.save_plot(fig, save_path, folder=folder)
        plt.show()
        return fig, axes_flat

    def plot_reconstructed_parity(
        self,
        datasets_reconstructed: dict,
        n_sigma: float = 2.0,
        model_name: str = "GPR",
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
        title: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Parity plot of the reconstructed full target (γ_cDFT) with uncertainty bars.

        All supplied splits are shown together.  When a 3-tuple
        ``(y_true, y_pred, y_std)`` is provided for a split, ±n_sigma
        error bars are drawn; otherwise plain scatter markers are used.
        The model name is shown in the plot title.

        Parameters
        ----------
        datasets_reconstructed : dict
            Mapping of split name to a 2- or 3-tuple::

                {
                    "train": (y_true, y_pred),            # no error bars
                    "test":  (y_true, y_pred, y_std),     # ±n_sigma bars
                    "val":   (y_true, y_pred, y_std),     # optional
                }

            Supported keys: ``"train"``, ``"test"``, ``"val"``.
        n_sigma : float, default 2.0
            Half-width of error bars in units of σ.
        model_name : str, default ``"GPR"``
            Model identifier shown in the plot title.
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.

        Returns
        -------
        Tuple[plt.Figure, plt.Axes]
            Figure and axes objects.
        """
        fig, ax = ps.plot_init()

        all_true, all_pred = [], []
        split_labels = {"train": "Train", "test": "Test", "val": "Validation"}

        for key in ["train", "test", "val"]:
            if key not in datasets_reconstructed:
                continue
            entry = datasets_reconstructed[key]
            y_t  = np.asarray(entry[0])
            y_p  = np.asarray(entry[1])
            y_s  = np.asarray(entry[2]) if len(entry) > 2 else None
            all_true.append(y_t)
            all_pred.append(y_p)

            colors = SPLIT_COLORS[key]
            r2 = r2_score(y_t, y_p)
            label = rf"{split_labels[key]} ($R^2$ = {r2:.2f})"

            if y_s is not None:
                ax.errorbar(
                    y_t, y_p,
                    yerr=n_sigma * y_s,
                    fmt="o",
                    markersize=3,
                    alpha=0.7,
                    color=colors["edgecolor"],
                    markerfacecolor=colors["facecolor"],
                    markeredgecolor=colors["edgecolor"],
                    elinewidth=0.6,
                    capsize=1.5,
                    capthick=0.6,
                    label=label,
                )
            else:
                ax.scatter(
                    y_t, y_p,
                    alpha=0.8, s=18,
                    facecolors=colors["facecolor"],
                    edgecolors=colors["edgecolor"],
                    linewidths=0.6,
                    label=label,
                )

        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        margin = 0.5
        lims = [
            min(all_true.min(), all_pred.min()) - margin,
            max(all_true.max(), all_pred.max()) + margin,
        ]
        ax.plot(lims, lims, "k--", linewidth=1.4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        target_label = self._get_target_label()
        target_sym   = self._get_target_symbol()
        unit = target_label.split("/")[-1].strip() if "/" in target_label else ""
        ax.set_xlabel(rf"Actual ${target_sym}$ / {unit}", fontsize=ps.label_fontsize)
        ax.set_ylabel(rf"Predicted ${target_sym}$ / {unit}", fontsize=ps.label_fontsize)
        if title is not None:
            ax.set_title(title, fontsize=ps.title_fontsize * 0.75, fontweight="bold")

        # Annotation: error bar caption
        ax.text(
            0.97, 0.03,
            rf"Error bars: $\pm {n_sigma:.0f}\sigma$",
            transform=ax.transAxes,
            fontsize=ps.label_fontsize * 0.6,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=0.85),
        )

        ps.apply_axis_style(ax)
        ax.legend(
            fontsize=ps.label_fontsize * 0.6,
            loc="upper left",
            edgecolor="black",
            framealpha=1.0,
        )
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="both", which="minor", length=3)

        plt.tight_layout()
        if save_path:
            ps.save_plot(fig, save_path, folder=folder)
        plt.show()
        return fig, ax

    def plot_parity_with_residual_distribution(
        self,
        model_name: str = "GPR",
        n_bins: int = 30,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        folder: str = "PLOTS",
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Parity scatter (left) + horizontal residual histogram (right).

        One figure, two side-by-side panels:

        * **Left (wide)**: actual vs predicted parity scatter, coloured by
          split (train / test / val) with per-split R² in the legend.
        * **Right (narrow)**: horizontal histogram of signed residuals
          ``y_true − y_pred`` per split.  Bars run left-to-right; the residual
          value is on the y-axis with a dashed zero-line so symmetry is
          immediately visible.

        Parameters
        ----------
        model_name : str, default ``"GPR"``
            Legend label used only in single-split mode (no ``datasets``).
        n_bins : int, default ``30``
            Number of histogram bins for the residual distribution.
        title : str, optional
            Figure suptitle.
        save_path : str, optional
            Base filename passed to :func:`~thermoift.PLOT_SETTINGS.save_plot`.
        folder : str, default ``"PLOTS"``
            Output directory for the saved figure.

        Returns
        -------
        Tuple[plt.Figure, np.ndarray]
            Figure and ``np.array([ax_scatter, ax_hist])``.
        """
        from matplotlib.gridspec import GridSpec

        target_label = self._get_target_label()
        target_sym   = self._get_target_symbol()
        unit = target_label.split("/")[-1].strip() if "/" in target_label else ""

        split_labels = {"train": "Train", "test": "Test", "val": "Validation"}

        # ── per-split data ──────────────────────────────────────────────────
        if self._datasets is not None:
            splits_data = {}
            for key in ["train", "test", "val"]:
                if key in self._datasets:
                    y_t = np.asarray(self._datasets[key][0])
                    y_p = np.asarray(self._datasets[key][1])
                    splits_data[key] = (y_t, y_p, y_t - y_p)
        else:
            splits_data = {
                "test": (self.y_true, self.y_pred, self.residuals)
            }

        all_true  = np.concatenate([v[0] for v in splits_data.values()])
        all_pred  = np.concatenate([v[1] for v in splits_data.values()])
        all_resid = np.concatenate([v[2] for v in splits_data.values()])

        margin      = 0.5
        parity_lims = [
            min(all_true.min(), all_pred.min()) - margin,
            max(all_true.max(), all_pred.max()) + margin,
        ]
        r_abs        = max(abs(all_resid.min()), abs(all_resid.max()))
        resid_lims   = [-(r_abs * 1.15), r_abs * 1.15]

        # ── figure ──────────────────────────────────────────────────────────
        with plt.style.context(["ieee"]):
            plt.rcParams["font.family"]      = ps.graphic_font
            plt.rcParams["mathtext.fontset"] = ps.math_font
            plt.rcParams["text.usetex"]      = True

            fig = plt.figure(figsize=(5.5, 3.5), layout="constrained")
            gs  = GridSpec(1, 2, width_ratios=[1.5, 1], wspace=0.08, figure=fig)
            ax_scatter = fig.add_subplot(gs[0])
            ax_hist    = fig.add_subplot(gs[1])

        ps.apply_axis_style(ax_scatter)
        ps.apply_axis_style(ax_hist)

        # ── left: parity scatter ─────────────────────────────────────────────
        for key, (y_t, y_p, _) in splits_data.items():
            c   = SPLIT_COLORS[key]
            r2  = r2_score(y_t, y_p)
            lbl = rf"{split_labels[key]} ($R^2$={r2:.2f})"
            ax_scatter.scatter(
                y_t, y_p,
                alpha=0.7, s=14,
                facecolors=c["facecolor"],
                edgecolors=c["edgecolor"],
                linewidths=0.5,
                label=lbl,
            )

        ax_scatter.plot(parity_lims, parity_lims, "k--", linewidth=1.2)
        ax_scatter.set_xlim(parity_lims)
        ax_scatter.set_ylim(parity_lims)
        ax_scatter.set_xlabel(
            rf"Actual ${target_sym}$ / {unit}",
            fontsize=ps.label_fontsize * 0.85,
        )
        ax_scatter.set_ylabel(
            rf"Predicted ${target_sym}$ / {unit}",
            fontsize=ps.label_fontsize * 0.85,
        )
        ax_scatter.minorticks_on()
        ax_scatter.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_scatter.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax_scatter.legend(
            fontsize=ps.legend_fontsize * 0.85,
            loc="upper left",
            edgecolor="black",
            framealpha=1.0,
        )

        # ── right: horizontal residual histogram + KDE curve ─────────────────
        from scipy.stats import gaussian_kde

        for key, (_, _, resid) in splits_data.items():
            c = SPLIT_COLORS[key]
            counts, bin_edges, _ = ax_hist.hist(
                resid,
                bins=n_bins,
                orientation="horizontal",
                facecolor=c["facecolor"],
                edgecolor=c["edgecolor"],
                linewidth=0.4,
                alpha=0.65,
                label=split_labels[key],
            )
            # KDE scaled to match histogram counts
            kde        = gaussian_kde(resid, bw_method=0.3)
            y_grid     = np.linspace(-1.0, 1.0, 300)
            kde_vals   = kde(y_grid)
            bin_width  = bin_edges[1] - bin_edges[0]
            kde_scaled = kde_vals * len(resid) * bin_width
            ax_hist.plot(
                kde_scaled, y_grid,
                color=c["edgecolor"],
                linewidth=1.2,
                linestyle="-",
                zorder=4,
            )

        ax_hist.axhline(0.0, color="k", linestyle="--", linewidth=0.8, zorder=3)
        ax_hist.set_ylim([-1.0, 1.0])
        ax_hist.set_xlim([0, 400])
        ax_hist.set_xlabel("Count / [-]", fontsize=ps.label_fontsize * 0.75)
        ax_hist.set_ylabel(
            rf"Residual ${target_sym}$ / {unit}",
            fontsize=ps.label_fontsize * 0.75,
        )
        ax_hist.yaxis.set_label_position("right")
        ax_hist.minorticks_on()
        ax_hist.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax_hist.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax_hist.tick_params(
            axis="both", which="major", direction="in",
            width=ps.tick_width, length=ps.tick_length,
            bottom=True, top=True, left=True, right=True,
        )
        ax_hist.tick_params(
            axis="both", which="minor", direction="in",
            width=ps.minor_tick_width, length=ps.minor_tick_length,
            bottom=True, top=True, left=True, right=True,
        )
        ax_hist.yaxis.set_tick_params(labelright=True, labelleft=False)

        if title is not None:
            fig.suptitle(title, fontsize=ps.title_fontsize * 0.75,
                         fontweight="bold")

        if save_path:
            ps.save_plot(fig, save_path, folder=folder)
        plt.show()
        return fig, np.array([ax_scatter, ax_hist])

    def summary(self) -> dict:
        """
        Return a dictionary of performance metrics for the primary split.

        Keys
        ----
        target : str
            Target variable name.
        n_samples : int
            Number of samples in the primary split.
        r2 : float
            R² score.
        rmse : float
            Root-mean-squared error.
        mae : float
            Mean absolute error.
        mean_residual : float
            Mean signed residual (bias indicator).
        std_residual : float
            Standard deviation of residuals.
        min_residual : float
            Minimum residual.
        max_residual : float
            Maximum residual.

        Returns
        -------
        dict
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
        """
        Print a formatted performance summary to stdout.

        Displays R², RMSE, MAE, residual statistics, and — when
        ``feature_importances`` was provided — a ranked bar chart of
        feature contributions.
        """
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


def print_model_metrics(
    y_train: Union[np.ndarray, pd.Series],
    y_train_pred: np.ndarray,
    y_test: Union[np.ndarray, pd.Series],
    y_test_pred: np.ndarray,
    target: str,
    unit: str = "bar",
    y_val: Optional[Union[np.ndarray, pd.Series]] = None,
    y_val_pred: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute and print train/test/validation metrics for a regression model.

    Parameters
    ----------
    y_train : array-like
        True training target values.
    y_train_pred : array-like
        Predicted training target values.
    y_test : array-like
        True test target values.
    y_test_pred : array-like
        Predicted test target values.
    target : str
        Target variable name (for display).
    unit : str, default "bar"
        Unit string for RMSE/MAE display.
    y_val : array-like, optional
        True validation target values.
    y_val_pred : array-like, optional
        Predicted validation target values.

    Returns
    -------
    dict
        Dictionary with train/test(/val) R², RMSE, and MAE values.
    """
    metrics = {
        "train_r2": r2_score(y_train, y_train_pred),
        "test_r2": r2_score(y_test, y_test_pred),
        "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
        "train_mae": np.mean(np.abs(np.asarray(y_train) - np.asarray(y_train_pred))),
        "test_mae": np.mean(np.abs(np.asarray(y_test) - np.asarray(y_test_pred))),
    }

    print("=" * 60)
    print(f"Model Performance for {target}")
    print("=" * 60)
    print(f"\nTraining Set:")
    print(f"  R²:   {metrics['train_r2']:.6f}")
    print(f"  RMSE: {metrics['train_rmse']:.6f} {unit}")
    print(f"  MAE:  {metrics['train_mae']:.6f} {unit}")
    print(f"\nTest Set:")
    print(f"  R²:   {metrics['test_r2']:.6f}")
    print(f"  RMSE: {metrics['test_rmse']:.6f} {unit}")
    print(f"  MAE:  {metrics['test_mae']:.6f} {unit}")

    if y_val is not None and y_val_pred is not None:
        metrics["val_r2"] = r2_score(y_val, y_val_pred)
        metrics["val_rmse"] = np.sqrt(mean_squared_error(y_val, y_val_pred))
        metrics["val_mae"] = np.mean(np.abs(np.asarray(y_val) - np.asarray(y_val_pred)))
        print(f"\nValidation Set:")
        print(f"  R²:   {metrics['val_r2']:.6f}")
        print(f"  RMSE: {metrics['val_rmse']:.6f} {unit}")
        print(f"  MAE:  {metrics['val_mae']:.6f} {unit}")

    return metrics


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    method: str = "spearman",
    save_path: Optional[str] = None,
    folder: str = "PLOTS",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a correlation heatmap for features and target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    features : List[str]
        List of feature column names.
    target : str
        Target column name.
    method : str, default "spearman"
        Correlation method ("pearson", "spearman", or "kendall").
    save_path : str, optional
        Base filename to save the figure.

    Returns
    -------
    Tuple[plt.Figure, plt.Axes]
        Figure and axes objects.
    """
    columns = features + [target]

    # Print correlation with target
    corr_with_target = df[columns].corr()[target].drop(target).sort_values(ascending=False)
    print(f"Feature correlations with {target}:")
    print(corr_with_target)

    # Build label map from PLOT_SETTINGS symbol_map
    label_map = {
        col: f"${ps.symbol_map[col.lower()]}$"
        for col in columns
        if col.lower() in ps.symbol_map
    }

    # Correlation heatmap
    numeric_cols = [col for col in columns if df[col].nunique() > 1]
    corr_matrix = df[numeric_cols].corr(method=method)
    corr_matrix = corr_matrix.rename(index=label_map, columns=label_map)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

    fig, ax = ps.plot_init()

    cm = sns.heatmap(
        corr_matrix,
        ax=ax,
        annot=True,
        fmt=".2f",
        cmap=ps.map,
        center=0,
        mask=mask,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 6},
        cbar_kws={"shrink": 1},
    )

    cbar = cm.collections[0].colorbar
    ps.style_colorbar(cbar)
    ps.apply_axis_style(ax)
    ax.tick_params(axis="both", length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    if save_path:
        ps.save_plot(fig, save_path, folder=folder)

    plt.show()
    return fig, ax
