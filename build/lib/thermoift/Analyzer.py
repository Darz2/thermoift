#!/usr/bin/env python
"""
Analyzer module for ThermoIFT.

Provides utilities for composition analysis, validation, and formatting
of thermodynamic data from vapor-liquid equilibrium calculations.

Classes
-------
Analyzer
    Utility class for composition analysis and formatting.

Example
-------
>>> from thermoift import Analyzer
>>> result = Analyzer.load_csv("vle_results.csv")
>>> print(result["summary"])
>>> table = Analyzer.format_table(filepath="vle_results.csv", row_index=0)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List


class Analyzer:
    """
    Utility class for composition analysis and formatting.

    This class provides static methods for analyzing and validating
    thermodynamic composition data from vapor-liquid equilibrium (VLE)
    calculations. It supports both CSV-first workflows with auto-detection
    of components and manual array inputs.

    The class handles three types of compositions:
    - z: Feed composition (overall)
    - x: Liquid phase composition
    - y: Vapor phase composition
    - E: Enrichment factors (optional)

    Class Attributes
    ----------------
    DEFAULT_DECIMALS : int
        Default decimal places for formatting (6).
    DEFAULT_TOL : float
        Default tolerance for sum validation (1e-6).
    DEFAULT_COMPONENTS : List[str]
        List of commonly used component names.

    Methods
    -------
    Validation:
        validate_sums(z, x, y, tol)
            Validate that compositions sum to 1.

    Formatting:
        format_table(components, z, x, y, E, filepath, row_index)
            Format one composition row as a DataFrame.

    Reporting:
        report(components, z, x, y, E, T_K, P_bar, filepath, row_index)
            Create a complete report for one composition row.

    Data Loading:
        load_csv(filepath, components)
            Load CSV and validate all rows automatically.

    Example
    -------
    >>> from thermoift import Analyzer
    >>>
    >>> # CSV mode - auto-detect components
    >>> result = Analyzer.load_csv("vle_results.csv")
    >>> print(f"Valid rows: {result['n_valid_rows']}/{result['n_rows']}")
    >>> print(result["summary"])
    >>>
    >>> # Format a single row
    >>> table = Analyzer.format_table(filepath="vle_results.csv", row_index=0)
    >>> print(table)
    >>>
    >>> # Get complete report
    >>> report = Analyzer.report(filepath="vle_results.csv", row_index=0)
    >>> print(f"T = {report['T_K']} K, P = {report['P_bar']} bar")
    >>> print(report["table"])
    >>>
    >>> # Manual mode
    >>> components = ["carbon dioxide", "methane"]
    >>> z = [0.9, 0.1]
    >>> x = [0.95, 0.05]
    >>> y = [0.85, 0.15]
    >>> validation = Analyzer.validate_sums(z, x, y)
    >>> print(f"All valid: {validation['all_valid']}")
    """

    DEFAULT_DECIMALS = 6
    DEFAULT_TOL = 1e-6

    DEFAULT_COMPONENTS = [
        "carbon dioxide",
        "hydrogen",
        "nitrogen",
        "argon",
        "methane",
        "oxygen",
        "water",
        "carbon monoxide",
        "nitrogen dioxide",
        "nitrogen oxide",
        "sulfur dioxide",
        "hydrogen sulfide",
        "propane",
        "ethane",
    ]

    @staticmethod
    def _read_csv(filepath: str) -> pd.DataFrame:
        """
        Read CSV file into DataFrame.

        Parameters
        ----------
        filepath : str
            Path to CSV file.

        Returns
        -------
        pd.DataFrame
            Loaded data.
        """
        return pd.read_csv(filepath)

    @staticmethod
    def _detect_components(df: pd.DataFrame) -> List[str]:
        """
        Auto-detect components from column naming convention.

        Looks for columns matching z_component, x_component, y_component
        and returns components present in all three.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.

        Returns
        -------
        List[str]
            Sorted list of detected component names.

        Raises
        ------
        ValueError
            If no common components found across z_/x_/y_ columns.
        """
        z_components = {col[2:] for col in df.columns if col.startswith("z_")}
        x_components = {col[2:] for col in df.columns if col.startswith("x_")}
        y_components = {col[2:] for col in df.columns if col.startswith("y_")}

        common = sorted(z_components & x_components & y_components)

        if not common:
            raise ValueError(
                "Could not detect any common components from z_/x_/y_ columns in the CSV."
            )

        return common

    @staticmethod
    def _resolve_components(
        df: pd.DataFrame,
        components: Optional[List[str]] = None
    ) -> List[str]:
        """
        Resolve component list from user input or auto-detection.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        components : List[str], optional
            User-provided component list. If None, auto-detects.

        Returns
        -------
        List[str]
            Resolved component list.

        Raises
        ------
        KeyError
            If user-provided components are missing required columns.
        """
        if components is None:
            return Analyzer._detect_components(df)

        missing = []
        for c in components:
            for prefix in ("z_", "x_", "y_"):
                col = f"{prefix}{c}"
                if col not in df.columns:
                    missing.append(col)

        if missing:
            raise KeyError(
                "The following required columns were not found in the CSV:\n"
                + "\n".join(missing)
            )

        return components

    @staticmethod
    def _build_column_names(components: List[str]):
        """
        Build column name lists for z, x, y compositions.

        Parameters
        ----------
        components : List[str]
            List of component names.

        Returns
        -------
        Tuple[List[str], List[str], List[str]]
            (z_cols, x_cols, y_cols) column name lists.
        """
        z_cols = [f"z_{c}" for c in components]
        x_cols = [f"x_{c}" for c in components]
        y_cols = [f"y_{c}" for c in components]
        return z_cols, x_cols, y_cols

    @staticmethod
    def validate_sums(
        z,
        x,
        y,
        tol: Optional[float] = None
    ) -> Dict:
        """
        Validate that composition arrays sum to 1.

        Parameters
        ----------
        z : array-like
            Feed composition (1D array).
        x : array-like
            Liquid phase composition (1D array).
        y : array-like
            Vapor phase composition (1D array).
        tol : float, optional
            Tolerance for validation. Defaults to DEFAULT_TOL (1e-6).

        Returns
        -------
        Dict
            Dictionary containing:
            - z_sum: Sum of z composition
            - x_sum: Sum of x composition
            - y_sum: Sum of y composition
            - all_valid: True if all sums are within tolerance of 1.0

        Example
        -------
        >>> z = [0.9, 0.1]
        >>> x = [0.95, 0.05]
        >>> y = [0.85, 0.15]
        >>> result = Analyzer.validate_sums(z, x, y)
        >>> print(result["all_valid"])  # True
        """
        if tol is None:
            tol = Analyzer.DEFAULT_TOL

        z = np.asarray(z, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        z_sum = np.sum(z)
        x_sum = np.sum(x)
        y_sum = np.sum(y)

        all_valid = (
            np.isclose(z_sum, 1.0, atol=tol) and
            np.isclose(x_sum, 1.0, atol=tol) and
            np.isclose(y_sum, 1.0, atol=tol)
        )

        return {
            "z_sum": float(z_sum),
            "x_sum": float(x_sum),
            "y_sum": float(y_sum),
            "all_valid": bool(all_valid),
        }

    @staticmethod
    def format_table(
        components: Optional[List[str]] = None,
        z=None,
        x=None,
        y=None,
        E=None,
        decimals: Optional[int] = None,
        filepath: Optional[str] = None,
        row_index: int = 0,
    ) -> pd.DataFrame:
        """
        Format one composition row as a compact DataFrame.

        Supports two modes:
        1. Manual mode: Provide components, z, x, y arrays directly
        2. CSV mode: Provide filepath and row_index

        Parameters
        ----------
        components : List[str], optional
            Component names (required for manual mode).
        z : array-like, optional
            Feed composition (required for manual mode).
        x : array-like, optional
            Liquid composition (required for manual mode).
        y : array-like, optional
            Vapor composition (required for manual mode).
        E : array-like, optional
            Enrichment factors. Defaults to NaN if not provided.
        decimals : int, optional
            Decimal places for formatting. Defaults to DEFAULT_DECIMALS.
        filepath : str, optional
            Path to CSV file (CSV mode).
        row_index : int, default 0
            Row index to extract (CSV mode).

        Returns
        -------
        pd.DataFrame
            Formatted table with columns:
            Component, z (Feed), x (Liquid), y (Vapor), E (Enrichment)

        Raises
        ------
        ValueError
            If neither manual arrays nor filepath provided.
        IndexError
            If row_index is out of bounds.

        Example
        -------
        >>> # CSV mode
        >>> table = Analyzer.format_table(filepath="data.csv", row_index=0)
        >>>
        >>> # Manual mode
        >>> table = Analyzer.format_table(
        ...     components=["CO2", "CH4"],
        ...     z=[0.9, 0.1],
        ...     x=[0.95, 0.05],
        ...     y=[0.85, 0.15]
        ... )
        """
        if decimals is None:
            decimals = Analyzer.DEFAULT_DECIMALS

        if filepath is not None:
            df = Analyzer._read_csv(filepath)
            components = Analyzer._resolve_components(df, components)
            z_cols, x_cols, y_cols = Analyzer._build_column_names(components)

            if row_index < 0 or row_index >= len(df):
                raise IndexError(
                    f"row_index={row_index} is out of bounds for CSV with {len(df)} rows."
                )

            row = df.iloc[row_index]

            z = row[z_cols].to_numpy(dtype=float)
            x = row[x_cols].to_numpy(dtype=float)
            y = row[y_cols].to_numpy(dtype=float)

            # Try to read E columns if present; otherwise fill with NaN
            e_cols = [f"E_{c}" for c in components]
            if all(col in df.columns for col in e_cols):
                E = row[e_cols].to_numpy(dtype=float)
            else:
                E = np.full(len(components), np.nan, dtype=float)

        if components is None or z is None or x is None or y is None:
            raise ValueError(
                "You must provide either:\n"
                "1) filepath (CSV mode), or\n"
                "2) components, z, x, y (manual mode)."
            )

        if E is None:
            E = np.full(len(components), np.nan, dtype=float)

        z = np.asarray(z, dtype=float)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        E = np.asarray(E, dtype=float)

        if not (len(components) == len(z) == len(x) == len(y) == len(E)):
            raise ValueError(
                "Length mismatch: components, z, x, y, and E must all have the same length."
            )

        out = pd.DataFrame({
            "Component": components,
            "z (Feed)": z,
            "x (Liquid)": x,
            "y (Vapor)": y,
            "E (Enrichment)": E,
        })

        for col in ["z (Feed)", "x (Liquid)", "y (Vapor)", "E (Enrichment)"]:
            out[col] = out[col].apply(
                lambda v: "" if pd.isna(v) else f"{v:.{decimals}f}"
            )

        return out

    @staticmethod
    def report(
        components: Optional[List[str]] = None,
        z=None,
        x=None,
        y=None,
        E=None,
        T_K: Optional[float] = None,
        P_bar: Optional[float] = None,
        filepath: Optional[str] = None,
        row_index: int = 0,
    ) -> Dict:
        """
        Create a complete report for one composition row.

        Includes formatted table, validation results, and thermodynamic
        conditions (T, P) if available.

        Parameters
        ----------
        components : List[str], optional
            Component names.
        z, x, y : array-like, optional
            Composition arrays.
        E : array-like, optional
            Enrichment factors.
        T_K : float, optional
            Temperature in Kelvin.
        P_bar : float, optional
            Pressure in bar.
        filepath : str, optional
            Path to CSV file.
        row_index : int, default 0
            Row index to extract.

        Returns
        -------
        Dict
            Dictionary containing:
            - table: Formatted composition DataFrame
            - validation: Results from validate_sums()
            - T_K: Temperature (if available)
            - P_bar: Pressure (if available)

        Example
        -------
        >>> report = Analyzer.report(filepath="data.csv", row_index=5)
        >>> print(f"T = {report['T_K']} K")
        >>> print(report["table"])
        >>> print(f"Valid: {report['validation']['all_valid']}")
        """
        if filepath is not None:
            df = Analyzer._read_csv(filepath)
            components = Analyzer._resolve_components(df, components)
            z_cols, x_cols, y_cols = Analyzer._build_column_names(components)

            if row_index < 0 or row_index >= len(df):
                raise IndexError(
                    f"row_index={row_index} is out of bounds for CSV with {len(df)} rows."
                )

            row = df.iloc[row_index]

            z = row[z_cols].to_numpy(dtype=float)
            x = row[x_cols].to_numpy(dtype=float)
            y = row[y_cols].to_numpy(dtype=float)

            e_cols = [f"E_{c}" for c in components]
            if all(col in df.columns for col in e_cols):
                E = row[e_cols].to_numpy(dtype=float)
            else:
                E = np.full(len(components), np.nan, dtype=float)

            if T_K is None and "T_K" in df.columns:
                T_K = float(row["T_K"])
            if P_bar is None and "P_bar" in df.columns:
                P_bar = float(row["P_bar"])

        table = Analyzer.format_table(
            components=components,
            z=z,
            x=x,
            y=y,
            E=E,
        )
        validation = Analyzer.validate_sums(z, x, y)

        return {
            "table": table,
            "validation": validation,
            "T_K": T_K,
            "P_bar": P_bar,
        }

    @staticmethod
    def load_csv(
        filepath: str,
        components: Optional[List[str]] = None
    ) -> Dict:
        """
        Load CSV and validate all rows automatically.

        Performs bulk validation of composition data and provides
        summary statistics including ranges for T, P, gamma, and E.

        Parameters
        ----------
        filepath : str
            Path to CSV file.
        components : List[str], optional
            Component names. If None, auto-detected from columns.

        Returns
        -------
        Dict
            Dictionary containing:
            - df: Full DataFrame
            - components: List of component names
            - valid: Boolean array indicating valid rows
            - n_rows: Total number of rows
            - n_valid_rows: Number of valid rows
            - summary: Dict with detailed statistics including:
                - components, total_rows, valid_rows, invalid_rows
                - all_valid, z_sum_range, x_sum_range, y_sum_range
                - T_range_K, P_range_bar (if columns exist)
                - gamma_range (if column exists)
                - E_range, E_range_by_component (if columns exist)

        Example
        -------
        >>> result = Analyzer.load_csv("vle_results.csv")
        >>> print(f"Loaded {result['n_rows']} rows")
        >>> print(f"Valid: {result['n_valid_rows']}")
        >>> print(f"T range: {result['summary']['T_range_K']}")
        >>> print(f"P range: {result['summary']['P_range_bar']}")
        >>>
        >>> # Access the full DataFrame
        >>> df = result["df"]
        >>> valid_df = df[result["valid"]]
        """
        df = Analyzer._read_csv(filepath)
        components = Analyzer._resolve_components(df, components)
        z_cols, x_cols, y_cols = Analyzer._build_column_names(components)

        z = df[z_cols].to_numpy(dtype=float)
        x = df[x_cols].to_numpy(dtype=float)
        y = df[y_cols].to_numpy(dtype=float)

        tol = Analyzer.DEFAULT_TOL

        z_sums = np.sum(z, axis=1)
        x_sums = np.sum(x, axis=1)
        y_sums = np.sum(y, axis=1)

        all_valid = (
            np.isclose(z_sums, 1.0, atol=tol) &
            np.isclose(x_sums, 1.0, atol=tol) &
            np.isclose(y_sums, 1.0, atol=tol)
        )

        n_valid_rows = int(np.sum(all_valid))
        n_rows = len(df)

        summary = {
            "components": components,
            "total_rows": n_rows,
            "valid_rows": n_valid_rows,
            "invalid_rows": int(n_rows - n_valid_rows),
            "all_valid": bool(np.all(all_valid)),
            "z_sum_range": (float(np.min(z_sums)), float(np.max(z_sums))),
            "x_sum_range": (float(np.min(x_sums)), float(np.max(x_sums))),
            "y_sum_range": (float(np.min(y_sums)), float(np.max(y_sums))),
        }

        # Temperature range
        if "T_K" in df.columns:
            summary["T_range_K"] = (
                float(df["T_K"].min()),
                float(df["T_K"].max())
            )

        # Pressure range
        if "P_bar" in df.columns:
            summary["P_range_bar"] = (
                float(df["P_bar"].min()),
                float(df["P_bar"].max())
            )

        # Gamma range
        if "gamma" in df.columns:
            summary["gamma_range"] = (
                float(df["gamma"].min()),
                float(df["gamma"].max())
            )

        # E ranges
        e_cols = [f"E_{c}" for c in components if f"E_{c}" in df.columns]
        if e_cols:
            e_values = df[e_cols].to_numpy(dtype=float)

            summary["E_range"] = (
                float(np.nanmin(e_values)),
                float(np.nanmax(e_values))
            )

            summary["E_range_by_component"] = {
                c: (
                    float(df[f"E_{c}"].min()),
                    float(df[f"E_{c}"].max())
                )
                for c in components
                if f"E_{c}" in df.columns
            }

        return {
            "df": df,
            "components": components,
            "valid": all_valid,
            "n_rows": n_rows,
            "n_valid_rows": n_valid_rows,
            "summary": summary,
        }