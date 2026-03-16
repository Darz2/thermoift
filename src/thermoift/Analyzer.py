#!/usr/bin/env python
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

class Analyzer:
    """
    Utility class for composition analysis and formatting.

    Main idea:
    - CSV-first workflow
    - auto-detect components from z_/x_/y_ columns
    - allow manual arrays when needed
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
        """Read CSV file."""
        return pd.read_csv(filepath)

    @staticmethod
    def _detect_components(df: pd.DataFrame) -> List[str]:
        """
        Auto-detect components from columns like:
        z_component, x_component, y_component
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
        Use user-provided components if given, otherwise auto-detect from CSV.
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
        Validate one composition set (1D arrays) sums to 1.
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

        Two modes
        ---------
        1) Manual arrays:
           Analyzer.format_table(components, z, x, y, E)

        2) CSV mode:
           Analyzer.format_table(filepath="file.csv", row_index=0)

        In CSV mode:
        - components are auto-detected if not provided
        - E columns are optional
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

        Works in:
        - manual mode
        - CSV mode
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

        If components is None, they are auto-detected from the CSV.
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