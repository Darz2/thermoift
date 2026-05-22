#!/usr/bin/env python

import json, numpy as np, sympy as sp
import io, base64, matplotlib.pyplot as plt, matplotlib.image as mpimg
from pathlib import Path
from itertools import combinations

# Jupyter notebook display check
try:
    from IPython.display import display
    HAS_DISPLAY = True
except Exception:
    HAS_DISPLAY = False  
    
class KIJMatrixBuilder:
    """
    Build kij polynomial matrices from per-pair KIJ.json files.

    Run this inside the Kij/ directory that contains pair folders like:
        CO2_Ar/KIJ.json, H2_Ar/KIJ.json, CO2_H2/KIJ.json

    Supported KIJ.json format (multiple models per pair):
    {
      "T_min": ...,
      "T_max": ...,
      "constant": {"deg":0, "mean":..., "std":...},
      "linear": {"deg":1, "coeffs":[a1,a0]},
      "quadratic": {"deg":2, "coeffs":[a2,a1,a0]}
    }

    Model choices:
      - "quadratic" | "linear" | "constant" | "best" | "zero"
        * "best" selects quadratic->linear->constant (first available)
        * "zero" forces kij(T)=0 (deg=0 constant 0)
    """

    def __init__(self, root: Path | str = ".", kij_filename: str = "KIJ.json", verbose: bool = False):
        self.root = Path(root)
        self.kij_filename = kij_filename
        self.verbose = verbose
        
    # ============================================================
    # Pair handling
    # ============================================================

    @staticmethod
    def _pair_folder_candidates(c1: str, c2: str):
        return [f"{c1}_{c2}", f"{c2}_{c1}"]

    @staticmethod
    def _canon_pair(a: str, b: str):
        return tuple(sorted((a, b)))

    @staticmethod
    def _select_model_block(d: dict, model: str):
        """
        model: 'constant', 'linear', 'quadratic', 'best', or 'zero'
        Returns: (model_used, deg, coeffs_desc)
          - coeffs_desc: [a_deg, ..., a0] (descending power, numpy polyfit order)
          - constant uses mean -> deg=0: [a0]
          - zero forces deg=0 with a0=0.0
        """
        if model == "zero":
            return "zero", 0, [0.0]

        available = [k for k in ("quadratic", "linear", "constant") if k in d]

        if model == "best":
            for m in ("quadratic", "linear", "constant"):
                if m in d:
                    model = m
                    break
            else:
                raise KeyError("No model blocks found (expected constant/linear/quadratic).")

        if model not in d:
            raise KeyError(f"Requested model='{model}' not found. Available: {available}")

        block = d[model]

        if model == "constant":
            deg = 0
            coeffs_desc = [float(block["mean"])]
        else:
            deg = int(block["deg"])
            coeffs_desc = [float(x) for x in block["coeffs"]]

        return model, deg, coeffs_desc

    def load_pair_kij(self, c1: str, c2: str, model: str = "best"):
        """
        Load KIJ.json for (c1,c2) from either c1_c2/ or c2_c1/.

        Returns dict:
          {
            "model": str,
            "deg": int,
            "coeffs_desc": [...],   # [a_deg, ..., a0]
            "coeffs_asc":  [...],   # [a0, a1, ..., a_deg]
            "T_min": float|None,
            "T_max": float|None,
            "path": str
          }
        """
        # Handle "zero" model without requiring a file
        if model == "zero":
            return {
                "model": "zero",
                "deg": 0,
                "coeffs_desc": [0.0],
                "coeffs_asc": [0.0],
                "T_min": None,
                "T_max": None,
                "path": None,
            }

        for folder in self._pair_folder_candidates(c1, c2):
            path = self.root / folder / self.kij_filename
            if path.exists():
                d = json.loads(path.read_text(encoding="utf-8"))
                model_used, deg, coeffs_desc = self._select_model_block(d, model=model)
                coeffs_asc = coeffs_desc[::-1]  # [a0, a1, ...]

                return {
                    "model": model_used,
                    "deg": deg,
                    "coeffs_desc": coeffs_desc,
                    "coeffs_asc": coeffs_asc,
                    "T_min": d.get("T_min", None),
                    "T_max": d.get("T_max", None),
                    "path": str(path),
                }

        raise FileNotFoundError(
            f"Missing {self.kij_filename} for pair {c1}-{c2}. "
            f"Tried folders: {', '.join(self._pair_folder_candidates(c1, c2))}"
        )

    def load_pairs(self, components, model_map: dict | None = None):
        """
        Load all binary pairs for a system of N components.

        model_map keys can be ('A','B') or ('B','A') – order does not matter.
        Example:
        {('CO2','Ar'):'constant', ('H2','Ar'):'linear', ('CO2','H2'):'linear'}

        If a pair is not provided in model_map: default = 'best'

        Parameters
        ----------
        components : list of str
            List of component names (any length >= 2).
        model_map : dict, optional
            Maps component pairs to kij model type.

        Returns
        -------
        pair_data : dict
            Dictionary keyed by (a, b) tuples with loaded kij pair data.
        """
        if len(components) < 2:
            raise ValueError("components must have at least 2 components")

        required_pairs = list(combinations(components, 2))
        pair_data = {}

        for a, b in required_pairs:
            chosen_model = "best"
            if model_map is not None:
                if (a, b) in model_map:
                    chosen_model = model_map[(a, b)]
                elif (b, a) in model_map:
                    chosen_model = model_map[(b, a)]
                else:
                    key = self._canon_pair(a, b)
                    if key in model_map:
                        chosen_model = model_map[key]

            pair_data[(a, b)] = self.load_pair_kij(a, b, model=chosen_model)

        return pair_data
    
    # ============================================================
    # Model-map helpers from a pair-status table (your grid)
    # ============================================================

    @classmethod
    def make_model_map_from_status(
        cls,
        pair_status: dict,
        checked_model: str = "best",
        zero_model: str = "zero",
    ):
        """
        Convert a pair-status table into model_map.

        pair_status format:
          {
            ("CO2","H2"): "check",     # green checkmark
            ("H2","O2"): 0,            # yellow '0'
            ("CO2","Ar"): "linear",    # explicit
          }

        Values:
          - "check"  -> checked_model (default: "best")
          - 0 or "0" -> zero_model (default: "zero")
          - "best" | "linear" | "quadratic" | "constant" | "zero" -> used directly

        Returns model_map with canonical (sorted) keys.
        """
        model_map = {}
        for (a, b), v in pair_status.items():
            key = cls._canon_pair(a, b)

            if v == "check":
                model_map[key] = checked_model
            elif v == 0 or v == "0":
                model_map[key] = zero_model
            elif isinstance(v, str):
                model_map[key] = v
            else:
                raise ValueError(f"Unsupported pair_status value {v!r} for pair {(a, b)}")

        return model_map

    @classmethod
    def default_pair_status_table(cls):
        """
        Placeholder: Fill this once to mirror your grid.

        Use:
        explicit 'linear'/'constant'/'quadratic'/'best'/'zero'

        NOTE: This is intentionally not complete; populate it to match your grid.
        """
        return {
            # Examples (edit/extend):
            ("CO2", "H2"): "check",
            ("CO2", "Ar"): "check",
            ("CO2", "N2"): "check",
            ("CO2", "CH4"): "check",
            ("CO2", "O2"): "check",
            ("CO2", "CO"): "check",

            ("H2", "Ar"): "check",
            ("H2", "N2"): "check",
            ("H2", "CH4"): "check",
            ("H2", "O2"): 0,

            ("Ar", "N2"): "check",
            ("Ar", "CH4"): "check",
            ("Ar", "H2O"): 0,

            ("N2", "CH4"): "check",
            ("N2", "H2O"): 0,

            ("CH4", "O2"): 0,

            ("O2", "H2O"): 0,
            ("O2", "CO"): 0,
        }

    # ============================================================
    # Tensor build + evaluation
    # ============================================================

    @staticmethod
    def build_polynomial_tensor(components, pair_data):
        """
        Build A[k] matrices for kij(T) = sum_k A[k] * T^k.

        Returns:
        components: list of component names
        A: list of numpy arrays (A[0], A[1], ..., A[max_deg])
        max_deg: maximum polynomial degree across all pairs
        """
        components  = list(components)
        idx         = {c: i for i, c in enumerate(components)}
        n           = len(components)

        max_deg     = max(info["deg"] for info in pair_data.values())
        A           = [np.zeros((n, n), float) for _ in range(max_deg + 1)]

        for (c1, c2), info in pair_data.items():

            i, j = idx[c1], idx[c2]
            coeffs = info["coeffs_asc"]

            for k, ak in enumerate(coeffs):
                A[k][i, j] = ak
                A[k][j, i] = ak

        for Ak in A:
            np.fill_diagonal(Ak, 0.0)

        return components, A, max_deg

    @staticmethod
    def kij_numeric_matrix(T, A):
        """Evaluate K(T) = sum_k A[k] * T^k."""
        T = float(T)
        K = np.zeros_like(A[0])
        for k, Ak in enumerate(A):
            K += Ak * (T ** k)
        return K
    
    # ============================================================
    # Sympy symbolic
    # ============================================================

    @staticmethod
    def kij_sympy_matrix(A, digits=4):
        """Build symbolic K(T) from A[k], rounding for display."""
        T = sp.Symbol("T", real=True)
        n = A[0].shape[0]
        K = sp.zeros(n, n)

        for k, Ak in enumerate(A):
            Ak_round = np.round(Ak, digits)
            K += sp.Matrix(Ak_round) * (T ** k)

        K = K.applyfunc(lambda e: sp.Poly(sp.expand(e), T).as_expr())
        return T, K

    @staticmethod
    def kij_sympy_equation_for_pair(components, A, pair, digits=4):
        """Return sympy expression kij(T) for one pair using A[k]."""
        c1, c2 = pair
        idx = {c: i for i, c in enumerate(components)}
        i, j = idx[c1], idx[c2]

        T = sp.Symbol("T", real=True)
        expr = 0
        for k, Ak in enumerate(A):
            expr += round(float(Ak[i, j]), digits) * (T ** k)

        expr = sp.Poly(sp.expand(expr), T).as_expr()
        return T, expr
    
    # ============================================================
    # Validation + reporting
    # ============================================================

    @staticmethod
    def validate_pairs(pair_data, components):
        required_pairs = list(combinations(components, 2))
        for p in required_pairs:
            if p not in pair_data:
                raise FileNotFoundError(f"Missing pair data for {p}")

        degs = {p: pair_data[p]["deg"] for p in required_pairs}
        unique_degs = sorted(set(degs.values()))
        if len(unique_degs) > 1:
            print(f"Mixed polynomial degrees across pairs: {degs} (allowed)")
        else:
            print(f"All pairs use deg={unique_degs[0]}")
        return degs

    @staticmethod
    def warn_if_T_outside_ranges(pair_data, components, T):
        
        T = float(T)
        required_pairs = list(combinations(components, 2))

        for a, b in required_pairs:
            info = pair_data[(a, b)]
            Tmin = info.get("T_min", None)
            Tmax = info.get("T_max", None)

            if Tmin is None or Tmax is None:
                print(f"No T_min/T_max for {a}-{b} ({info['path']})")
                continue

            if T < Tmin or T > Tmax:
                print(f"Warning: T={T} K outside [{Tmin}, {Tmax}] for pair {a}-{b} (file: {info['path']})")

    @staticmethod
    def print_pair_summary(pair_data, components):
        required_pairs = list(combinations(components, 2))
        print("\nPair summary (from KIJ.json):")
        for a, b in required_pairs:
            info = pair_data[(a, b)]
            print(
                f"  {a}-{b}: model={info['model']}, deg={info['deg']}, "
                f"coeffs={info['coeffs_desc']}, file={info['path']}"
            )
    
    # ============================================================
    # Convenience: end-to-end build + numeric/symbolic output
    # ============================================================

    def build_and_display(
        self,
        components,
        model_map=None,
        T_eval=300.0,
        digits=4,
        show_matrix=True,
        show_pair_equations=True,
    ):
        """
        End-to-end helper:
          load pairs -> validate -> build tensor -> numeric & symbolic output.

        Returns:
          components, A, Knum, (T_symbol, Ksym), pair_data
        """
        pair_data = self.load_pairs(components, model_map=model_map)

        if self.verbose:
            self.print_pair_summary(pair_data, components)
            self.validate_pairs(pair_data, components)
            self.warn_if_T_outside_ranges(pair_data, components, T_eval)

        components, A, max_deg = self.build_polynomial_tensor(components, pair_data)

        Knum = self.kij_numeric_matrix(T_eval, A)
        print(f"\nK({T_eval} K) =")
        print(Knum)

        T_symbol, Ksym = self.kij_sympy_matrix(A, digits=digits)

        if show_matrix:
            print("\nSymbolic K(T) matrix:")
            if HAS_DISPLAY:
                display(Ksym)
            else:
                sp.init_printing(use_unicode=True)
                print(sp.pretty(Ksym))

        if show_pair_equations:
            print("\nPair equations:")
            for pair in combinations(components, 2):
                Tpair, expr = self.kij_sympy_equation_for_pair(components, A, pair, digits=digits)
                ksym = sp.Function(f"k_{{{pair[0]},{pair[1]}}}")
                eq = sp.Eq(ksym(Tpair), expr)
                if HAS_DISPLAY:
                    display(eq)
                else:
                    print(sp.pretty(eq))

        return components, A, Knum, (T_symbol, Ksym), pair_data
    
    # ============================================================
    # Plot display from stored base64 images in KIJ.json
    # ============================================================

    def show_pair_plot(self, c1: str, c2: str, mode="plot"):
        """
        Display a stored base64 plot from a pair's KIJ.json.

        plot_key options typically:
            - "kij_vs_T"
            - "legend"
            - or any key stored under d["plots"]
        mode : str
            "plot" (default) - show kij_vs_T plot
            "legend" - show legend only
            "all" - show all available plots in the KIJ.json
        """
        for folder in self._pair_folder_candidates(c1, c2):
            
            path = (self.root / folder / self.kij_filename).resolve()
            
            if path.exists():
                
                d = json.loads(path.read_text(encoding="utf-8"))

                if "plots" not in d:
                    raise KeyError(f"No 'plots' key found in {path}")

                available_keys = list(d["plots"].keys())
                
                def _display_image(plot_key):
                    if plot_key not in d["plots"]:
                        print(f"Plot '{plot_key}' not found. Available: {available_keys}")
                        return

                    img_bytes = base64.b64decode(d["plots"][plot_key])
                    img = mpimg.imread(io.BytesIO(img_bytes))

                    fig, ax = plt.subplots()
                    ax.imshow(img)
                    ax.axis("off")
                    ax.set_title(f"{c1}-{c2} | {plot_key}")
                    plt.tight_layout()
                    plt.show()
                
                if mode == "plot":
                    _display_image("kij_vs_T")
                elif mode == "legend":
                    _display_image("legend")
                elif mode == "all":
                    _display_image("kij_vs_T")
                    _display_image("legend")
                else:
                    raise ValueError(
                        f"Invalid mode '{mode}'. Use 'plot', 'legend', or 'all'."
                        
                    )
                return

        raise FileNotFoundError(
            f"Could not find {self.kij_filename} for pair {c1}-{c2}. "
            f"Tried folders: {', '.join(self._pair_folder_candidates(c1, c2))}"
        )
        
    def show_pair_plots(self, components, mode="plot"):
        """
        Display all pair plots for a component system in a single figure.

        mode: "plot" - kij_vs_T only (1 row, 3 cols)
            "all"  - kij_vs_T + legend side by side (3 rows, 2 cols)
        """
        pairs = list(combinations(components, 2))
        n_pairs = len(pairs)

        if mode == "plot":

            _, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4))
            axes = np.atleast_1d(axes)

            for ax, (c1, c2) in zip(axes, pairs):
                plot_found = False
                try:
                    for folder in self._pair_folder_candidates(c1, c2):
                        path = (self.root / folder / self.kij_filename).resolve()
                        if path.exists():
                            try:
                                d = json.loads(path.read_text(encoding="utf-8"))
                                img_bytes = base64.b64decode(d["plots"]["kij_vs_T"])
                                img = mpimg.imread(io.BytesIO(img_bytes))
                                ax.imshow(img)
                                ax.axis("off")
                                ax.set_title(f"{c1}-{c2}")
                                plot_found = True
                                break
                            except (KeyError, ValueError, IOError) as e:
                                continue
                except Exception as e:
                    print(f"Warning: Could not load plot for {c1}-{c2}: {e}")

                if not plot_found:
                    ax.axis("off")
                    ax.set_title(f"{c1}-{c2}\nNo data available")
                    print(f"Warning: No data available for pair {c1}-{c2}")

            plt.tight_layout()
            plt.show()

        elif mode == "all":

            _, axes = plt.subplots(n_pairs, 2, figsize=(12, 4 * n_pairs))
            axes = np.atleast_2d(axes)  # Ensure axes is 2D for consistent indexing

            for row, (c1, c2) in enumerate(pairs):
                plot_found = False
                try:
                    for folder in self._pair_folder_candidates(c1, c2):
                        path = (self.root / folder / self.kij_filename).resolve()
                        if path.exists():
                            try:
                                d = json.loads(path.read_text(encoding="utf-8"))
                                # Left: kij_vs_T plot
                                img_bytes = base64.b64decode(d["plots"]["kij_vs_T"])
                                img = mpimg.imread(io.BytesIO(img_bytes))
                                axes[row, 0].imshow(img)
                                axes[row, 0].axis("off")
                                axes[row, 0].set_title(f"{c1}-{c2}")
                                # Right: legend
                                if "legend" in d["plots"]:
                                    img_bytes = base64.b64decode(d["plots"]["legend"])
                                    img = mpimg.imread(io.BytesIO(img_bytes))
                                    axes[row, 1].imshow(img)
                                    axes[row, 1].axis("off")
                                    # axes[row, 1].set_title(f"{c1}-{c2} | legend")
                                else:
                                    axes[row, 1].axis("off")
                                plot_found = True
                                break
                            except (KeyError, ValueError, IOError) as e:
                                continue
                except Exception as e:
                    print(f"Warning: Could not load plot for {c1}-{c2}: {e}")

                if not plot_found:
                    axes[row, 0].axis("off")
                    axes[row, 1].axis("off")
                    axes[row, 0].set_title(f"{c1}-{c2}\nNo data available")
                    print(f"Warning: No data available for pair {c1}-{c2}")

            plt.tight_layout()
            plt.subplots_adjust(wspace=-0.1)
            plt.show()
