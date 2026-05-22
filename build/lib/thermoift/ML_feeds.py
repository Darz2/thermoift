#!/usr/bin/env python

import  itertools
import  json
import  numpy as np
import  pandas as pd
from    pathlib import Path
from    .rng_utils import get_rng

_HERE = Path(__file__).parent
_DEFAULT_TEMPLATES_FILE = _HERE / "IndustrialFeeds.json"


class FeedsBuilder:
    """Build industrial, random, and systematic CO2-rich mixture feeds."""

    def __init__(self, templates_file=None, rng_type="PCG64", seed=None):
        """
        Initialize FeedsBuilder with optional custom templates file and RNG.

        Parameters
        ----------
        templates_file : str or Path, optional
            Path to industrial templates JSON file. Defaults to bundled file.
        rng_type : str, default="PCG64"
            Random number generator type: "PCG64", "MT19937", "SFC64", or "Philox"
        seed : int, optional
            Random seed for reproducibility. If None, uses system entropy.
        """
        self.templates_file = templates_file or _DEFAULT_TEMPLATES_FILE
        self.templates = None
        self.rng_type = rng_type
        self.rng = get_rng(seed, rng_type=rng_type)

    def load_industrial_templates(self, templates_file=None):
        """Load industrial feed templates from JSON file.

        Parameters
        ----------
        templates_file : str or Path, optional
            Path to industrial templates JSON file. If provided, overrides
            the instance's templates_file. If not provided, uses the
            instance's templates_file (which defaults to the bundled file).
        """
        file_to_load = templates_file if templates_file is not None else self.templates_file
        with open(file_to_load) as f:
            self.templates = json.load(f)
        return self.templates

    def random_impurity_split(self, n_impurities, total_impurity, n_samples, alpha=1.0, rng=None):
        """Generate random impurity splits using Dirichlet distribution.

        Parameters
        ----------
        n_impurities : int
            Number of impurity components.
        total_impurity : float
            Total impurity fraction (1 - CO2).
        n_samples : int
            Number of samples to generate.
        alpha : float or array-like, default=1.0
            Dirichlet concentration parameter(s).
            - alpha=1.0: uniform distribution over simplex (default)
            - alpha<1.0: samples concentrate near edges/corners (sparse)
            - alpha>1.0: samples concentrate near center (dense)
            Can be a single value (same for all) or array of length n_impurities.
        rng : int or numpy.random.Generator, optional
            Backward-compatible RNG override. If omitted, uses the builder RNG.
        """
        if n_impurities == 1:
            return np.full((n_samples, 1), total_impurity)

        if np.isscalar(alpha):
            alpha_vec = np.full(n_impurities, alpha)
        else:
            alpha_vec = np.asarray(alpha)
            if len(alpha_vec) != n_impurities:
                raise ValueError(f"alpha length {len(alpha_vec)} != n_impurities {n_impurities}")

        rng_obj = rng if isinstance(rng, np.random.Generator) else (
            get_rng(rng, rng_type=self.rng_type) if rng is not None else self.rng
        )

        samples = rng_obj.dirichlet(alpha=alpha_vec, size=n_samples)
        return samples * total_impurity

    @staticmethod
    def systematic_impurity_split(n_impurities, total_impurity, step=0.01, min_fraction=0.0):
        """Generate systematic impurity splits at fixed step intervals.

        Parameters
        ----------
        n_impurities : int
            Number of impurity components.
        total_impurity : float
            Total impurity fraction (1 - CO2).
        step : float, default=0.01
            Step size for the grid.
        min_fraction : float, default=0.0
            Minimum fraction for each impurity. Compositions where any
            impurity is below this threshold are excluded. Set to e.g. 0.005
            to ensure each impurity has at least 0.5% of total.
        """
        if n_impurities == 1:
            return np.array([[total_impurity]])

        n_units = round(total_impurity / step)
        if not np.isclose(n_units * step, total_impurity, atol=1e-12):
            raise ValueError(
                f"step={step} does not match total_impurity={total_impurity}. "
                f"Choose a step such that total_impurity / step is an integer."
            )

        min_units = int(np.ceil(min_fraction / step)) if min_fraction > 0 else 0

        results = []

        def generate_compositions(parts, total):
            if parts == 1:
                if total >= min_units:
                    yield (total,)
            else:
                for i in range(min_units, total - min_units * (parts - 1) + 1):
                    for rest in generate_compositions(parts - 1, total - i):
                        yield (i,) + rest

        for comp in generate_compositions(n_impurities, n_units):
            results.append(np.array(comp) * step)

        return np.array(results) if results else np.empty((0, n_impurities))

    def industrial_feeds_from_bounds(
        self,
        components,
        co2_levels,
        templates=None,
        n_samples_per_template=20,
        co2_name="CO2",
    ):
        """
        Generate industrial feeds from impurity upper bounds.

        Template format
        ---------------
        {
            "name": "netl_limits",
            "bounds": {
                "O2": 1.0e-5,
                "H2S": 1.0e-4,
                "N2": 0.04,
                "CO": 3.5e-5
            }
        }

        Rules
        -----
        1. CO2 is fixed to one of the requested co2_levels.
        2. Non-CO2 species satisfy 0 <= x_i <= bound_i.
        3. Sum of impurities is exactly 1 - x_CO2.
        4. If sum(bounds) < 1 - x_CO2, that case is infeasible and skipped.
        """
        if templates is None:
            if self.templates is None:
                self.load_industrial_templates()
            templates = self.templates

        rows = []

        co2_levels = tuple(x for x in co2_levels if x >= 0.95)

        for template in templates:
            if "bounds" not in template:
                continue

            name = template["name"]
            bounds_dict = template["bounds"]

            impurity_species = [
                c for c in bounds_dict.keys()
                if c in components and c != co2_name
            ]

            if not impurity_species:
                continue

            upper = np.array([bounds_dict[c] for c in impurity_species], dtype=float)

            for x_co2 in co2_levels:
                target_non_co2 = 1.0 - x_co2

                # Feasibility check
                if upper.sum() < target_non_co2 - 1e-12:
                    continue

                for _ in range(n_samples_per_template):
                    split = np.zeros_like(upper)
                    remaining = target_non_co2

                    # Randomize allocation order for diversity
                    order = self.rng.permutation(len(upper))
                    feasible = True

                    for pos, idx in enumerate(order):
                        remaining_upper = upper[order[pos + 1:]].sum() if pos + 1 < len(order) else 0.0

                        # minimum required here so the remaining variables can still fill the target
                        low = max(0.0, remaining - remaining_upper)

                        # maximum allowed here
                        high = min(upper[idx], remaining)

                        if high < low - 1e-14:
                            feasible = False
                            break

                        if pos == len(order) - 1:
                            value = remaining
                        else:
                            value = self.rng.uniform(low, high)

                        split[idx] = value
                        remaining -= value

                    if not feasible:
                        continue

                    if not np.isclose(split.sum(), target_non_co2, atol=1e-10):
                        continue

                    row = {c: 0.0 for c in components}
                    row["feed_source"] = "industrial_bound"
                    row["template_name"] = name
                    row[co2_name] = x_co2

                    for comp, x in zip(impurity_species, split):
                        row[comp] = x

                    active = [co2_name] + [c for c, x in zip(impurity_species, split) if x > 0.0]
                    row["mixture_size"] = len(active)
                    row["active_components"] = ",".join(active)

                    total = sum(row[c] for c in components)
                    if not np.isclose(total, 1.0, atol=1e-10):
                        continue

                    rows.append(row)

        return pd.DataFrame(rows)
    
    def industrial_feeds_from_ranges(
        self,
        components,
        templates=None,
        n_samples_per_template=20,
        co2_name="CO2",
        trace_value=1.0e-6,
        max_attempts_per_template=1000,
    ):
        """
        Generate industrial feeds from reported composition ranges.

        Template format
        ---------------
        {
            "name": "cortez_pipeline",
            "ranges": {
                "CO2": [0.95, 0.95],
                "H2S": [2.0e-5, 2.0e-5],
                "N2": [0.04, 0.04],
                "CH4": [0.01, 0.05]
            }
        }

        Optional notes block
        --------------------
        {
            "name": "jackson_dome_pipeline",
            "ranges": {"CO2": [0.987, 0.994]},
            "notes": {"H2S": "trace", "N2": "trace", "CH4": "trace"}
        }

        Best-practice rules
        -------------------
        1. Sample non-CO2 species from their reported ranges.
        2. Add trace species with a tiny value.
        3. Compute CO2 as the exact remainder:
               x_CO2 = 1 - sum(non-CO2)
        4. Accept only if computed CO2 lies inside the reported CO2 range.
        5. No renormalization of impurities.
        6. Reject infeasible samples and resample.
        """
        if templates is None:
            if self.templates is None:
                self.load_industrial_templates()
            templates = self.templates

        rows = []

        for template in templates:
            if "ranges" not in template:
                continue

            name = template["name"]
            ranges = template["ranges"]
            notes = template.get("notes", {})

            if co2_name not in ranges:
                continue

            co2_low, co2_high = ranges[co2_name]
            if co2_low > co2_high:
                raise ValueError(f"Invalid CO2 range in template '{name}': {ranges[co2_name]}")

            # Explicit modeled impurities from ranges
            impurity_species = [
                c for c in ranges.keys()
                if c != co2_name and c in components
            ]

            # Trace species from notes
            trace_species = [
                c for c, note in notes.items()
                if c in components and c != co2_name and isinstance(note, str) and note.lower() == "trace"
            ]

            accepted = 0
            attempts = 0

            while accepted < n_samples_per_template and attempts < max_attempts_per_template:
                attempts += 1

                sampled = {}

                # Sample explicit impurity ranges
                bad_range = False
                for comp in impurity_species:
                    low, high = ranges[comp]
                    if low > high:
                        bad_range = True
                        break
                    sampled[comp] = self.rng.uniform(low, high) if low < high else low

                if bad_range:
                    raise ValueError(f"Invalid impurity range in template '{name}'")

                # Add trace species
                for comp in trace_species:
                    if comp not in sampled:
                        sampled[comp] = trace_value

                non_co2_sum = sum(sampled.values())
                x_co2 = 1.0 - non_co2_sum

                # Reject if total impurities exceed 1
                if x_co2 < -1e-12:
                    continue

                # Reject if computed CO2 is outside the reported CO2 range
                if x_co2 < co2_low - 1e-12 or x_co2 > co2_high + 1e-12:
                    continue

                # Enforce minimum CO2 >= 95%
                if x_co2 < 0.95 - 1e-12:
                    continue

                # Build row
                row = {c: 0.0 for c in components}
                row["feed_source"] = "industrial_range"
                row["template_name"] = name
                row[co2_name] = x_co2

                for comp, x in sampled.items():
                    row[comp] = x

                active = [co2_name] + [c for c in components if c != co2_name and row[c] > 0.0]
                row["mixture_size"] = len(active)
                row["active_components"] = ",".join(active)

                total = sum(row[c] for c in components)
                if not np.isclose(total, 1.0, atol=1e-10):
                    continue

                rows.append(row)
                accepted += 1

        return pd.DataFrame(rows)

    def generate_random_feeds(
        self,
        components,
        co2_name="CO2",
        mixture_sizes=(2, 3, 4, 5, 6, 7, 8),
        co2_levels=(0.95, 0.96, 0.97, 0.98, 0.99),
        n_random_samples=100,
        alpha=1.0,
        rng=None,
    ):
        """Generate random feed compositions with specified mixture sizes and CO2 levels.

        Parameters
        ----------
        components : list
            List of component names (must include co2_name).
        co2_name : str, default="CO2"
            Name of the CO2 component.
        mixture_sizes : tuple, default=(2, 3, 4, 5, 6, 7, 8)
            Number of components in each mixture (including CO2).
        co2_levels : tuple, default=(0.95, 0.96, 0.97, 0.98, 0.99)
            CO2 mole fractions to sample.
        n_random_samples : int, default=100
            Number of random samples per combination.
        alpha : float or dict, default=1.0
            Dirichlet concentration parameter.
            - float: same alpha for all impurities (1.0=uniform, <1=sparse, >1=dense)
            - dict: {component_name: alpha_value} for per-component control
              e.g., {"H2": 0.5, "N2": 2.0, "CH4": 1.0}
              Components not in dict default to 1.0
        rng : int or numpy.random.Generator, optional
            Backward-compatible RNG override. Prefer passing seed to FeedsBuilder.
        """
        if co2_name not in components:
            raise ValueError(f"{co2_name} must be in components")

        co2_levels = tuple(x for x in co2_levels if x >= 0.95)

        impurities_pool = [c for c in components if c != co2_name]
        rows = []

        for mixture_size in mixture_sizes:
            n_impurities = mixture_size - 1
            if n_impurities < 1 or n_impurities > len(impurities_pool):
                continue

            for impurity_set in itertools.combinations(impurities_pool, n_impurities):
                # Build alpha vector for this impurity set
                if isinstance(alpha, dict):
                    alpha_vec = np.array([alpha.get(c, 1.0) for c in impurity_set])
                else:
                    alpha_vec = alpha

                for x_co2 in co2_levels:
                    total_impurity = 1.0 - x_co2
                    impurity_splits = self.random_impurity_split(
                        n_impurities=n_impurities,
                        total_impurity=total_impurity,
                        n_samples=n_random_samples,
                        alpha=alpha_vec,
                        rng=rng,
                    )

                    for split in impurity_splits:
                        row = {c: 0.0 for c in components}
                        row["mixture_size"] = mixture_size
                        row["active_components"] = ",".join([co2_name] + list(impurity_set))
                        row["feed_source"] = "Random (Dirichlet)"
                        row["template_name"] = "dirichlet"
                        row[co2_name] = x_co2

                        for comp, x in zip(impurity_set, split):
                            row[comp] = x

                        rows.append(row)

        return pd.DataFrame(rows)

    def generate_systematic_feeds(
        self,
        components,
        co2_name="CO2",
        mixture_sizes=(2, 3, 4),
        co2_levels=(0.95, 0.96, 0.97, 0.98, 0.99),
        step=0.01,
        min_fraction=0.0,
    ):
        """Generate systematic feed compositions at fixed step intervals.

        Parameters
        ----------
        components : list
            List of component names (must include co2_name).
        co2_name : str, default="CO2"
            Name of the CO2 component.
        mixture_sizes : tuple, default=(2, 3, 4)
            Number of components in each mixture (including CO2).
        co2_levels : tuple, default=(0.95, 0.96, 0.97, 0.98, 0.99)
            CO2 mole fractions to sample.
        step : float, default=0.01
            Step size for the grid (1% = 0.01).
        min_fraction : float, default=0.0
            Minimum fraction for each impurity. Set to e.g. 0.005 to ensure
            each active impurity has at least 0.5%. This removes extreme
            compositions where one impurity dominates.
        """
        if co2_name not in components:
            raise ValueError(f"{co2_name} must be in components")

        co2_levels = tuple(x for x in co2_levels if x >= 0.95)

        impurities_pool = [c for c in components if c != co2_name]
        rows = []

        for mixture_size in mixture_sizes:
            n_impurities = mixture_size - 1
            if n_impurities < 1 or n_impurities > len(impurities_pool):
                continue

            for impurity_set in itertools.combinations(impurities_pool, n_impurities):
                for x_co2 in co2_levels:
                    total_impurity = 1.0 - x_co2
                    impurity_splits = self.systematic_impurity_split(
                        n_impurities=n_impurities,
                        total_impurity=total_impurity,
                        step=step,
                        min_fraction=min_fraction,
                    )

                    if len(impurity_splits) == 0:
                        continue

                    for split in impurity_splits:
                        row = {c: 0.0 for c in components}
                        row["mixture_size"] = mixture_size
                        row["active_components"] = ",".join([co2_name] + list(impurity_set))
                        row["feed_source"] = "Systematic"
                        row["template_name"] = "systematic_grid"
                        row[co2_name] = x_co2

                        for comp, x in zip(impurity_set, split):
                            row[comp] = x

                        rows.append(row)

        return pd.DataFrame(rows)

    def combine_feeds_random_systematic_industrial(
        self,
        components,
        n_total_target=10000,
        co2_levels=(0.95, 0.96, 0.97, 0.98, 0.99),
        frac_random=0.70,
        frac_systematic=0.20,
        frac_industrial=0.10,
    ):
        """
        Build a combined dataset with specified fractions.

        Parameters
        ----------
        components : list
            List of component names.
        n_total_target : int, default=10000
            Target total number of feeds.
        co2_levels : tuple, default=(0.95, 0.96, 0.97, 0.98, 0.99)
            CO2 mole fractions to sample.
        frac_random : float, default=0.70
            Fraction of random feeds (0-1).
        frac_systematic : float, default=0.20
            Fraction of systematic feeds (0-1).
        frac_industrial : float, default=0.10
            Fraction of industrial feeds (0-1).

        Note: fractions should sum to 1.0. If they don't, they will be normalized.
        """
        # Normalize fractions
        total_frac = frac_random + frac_systematic + frac_industrial
        if not np.isclose(total_frac, 1.0, atol=1e-6):
            frac_random /= total_frac
            frac_systematic /= total_frac
            frac_industrial /= total_frac

        n_random_target = int(frac_random * n_total_target)
        n_systematic_target = int(frac_systematic * n_total_target)
        n_industrial_target = n_total_target - n_random_target - n_systematic_target

        # --- random pool ---
        df_random_pool = self.generate_random_feeds(
            components=components,
            mixture_sizes=(2, 3, 4, 5, 6, 7, 8),
            co2_levels=co2_levels,
            n_random_samples=20,
        )

        # --- systematic pool ---
        df_systematic_pool = self.generate_systematic_feeds(
            components=components,
            mixture_sizes=(2, 3, 4),
            co2_levels=co2_levels,
            step=0.01,
        )

        # --- industrial templates ---
        if self.templates is None:
            self.load_industrial_templates()

        df_ind_bounds = self.industrial_feeds_from_bounds(
            components=components,
            co2_levels=co2_levels,
            templates=self.templates,
        )
        df_ind_ranges = self.industrial_feeds_from_ranges(
            components=components,
            templates=self.templates,
        )
        df_industrial_pool = pd.concat([df_ind_bounds, df_ind_ranges], ignore_index=True)

        # sample from each pool
        def sample_df(df, n):
            if len(df) <= n:
                return df.copy()
            idx = self.rng.choice(df.index, size=n, replace=False)
            return df.loc[idx].copy()

        df_random = sample_df(df_random_pool, n_random_target)
        df_systematic = sample_df(df_systematic_pool, n_systematic_target)

        # industrial pool may be small, so allow replacement if needed
        if len(df_industrial_pool) == 0:
            df_industrial = df_industrial_pool.copy()
        elif len(df_industrial_pool) < n_industrial_target:
            idx = self.rng.choice(df_industrial_pool.index, size=n_industrial_target, replace=True)
            df_industrial = df_industrial_pool.loc[idx].copy().reset_index(drop=True)
        else:
            df_industrial = sample_df(df_industrial_pool, n_industrial_target)

        df_all = pd.concat([df_random, df_systematic, df_industrial], ignore_index=True)

        ordered_cols = ["feed_source", "template_name", "mixture_size", "active_components"] + components
        df_all = df_all[ordered_cols]

        return df_all


if __name__ == "__main__":

    from .rng_utils import describe_rngs

    components = ["CO2", "N2", "O2", "Ar", "CH4", "H2", "CO", "H2S", "SO2", "NO", "NO2"]

    # Show available RNG options
    describe_rngs()
    print()

    # Example 1: Using default PCG64
    print("Generating feeds with PCG64 (default)...")
    builder_pcg = FeedsBuilder(rng_type="PCG64", seed=42)
    df_pcg = builder_pcg.combine_feeds_random_systematic_industrial(
        components=components,
        n_total_target=5000,
        co2_levels=(0.95, 0.96, 0.97, 0.98, 0.99),
    )
    print(f"Generated {len(df_pcg)} feeds")
    print(df_pcg.head())
    print(df_pcg["feed_source"].value_counts())
    print()

    # Example 2: Using MT19937 (Mersenne Twister)
    print("Generating feeds with MT19937...")
    builder_mt = FeedsBuilder(rng_type="MT19937", seed=42)
    df_mt = builder_mt.combine_feeds_random_systematic_industrial(
        components=components,
        n_total_target=5000,
        co2_levels=(0.95, 0.96, 0.97, 0.98, 0.99),
    )
    print(f"Generated {len(df_mt)} feeds with MT19937")
    df_mt.to_csv("co2_rich_mixed_strategy_feeds_mt19937.csv", index=False)
    df_pcg.to_csv("co2_rich_mixed_strategy_feeds_pcg64.csv", index=False)
