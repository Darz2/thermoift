"""
Random Number Generator utilities for flexible RNG selection across the package.

Provides a single interface to select and instantiate various numpy random generators.
"""

import numpy as np


_RNG_REGISTRY = {
    "PCG64": np.random.PCG64,
    "MT19937": np.random.MT19937,
    "SFC64": np.random.SFC64,
    "Philox": np.random.Philox,
}


def get_rng(seed=None, rng_type="PCG64"):
    """
    Create a Generator instance with specified RNG bit generator.

    Parameters
    ----------
    seed : int, optional
        Seed for the random number generator. If None, uses entropy from OS.
    rng_type : str, default="PCG64"
        Type of random number generator to use.
        Options: "PCG64", "MT19937", "SFC64", "Philox"

    Returns
    -------
    np.random.Generator
        A numpy Generator with the specified bit generator.

    Raises
    ------
    ValueError
        If rng_type is not recognized.

    Examples
    --------
    rng = get_rng(seed=42, rng_type="MT19937")
    rng.normal(0, 1, 10)

    rng = get_rng(seed=123, rng_type="PCG64")
    rng.dirichlet(alpha=np.ones(3), size=5)
    """
    if rng_type not in _RNG_REGISTRY:
        available = ", ".join(_RNG_REGISTRY.keys())
        raise ValueError(
            f"Unknown rng_type='{rng_type}'. Available options: {available}"
        )

    bit_generator_class = _RNG_REGISTRY[rng_type]
    bit_gen = bit_generator_class(seed)
    return np.random.Generator(bit_gen)


def describe_rngs():
    """Print a description of available RNG types."""
    descriptions = {
        "PCG64": "Modern, fast, good statistical properties (default)",
        "MT19937": "Classic Mersenne Twister, widely used, well-tested",
        "SFC64": "Fast, space-efficient Squares Congruential Generator",
        "Philox": "Parallel-safe generator, good for concurrent use",
    }
    print("Available Random Number Generators:")
    for name, desc in descriptions.items():
        print(f"  {name:10} - {desc}")
