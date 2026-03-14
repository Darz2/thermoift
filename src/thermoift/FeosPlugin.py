"""
FeosPlugin.py
=============
Utilities for PC-SAFT phase equilibrium and cDFT interfacial tension
calculations for N-component mixtures using the feos framework.

Modules
-------
- Feed generation and normalization
- Parameter building with T-dependent kij
- VLE: bubble/dew curves, critical point, TP flash
- cDFT: planar interface, Interfacial tension, Interfacial thickness, interfacial enrichment
- Results storage and CSV export
- PT and gamma colormap plotting
"""
#!/usr/bin/env python

############# Required Packages ############
import numpy as np, feos, json, matplotlib.pyplot as plt, os, re, pandas as pd, si_units as si
from molmass import Formula
from pathlib import Path
from itertools import combinations
from scipy.interpolate import griddata, PchipInterpolator
from matplotlib.path import Path as MplPath
from scipy.ndimage import binary_erosion

from . import PLOT_SETTINGS as ps
from .BINARY_INTERACTION_PARAMETERS.KIJ import BinaryInteractions as KIJ

_HERE = Path(__file__).parent

############# Helper functions ############

def print_array2d(feeds, decimals=4):
    ncols = feeds.shape[1]

    header = " | ".join([f"{'z'+str(j+1):>10}" for j in range(ncols)])
    print(f"{'i':>3} | {header}")
    print("-" * (15 + 14*ncols))

    for i, row in enumerate(feeds, start=1):
        row_str = " ".join(f"{val:12.4f}" for val in row)
        print(f"{i:3d} | {row_str}")

def latex_formula(formula: str) -> str:
    """
    Convert a chemical formula to LaTeX format.
    Example: 'CO2' -> 'CO$_2$'
             'Ar'  -> 'Ar'
    """
    return re.sub(r"(\d+)", r"$_{\1}$", formula)

def _load_registry(json_path=None):
    
    if json_path is None:
        json_path = _HERE / "parameters.json"
    with open(json_path, "r") as f:
        raw = json.load(f)

    registry        = {}
    kij_labels_map  = {}

    for entry in raw:
        
        name    = entry["identifier"]["name"]
        p       = entry
        
        kwargs  = {
            "molarweight": p["molarweight"],
            "m":           p["m"],
            "sigma":       p["sigma"],
            "epsilon_k":   p["epsilon_k"],
        }
        
        if "mu" in p:
            kwargs["mu"] = p["mu"]
        
        if "association_sites" in p:
            kwargs["association_sites"] = p["association_sites"]

        registry[name] = feos.PureRecord(
            feos.Identifier(
                cas=p["identifier"]["cas"],
                name=name,
                iupac_name=p["identifier"]["iupac_name"],
            ),
            **kwargs,
        )
        kij_labels_map[name] = p["kij_label"]

    return registry, kij_labels_map

# Load registry and labels
_REGISTRY, _KIJ_LABELS = _load_registry()

def components(parameters):
    """
    Extract component names from a feos parameters object.
    Works for any number of components (unary, binary, ternary, etc.)

    Parameters
    ----------
    parameters : feos parameters object

    Returns
    -------
    tuple of str
        Component names extracted from pure_records identifiers.
    """
    labels = []
    for pr in parameters.pure_records:
        id_str = str(pr.identifier)
        match  = re.search(r"name='([^']+)'", id_str)
        if not match:
            match = re.search(r'name=([^,)]+)', id_str)
        name = match.group(1).strip()
        labels.append(_KIJ_LABELS.get(name, name))

    return tuple(labels)

def _kij_labels(component_names: list[str]) -> list[str]:
    """Convert common component names to KIJ labels using the loaded registry."""
    missing = [n for n in component_names if n not in _KIJ_LABELS]
    if missing:
        raise ValueError(f"Unknown component(s): {missing}. Available: {sorted(_KIJ_LABELS.keys())}")
    return [_KIJ_LABELS[n] for n in component_names]

############# Functions ############

def generate_feeds(CO2, n_points):
    """
    Generate feed compositions for a ternary mixture:
    CO2 (C1) (fixed), H2 (C2) (varied), Ar (C3) (remainder).

    Parameters
    ----------
    CO2 : float
        Mole fraction of CO2 (component 1).
    n_points : int, optional
        Number of composition points (default is 21).

    Returns
    -------
    feeds : np.ndarray
        Array of shape (n_points, 3) with columns [CO2, H2, Ar].
    """
    C2_values = np.linspace(0.0, 1.0 - CO2, n_points)
    feeds = np.array([[CO2, C2, 1.0 - CO2 - C2] for C2 in C2_values])
    
    # Remove rows where any component is zero (if desired)
    # feeds = feeds[~np.any(np.isclose(feeds, 0.0), axis=1)]
    
    return feeds

def make_feeds(*compositions):
    """
    Create a feeds array from one or more explicit compositions.

    Parameters
    ----------
    *compositions : array-like
        Each argument is a composition vector [z1, z2, ...].
        All compositions must have the same length.
        Each will be normalized to sum to 1.

    Returns
    -------
    feeds : np.ndarray
        Array of shape (n_compositions, n_components).

    Examples
    --------
    # Single ternary composition
    feeds = make_feeds([0.95, 0.02, 0.03])

    # Multiple compositions
    feeds = make_feeds([0.95, 0.02, 0.03],
                       [0.80, 0.10, 0.10],
                       [0.50, 0.25, 0.25])

    # Binary composition
    feeds = make_feeds([0.95, 0.05])
    """
    feeds = []
    for z in compositions:
        z = np.asarray(z, dtype=float)
        z = z / z.sum()  # normalize
        feeds.append(z)
    return np.array(feeds)

def normalize_z(z):
    """Return z as float array normalized to sum to 1."""
    z = np.asarray(z, dtype=float)
    s = z.sum()
    if s == 0:
        raise ValueError("Composition z sums to 0.")
    return z / s

def compute_feed_moles(z):
    """Convert mole fractions z to mole amounts using n_total."""
    return np.asarray(z, dtype=float) * si.MOL

def value_in(q, unit):
    """Return the numeric value of q in the given unit."""
    if isinstance(q, (int, float, np.floating)):
        return float(q)
    return np.asarray(q / unit, dtype=float)

def molar_density_mol_m3(rho):
    """Convert molar density to mol/m^3."""
    mol_per_m3 = si.MOL / (si.METER**3)
    try:
        return value_in(rho, mol_per_m3)
    except Exception:
        pass
    try:
        kmol_per_m3 = si.KMOL / (si.METER**3)
        return value_in(rho, kmol_per_m3) * 1000.0
    except Exception:
        pass
    try:
        kmol_per_m3 = (1000.0 * si.MOL) / (si.METER**3)
        return value_in(rho, kmol_per_m3) * 1000.0
    except Exception as e:
        raise TypeError(f"Could not interpret density units") from e
    
def density_kg_m3(rho_molar, z, molar_masses, molar_mass_unit="g/mol"):
    """Convert molar density to mass density (kg/m^3)."""
    z = np.asarray(z, dtype=float)
    M = np.asarray(molar_masses, dtype=float)
    
    if molar_mass_unit.lower() == "g/mol":
        M = M / 1000.0
    elif molar_mass_unit.lower() == "kg/mol":
        pass
    else:
        raise ValueError("molar_mass_unit must be 'g/mol' or 'kg/mol'.")
    
    rho_mol_m3 = molar_density_mol_m3(rho_molar)
    M_mix = np.sum(z * M)
    return rho_mol_m3 * M_mix

# Guard to issue concering #254
def reduce_components(z, components, verbose=False):
    """
    Detect zero-fraction components and reduce the system accordingly.

    Parameters
    ----------
    z : array-like
        Mole fractions of the full component set.
    components : list
        Full list of component objects/names.
    verbose : bool
        Print reduction info if True.

    Returns
    -------
    active_z : np.ndarray
        Renormalized mole fractions of active components.
    active_components : list
        Reduced list of components with non-zero composition.
    is_reduced : bool
        True if any component was dropped.
    """
    z_arr              = np.asarray(z, dtype=float)
    mask               = ~np.isclose(z_arr, 0.0)
    active_components  = [c for c, keep in zip(components, mask) if keep]
    active_z           = z_arr[mask]
    active_z           = active_z / active_z.sum()
    is_reduced         = len(active_components) < len(components)

    if is_reduced and verbose:
        n = len(active_components)
        print(f"  --- Reduced to {n}-component system: {active_components}")

    return active_z, active_components, is_reduced

def reduce_kij_map(active_components, kij_map):
    """
    Filter kij_map to only include pairs of active components.

    Parameters
    ----------
    active_components : list of str
        Reduced list of component names with non-zero composition.
    kij_map : dict
        Original kij map keyed by (component_i, component_j) string tuples.

    Returns
    -------
    active_map : dict
        Filtered kij map with only active component pairs.
    """
    active_set = set(active_components)
    active_map = {
        (i, j): v for (i, j), v in kij_map.items()
        if i in active_set and j in active_set
    }
    return active_map

def molar_masses(parameters):
    """
    Extract molar masses from a feos parameters object.

    Parameters
    ----------
    parameters : feos parameters object

    Returns
    -------
    list of float
        Molar masses [g/mol] for each component in parameters.
    """
    return [Formula(c).mass for c in components(parameters)]

########### Parameters ###########

def PARAMETERS(
    component_names: list[str],
    T_K: float | None = None,
    kij_builder=None,
    model_map: dict | None = None,
) -> feos.Parameters:
    """
    Build a feos.Parameters object from inline PC-SAFT parameters,
    optionally loading T-dependent kij(T) from a KIJMatrixBuilder.

    Parameters
    ----------
    component_names : list[str]
        Ordered component names, e.g. ["carbon dioxide", "methane", "argon"].
        Order determines component indices throughout the calculation.
    T_K : float, optional
        Temperature in Kelvin at which to evaluate kij(T).
        Required when kij_builder is provided, ignored otherwise.
    kij_builder : KIJMatrixBuilder, optional
        If provided, loads kij polynomials from KIJ.json files and evaluates
        at T_K. If None, all kij = 0.
    model_map : dict, optional
        Passed to KIJMatrixBuilder.load_pairs().
        e.g. {("CO2", "Ar"): "constant", ("H2", "Ar"): "linear"}

    Returns
    -------
    feos.Parameters

    Raises
    ------
    ValueError
        If unknown component names are provided.
        If kij_builder is provided but T_K is None.

    Examples
    --------
    # Without kij
    params = parameters(["carbon dioxide", "hydrogen", "argon"])

    # With T-dependent kij
    builder = KIJ.KIJMatrixBuilder(root="Kij/")
    params = parameters(
             ["carbon dioxide", "hydrogen", "argon"],
             T_K=300.0,
             kij_builder=builder,
             model_map={("CO2", "Ar"): "constant", ("H2", "Ar"): "linear", ("CO2", "H2"): "linear"})
    """
    # --- Guard ---
    if kij_builder is not None and T_K is None:
        raise ValueError("T_K must be provided when using kij_builder.")

    # --- Validate component names ---
    missing = [n for n in component_names if n not in _REGISTRY]
    if missing:
        raise ValueError(f"Unknown component(s): {missing}. Available: {sorted(_REGISTRY.keys())}")

    records = [_REGISTRY[name] for name in component_names]

    # --- Pure component ---
    if len(records) == 1:
        return feos.Parameters.new_pure(records[0])

    # --- Build binary_records from kij_pairs ---
    binary_records = []

    if kij_builder is not None:
        
        kij_labels      = [_KIJ_LABELS[n] for n in component_names]
        label_to_record = dict(zip(kij_labels, records))
        pair_data       = kij_builder.load_pairs(kij_labels, model_map=model_map)
        _, A, _         = KIJ.KIJMatrixBuilder.build_polynomial_tensor(kij_labels, pair_data)
        K               = KIJ.KIJMatrixBuilder.kij_numeric_matrix(T_K, A)

        for (la, lb) in combinations(kij_labels, 2):
            i       = kij_labels.index(la)
            j       = kij_labels.index(lb)
            binary_records.append(feos.BinaryRecord(
                id1=label_to_record[la].identifier,
                id2=label_to_record[lb].identifier,
                k_ij=float(K[i, j]),
            ))

    return feos.Parameters.from_records(records, binary_records=binary_records)

########### VLE computations ############

def compute_bubble_curve(eos, T_vals, feed, verbose):
    feed = np.array(feed/si.MOL)
    bubble_pressures = []
    T_bubble = []

    for T_K in T_vals:
        T = T_K * si.KELVIN
        try:
            envelope = feos.PhaseEquilibrium.bubble_point(eos,temperature_or_pressure=T,liquid_molefracs=feed)
            P_bubble = envelope.liquid.pressure()

        except Exception as err:
            if verbose:
                print(f"Bubble calculation failed at T = {T_K} K: {err}")
            continue

        bubble_pressures.append(P_bubble / si.BAR)
        T_bubble.append(T_K)

    return T_bubble, bubble_pressures

def compute_dew_curve(eos, T_vals, feed, verbose):
    feed = np.array(feed/si.MOL)
    T_dew = []
    dew_pressures = []
    
    for T_K in T_vals:
        T = T_K * si.KELVIN
        try:
            envelope = feos.PhaseEquilibrium.dew_point(eos,temperature_or_pressure=T,vapor_molefracs=feed)
            P_dew = envelope.vapor.pressure()

        except Exception as err:
            if verbose:
                print(f"Dew calculation failed at T = {T_K} K: {err}")
            break

        dew_pressures.append(P_dew / si.BAR)
        T_dew.append(T_K)

    return T_dew, dew_pressures

def tp_flash(feos, eos, T, P, feed, molar_masses):
    """
    Run TP flash and return (eq, x, y).
    Raises exception if flash fails.
    """
    eq              = feos.PhaseEquilibrium.tp_flash(eos, T, P, feed)
    liq, vap        = eq.liquid, eq.vapor
    x               = np.array(liq.molefracs, dtype=float)
    y               = np.array(vap.molefracs, dtype=float)
    liquid_density  = density_kg_m3(liq.density, x, molar_masses) 
    vapor_density   = density_kg_m3(vap.density, y, molar_masses)
    
    return eq, x, y, liquid_density, vapor_density

############ Critical point computation ###########

def compute_CT(feos, eos, z, T_guess):
    """Compute critical point temperature for composition z."""
    cp = feos.State.critical_point(eos, z, initial_temperature=T_guess*si.KELVIN)
    return cp.temperature, cp.pressure()

############ Planar interfacial tension computation ###########

def build_planar_interface(eq, critical_temperature, n_grid, l_grid):
    """Create a planar interface object."""
    return feos.PlanarInterface.from_tanh(
        vle=eq,
        n_grid=n_grid,
        l_grid=l_grid * si.ANGSTROM,
        critical_temperature=critical_temperature,)

def solve_interface_properties(interface, si):
    """
    Solve planar interface and return:
    (gamma_mN_m, interfacial_thickness_nm, (E1, E2, E3))
    """
    sol = interface.solve()  # solve once

    surface_tension             = sol.surface_tension
    interfacial_thickness       = sol.interfacial_thickness
    enrichment                  = sol.interfacial_enrichment

    gamma_mN_m                  = float(surface_tension * 1e3 / si.NEWTON * si.METER)
    interfacial_thickness_nm    = float(interfacial_thickness() * 1e9 / si.METER)

    return gamma_mN_m, interfacial_thickness_nm, enrichment()

######################## Append and save data #########################

def row_append(
    T_K,
    P_bar,
    z,
    Tc,
    Pc,
    P_bubble,
    P_dew,
    liquid_density,
    vapor_density,
    x,
    y,
    gamma_mN_m,
    interfacial_thickness_nm,
    components,
    enrichment
    ):
    
    """
    Build a results dictionary for one composition case.
    Works for any number of components (binary, ternary, etc.)

    Parameters
    ----------
    T_K : float
        Temperature [K].
    P_bar : float
        Pressure [bar].
    z : array-like
        Overall mole fractions.
    Tc : float
        Critical temperature [K].
    Pc : float
        Critical pressure [bar].
    P_bubble : float
        Bubble point pressure [bar].
    P_dew : float
        Dew point pressure [bar].
    liquid_density : float
        Liquid density [kg/m3].
    vapor_density : float
        Vapor density [kg/m3].
    x : array-like
        Liquid phase mole fractions.
    y : array-like
        Vapor phase mole fractions.
    liquid_density : float
        Liquid density [kg/m3].
    vapor_density : float
        Vapor density [kg/m3].
    critical_temperature : float
        Critical temperature [K * si.KELVIN].
    critical pressure : float
        Critical pressure [bar * si.BAR].
    gamma_mN_m : float
        Surface tension [mN/m].
    interfacial_thickness_nm : float
        Interfacial thickness [nm].
    components : list of str
        Component names (any length).
    enrichment : tuple of float
        Enrichment values, one per component.

    Returns
    -------
    dict
        Results dictionary with keys named by component.
    """
    row = {}

    # Thermodynamic state
    row["temperature"]  = float(T_K)
    row["pressure"]     = float(P_bar)

    # Feed composition
    for i, c in enumerate(components):
        row[f"z_{c}"] = float(z[i])
        
    # Critical points
    row["Tc"] = float(Tc / si.KELVIN)
    row["Pc"] = float(Pc / si.BAR)
    
    # Phase envelope limits
    row["P_bubble"] = float(P_bubble)
    row["P_dew"]    = float(P_dew)

    # Bulk densities
    row["liquid_density"]   = float(liquid_density)
    row["vapor_density"]    = float(vapor_density)

    # Phase compositions
    for i, c in enumerate(components):
        row[f"x_{c}"] = float(x[i])

    for i, c in enumerate(components):
        row[f"y_{c}"] = float(y[i])

    # Interfacial properties
    row["gamma"] = float(gamma_mN_m)
    row["interfacial_thickness"] = float(interfacial_thickness_nm)

    # Interfacial enrichment
    for i, c in enumerate(components):
        row[f"E_{c}"] = float(enrichment[i])

    return row

def save_figure(fig, filename):
    """Save figure to filename at 1200 dpi."""
    fig.savefig(filename, dpi=1200, bbox_inches='tight')
    
def save_plot(fig, filename_base, folder="PLOTS"):
    """
    Save a figure as both PNG and PDF using save_figure.
    """
    os.makedirs(folder, exist_ok=True)

    png_path = os.path.join(folder, f"{filename_base}.png")
    pdf_path = os.path.join(folder, f"{filename_base}.pdf")

    save_figure(fig, png_path)
    save_figure(fig, pdf_path)

def VLE_DFT_summary(VLE_DFT):
    """
    Print a summary of VLE-DFT data structure.
    
    Parameters
    ----------
    VLE_DFT : dict
        Dictionary containing VLE data with structure:
        {feed_key: {
            "interfacial_data": {T_K: [data_dicts, ...]},
            "phase_envelope": {
                "bubble_curve": [...],
                "dew_curve": [...],
                "isothermal_lines": [...]
            }
        }}
    
    Returns
    -------
    dict
        Summary statistics for each feed:
        {feed_key: {
            "n_bubble_points": int,
            "n_dew_points": int,
            "n_isotherms": int,
            "total_interfacial_points": int
        }}
    """
    print("Available feeds:", list(VLE_DFT.keys()))
    
    summary = {}
    
    for feed_key in VLE_DFT:
        n_bubble = len(VLE_DFT[feed_key]["phase_envelope"]["bubble_curve"])
        n_dew = len(VLE_DFT[feed_key]["phase_envelope"]["dew_curve"])
        n_temps = len(VLE_DFT[feed_key]["interfacial_data"])
        
        # Count total interfacial data points
        total_interfacial = sum(
            len(data_list) 
            for data_list in VLE_DFT[feed_key]["interfacial_data"].values()
        )
        
        print(f"{feed_key}: {n_bubble} bubble pts, {n_dew} dew pts, {n_temps} isotherms, {total_interfacial} interfacial pts")
        
        summary[feed_key] = {
            "n_bubble_points": n_bubble,
            "n_dew_points": n_dew,
            "n_isotherms": n_temps,
            "total_interfacial_points": total_interfacial
        }
    
    return summary

def VLE_DFT_to_csv(VLE_DFT, folder="CSV", verbose: bool = False):
    
    """Save VLE-DFT interfacial and phase envelope data to CSV files."""
    os.makedirs(folder, exist_ok=True)
    saved_files = {}
    
    for feed_key in VLE_DFT:
        
        # Save interfacial data
        all_data = []
        for T_K, data_list in VLE_DFT[feed_key]["interfacial_data"].items():
            all_data.extend(data_list)
        
        df = pd.DataFrame(all_data)
        csv_filename = os.path.join(folder, f"{feed_key}_interfacial_results.csv")
        df.to_csv(csv_filename, index=False)
        
        if verbose:
            print(f"Saved {len(df)} rows to {csv_filename}")
        
        # Save phase envelope (isothermal lines only)
        envelope_filename = os.path.join(folder, f"{feed_key}_phase_envelope.csv")
        pd.DataFrame(VLE_DFT[feed_key]["phase_envelope"]["isothermal_lines"]).to_csv(
            envelope_filename, index=False)
        
        saved_files[feed_key] = {
            "interfacial": csv_filename,
            "envelope": envelope_filename}
    
    return saved_files
   
######################## PT PLOT SETTINGS #########################

def _pchip_smooth(T_raw, P_raw, Tc, Pc, n=1000, gap_tol=0.01):
    """
    Fit a shape-preserving PCHIP spline to the raw curve data.
    Appends (Tc, Pc) as a hard anchor only when there is a meaningful gap
    between the last computed point and the critical point (> gap_tol * Tc K).
    If the last point is already within that tolerance, the anchor is skipped
    to avoid near-duplicate T values that would force an unphysical steep segment.
    """
    T = np.asarray(T_raw, dtype=float)
    P = np.asarray(P_raw, dtype=float)
    idx = np.argsort(T)
    T, P = T[idx], P[idx]
    if (Tc - T[-1]) > gap_tol * Tc:
        T = np.append(T, Tc)
        P = np.append(P, Pc)
    T_s = np.linspace(T.min(), T.max(), n)
    return T_s, PchipInterpolator(T, P)(T_s)

def plot_PT(PT_results, feed_key, parameters):
    """
    Plot PT phase diagram (bubble + dew) for a given feed.
    Includes critical point.
    """
    if feed_key not in PT_results:
        raise KeyError(f"{feed_key} not found in PT_results dict.")

    feed_data   = PT_results[feed_key]
    comps       = components(parameters)
    
    # --- Extract curves ---
    T_bub = [pt["T_K"] for pt in feed_data["bubble"]]
    P_bub = [pt["P_bar"] for pt in feed_data["bubble"]]

    T_dew = [pt["T_K"] for pt in feed_data["dew"]]
    P_dew = [pt["P_bar"] for pt in feed_data["dew"]]

    # --- Critical point ---
    Tc = feed_data["TC_K"]
    Pc = feed_data["PC_bar"]
    z  = feed_data["z"]

    fig, ax = ps.plot_init()

    T_bub_s, P_bub_s = _pchip_smooth(T_bub, P_bub, Tc, Pc)
    T_dew_s, P_dew_s = _pchip_smooth(T_dew, P_dew, Tc, Pc)

    ax.plot(T_bub_s, P_bub_s, label="Bubble curve", linestyle="-", linewidth=ps.linewidth, color="blue")
    ax.plot(T_dew_s, P_dew_s, label="Dew curve", linestyle="-", linewidth=ps.linewidth, color="red")
    ax.plot(Tc, Pc, 'o', markersize=4, zorder=4, markerfacecolor="grey", markeredgewidth=1, 
            markeredgecolor="black", label="Critical point")

    ax.set_xlabel(r'$T \; / \; [\mathrm{K}]$', fontsize=ps.label_fontsize)
    ax.set_ylabel(r'$P \; / \; [\mathrm{bar}]$', fontsize=ps.label_fontsize)
    
    ax.set_xlim(left=200)
    
    z_str = ", ".join(f"{latex_formula(c)}={val:g}" for c, val in zip(comps, z))
    ax.set_title(f"{z_str}", fontsize=ps.label_fontsize)
    ax.minorticks_on()

    ps.style_legend(ax, loc='best',ncol=1,borderaxespad=1.0,frame=False)
    ax.grid(False)
    fig.tight_layout()
    
    return fig,ax

def plot_gamma_colormap(PT_results, VLE_DFT, feed_key, parameters):
    """
    Create a colormap plot of interfacial tension in PT space.
    
    Parameters
    ----------
    VLE_DFT : dict
        Full VLE dictionary containing phase envelope and interfacial data:
        {"tmeperature": list, "pressure": list, "gamma": list, "stats": dict}
    feed_key : str
        Feed composition identifier
    parameters : dict
        Plot parameters (colors, fonts, etc.)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the colormap plot
    """
    if feed_key not in PT_results:
        raise KeyError(f"{feed_key} not found in PT_results dict.")
    
    feed_data   = PT_results[feed_key]
    comps       = components(parameters)
        
    # Extract phase envelope
    bubble_data = VLE_DFT[feed_key]["phase_envelope"]["bubble_curve"]
    dew_data    = VLE_DFT[feed_key]["phase_envelope"]["dew_curve"]
    T_bubble    = [pt["T_K"] for pt in bubble_data]
    P_bubble    = [pt["P_bar"] for pt in bubble_data]
    T_dew       = [pt["T_K"] for pt in dew_data]
    P_dew       = [pt["P_bar"] for pt in dew_data]
    
    Tc          = feed_data["TC_K"]
    Pc          = feed_data["PC_bar"] 
    z           = feed_data["z"]
    
    # Extract interfacial tension data
    T_gamma, P_gamma, gamma_data = [], [], []
    
    for T_K, data_list in VLE_DFT[feed_key]["interfacial_data"].items():
        for point in data_list:
            if "gamma" in point and not np.isnan(point["gamma"]):
                T_gamma.append(point["temperature"])
                P_gamma.append(point["pressure"])
                gamma_data.append(point["gamma"])
    
    if len(gamma_data) < 4:
        print(f"Warning: Insufficient data for {feed_key} to create colormap (need at least 4 points).")
        return None
    
    # Create figure
    fig, ax = ps.plot_init()
    
    # Grid generation
    grid_size       = 100
    T_grid          = np.linspace(min(T_gamma), max(T_gamma), grid_size)
    P_grid          = np.linspace(min(P_gamma), max(P_gamma), grid_size )
    T_mesh, P_mesh  = np.meshgrid(T_grid, P_grid)
    
    # Anchor gamma=0 at (Tc, Pc) — exact thermodynamic condition at the critical point
    T_gamma_aug     = T_gamma  + [Tc]
    P_gamma_aug     = P_gamma  + [Pc]
    gamma_aug       = gamma_data + [0.0]

    # Cubic interpolation with critical point anchor
    gamma_mesh      = griddata((T_gamma_aug, P_gamma_aug), gamma_aug,
                           (T_mesh, P_mesh), method='cubic', fill_value=np.nan)

    # Nearest-neighbour fallback for any remaining NaN inside the envelope
    nan_mask = np.isnan(gamma_mesh)
    if nan_mask.any():
        gamma_fill           = griddata((T_gamma_aug, P_gamma_aug), gamma_aug,
                                        (T_mesh, P_mesh), method='nearest')
        gamma_mesh[nan_mask] = gamma_fill[nan_mask]
    
    # Mask regions outside phase envelope
    T_bub_s, P_bub_s = _pchip_smooth(T_bubble, P_bubble, Tc, Pc)
    T_dew_s, P_dew_s = _pchip_smooth(T_dew, P_dew, Tc, Pc)
    envelope_T       = list(T_bub_s) + list(T_dew_s[::-1])
    envelope_P       = list(P_bub_s) + list(P_dew_s[::-1])
    envelope_path    = MplPath(np.column_stack([envelope_T, envelope_P]))
    
    grid_points      = np.column_stack([T_mesh.ravel(), P_mesh.ravel()])
    inside           = envelope_path.contains_points(grid_points).reshape(T_mesh.shape)
    gamma_masked     = np.ma.masked_where(~inside, gamma_mesh)
    gamma_min        = np.nanmin(gamma_masked)
    gamma_max        = np.nanmax(gamma_masked)

    # Isoline levels
    levels_lines     = [2*n + 1 for n in range(10)]
    levels_lines     = [lev for lev in levels_lines if gamma_min <= lev <= gamma_max]

    # Color bands
    levels           = np.linspace(gamma_min, gamma_max, 35)
    contour          = ax.contourf(T_mesh, P_mesh, gamma_masked, levels=levels, 
                                 cmap=plt.cm.Blues, extend='both')
    
    # Contour lines
    inside_eroded    = binary_erosion(inside, iterations=2)
    gamma_masked     = np.ma.masked_where(~inside_eroded, gamma_mesh)
    contour_lines    = ax.contour(T_mesh, P_mesh, gamma_masked, levels=levels_lines,
                            colors='black', linewidths=ps.linewidth/2, alpha=0.6)

    ax.clabel(contour_lines, levels=levels_lines, inline=True,
            fontsize=ps.label_fontsize/2, fmt='%.0f',
            inline_spacing= 15, rightside_up=True)
        
    # Colorbar legend
    cbar = fig.colorbar(contour, ax=ax, pad=0.02,  format='%.0f')
    cbar.set_ticks(levels_lines)
    cbar.set_label(r'$\gamma \; / \; [\mathrm{mN \, m^{-1}}]$', fontsize=ps.label_fontsize)
    cbar.ax.tick_params(labelsize=10) 
    
    # Phase envelope curves
    ax.plot(T_bub_s, P_bub_s, 'b-', linewidth=ps.linewidth, label='Bubble curve', zorder=10)
    ax.plot(T_dew_s, P_dew_s, 'r-', linewidth=ps.linewidth, label='Dew curve', zorder=10)
    ax.plot(Tc, Pc, 'o', markersize=4, zorder=15, markerfacecolor="grey", 
        markeredgewidth=1, markeredgecolor="black", label="Critical point")
    
    ax.set_xlabel(r'$T \; / \; [\mathrm{K}]$', fontsize=ps.label_fontsize)
    ax.set_ylabel(r'$P \; / \; [\mathrm{bar}]$', fontsize=ps.label_fontsize)
    ps.style_legend(ax,fontsize=ps.legend_fontsize, loc='best', framealpha=0)
    
    z_str = ", ".join(f"{latex_formula(c)}={val:g}" for c, val in zip(comps, z))
    ax.set_title(f"{z_str}", fontsize=ps.label_fontsize/2)
    ax.set_xlim(left=200)
    ax.minorticks_on()
    
    plt.tight_layout()
    
    return fig