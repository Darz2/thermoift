#!/usr/bin/env python

############# Required Packages ############
import matplotlib.pyplot as plt, scienceplots, matplotlib.colors as mcolors, os
import matplotlib as mpl
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import re

############# Set LaTeX for text rendering ############
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{xcolor}'

############# PLOT SETTINGS ############
plot_size           = (4, 3)
colors              = ['#e41a1c', '#008000', '#377eb8', '#ff7f00', '#984ea3', '#a65628', '#f781bf']
markers             = ['o', 's', '^', 'D', 'h', 'v', 'p', '*', 'X', '<', '>', '8', 'P', '|', '_']

color_10            = [
    '#008000',  # green
    '#ff7f00',  # orange
    '#984ea3',  # purple
    '#a65628',  # brown
    '#f781bf',  # pink
    "#05545a",  # yellow
    '#4daf4a',  # light green
    '#ffb300',  # gold
    '#1f78b4',  # teal-ish (not blue)
    '#b15928',  # dark brown
    '#999999',  # gray
    "#570F0F",  # Dark Brown
    '#377eb8',  # blue
    '#dede00',  # light yellow
    '#a6cee3',  # light blue
    '#fdbf6f',  # light orange
    '#cab2d6',  # light purple
    '#ffff99',  # light greenish-yellow,
]

color_8 = [
    "#e41a1d",
    "#0b57f9",
    "#03fdf3",
    '#984ea3',
    '#377eb8'
]

dual_colors = [
    ("#06fb0b", "#008000"),   # green (face, edge)
    ("#56abff", "#0000FF"),   # blue  (face, edge)
    ("#FF5656", "#FF0000"),   # red   (face, edge)
    ("#F68FC2", "#FF1493"),   # pink  (face, edge)
    ("#A6CEE3", "#1F78B4"),   # light blue (face, edge)
]

graphic_font        = 'Arial'
math_font           = 'dejavuserif'  # ['dejavusans', 'dejavuserif', 'cm', 'stix', 'stixsans', 'custom']
spine_width         = 1.5
markersize          = 4
capsize             = 3
markeredgewidth     = 0.75
legend_linewidth    = 1
linewidth           = 1
tick_width          = 0.75
tick_length         = 4
minor_tick_width    = 0.5
minor_tick_length   = 2
tick_labelsize      = 10
legend_fontsize     = 8
title_fontsize      = 12
legend_boxwidth     = 0.75
label_fontsize      = 14
borderaxespad       = 0.6
alpha               = 0.5
resolution_value    = 1200
map                = "coolwarm"  # colormap for scatter plots (e.g., 'viridis', 'plasma', 'coolwarm', etc.)

############# FUNCTION TO PROCESS COLORS ############
def face_colors(colors, alpha):
    rgba_colors = [mcolors.to_rgba(c) for c in colors]
    
    return [(rgba[0], rgba[1], rgba[2], alpha) for rgba in rgba_colors]

face_colors = face_colors(colors, alpha)

####################### PLOT FUNCTIONS ##############

def plot_init(w=None, h=None):
    
    size = (w, h) if (w is not None and h is not None) else plot_size

    with plt.style.context(['ieee']):
        
        plt.rcParams['font.family']      = graphic_font
        plt.rcParams['mathtext.fontset'] = math_font
        plt.rcParams['text.usetex']      = True

        fig, ax = plt.subplots(figsize=size)

        for spine in ax.spines.values():
            spine.set_linewidth(spine_width)

        ax.tick_params(axis='both', which='major', direction='in', width=tick_width, length=tick_length,
                       labelsize=tick_labelsize, bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', width=minor_tick_width, length=minor_tick_length,
                       bottom=True, top=True, left=True, right=True)

        return fig, ax


def plot_init_multi(nrows=1, ncols=1, w=None, h=None):

    size = (w, h) if (w is not None and h is not None) else plot_size

    with plt.style.context(["ieee"]):
        plt.rcParams["font.family"]      = graphic_font
        plt.rcParams["mathtext.fontset"] = math_font
        plt.rcParams["text.usetex"]      = True

        fig, axes = plt.subplots(nrows, ncols, figsize=size)  # ← only once, uses size

        for ax in np.array(axes).flatten():
            for spine in ax.spines.values():
                spine.set_linewidth(spine_width)
            ax.tick_params(axis="both", which="major", direction="in",
                           width=tick_width, length=tick_length,
                           labelsize=tick_labelsize,
                           bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis="both", which="minor", direction="in",
                           width=minor_tick_width, length=minor_tick_length,
                           bottom=True, top=True, left=True, right=True)

        return fig, axes
    
def style_legend(
    ax,
    loc="upper right",
    ncol=1,
    edgecolor="black",
    frame=True,
    fontsize=None,
    boxwidth=None,
    **kwargs,):
    
    if fontsize is None:
        fontsize = legend_fontsize
    if boxwidth is None:
        boxwidth = legend_boxwidth

    # --- Harden types that must be ints ---
    try:
        ncol = int(ncol)
    except Exception:
        ncol = 1
    # If provided via kwargs, coerce too
    if "ncol" in kwargs:
        try:
            kwargs["ncol"] = int(kwargs["ncol"])
        except Exception:
            kwargs["ncol"] = ncol

    # scatterpoints must be an int
    sp = kwargs.pop("scatterpoints", 1)  # default 1 marker per entry
    try:
        sp = int(sp)
    except Exception:
        sp = 1

    defaults = dict(
        handletextpad=0.3,   # Horizontal space between the legend marker/line and its label text
        labelspacing=0.5,    # Vertical space between legend entries (line-to-line spacing)
        borderpad=0.5,       # Padding inside the legend box (distance between text and box edge)
        borderaxespad=0.5,   # Distance between the legend box and the axes (outside gap)
        columnspacing=0.6,   # Horizontal space between columns if legend has multiple columns
        handlelength=1.5,    # Length of the legend line handles (for Line2D entries)
        markerscale=0.9,     # Scale factor for legend markers relative to original plot markers
        scatterpoints=sp,    # Number of points to show in legend for scatter plots (must be int)
        fontsize=fontsize,   # Font size of the legend text
        loc=loc,             # Location of the legend (e.g., 'upper right', 'lower left', etc.)
        ncol=ncol,           # Number of columns in the legend (must be int)
    )
        
    defaults.update(kwargs)

    legend = ax.legend(**defaults)

    if frame:
        legend.get_frame().set_linewidth(boxwidth)
        legend.get_frame().set_edgecolor(edgecolor)
    else:
        legend.get_frame().set_visible(False)

    return legend

def apply_axis_style(ax):
    """Apply the same styling as plot_init() to an existing axis."""
    # Set spine widths
    for spine in ax.spines.values():
        spine.set_linewidth(spine_width)

    ax.tick_params(
        axis='both', which='major', direction='in',
        width=tick_width, length=tick_length,
        labelsize=tick_labelsize,
        bottom=True, top=True, left=True, right=True
    )
    ax.tick_params(
        axis='both', which='minor', direction='in',
        width=minor_tick_width, length=minor_tick_length,
        bottom=True, top=True, left=True, right=True
    )

def style_colorbar(cbar):
    cbar.ax.tick_params(
        width=tick_width, length=tick_length,
        labelsize=tick_labelsize
    )
    try:
        cbar.outline.set_linewidth(spine_width)
    except Exception:
        pass
       
def save_figure(fig, filename):
    """Saves the figure with the predefined resolution."""
    output_dir = os.getcwd()
    file_path = os.path.join(output_dir, filename)
    fig.savefig(file_path, dpi=resolution_value, bbox_inches='tight')
    fig.savefig(fr"{filename}", dpi=resolution_value, bbox_inches='tight')
    

def save_plot(fig, filename_base, folder="PLOTS"):
    """
    Save a figure as both PNG and PDF using save_figure.
    """
    os.makedirs(folder, exist_ok=True)

    png_path = os.path.join(folder, f"{filename_base}.png")
    pdf_path = os.path.join(folder, f"{filename_base}.pdf")

    save_figure(fig, png_path)
    save_figure(fig, pdf_path)


############# COMPONENT LATEX MAP ############

component_map = {
    "CO2": r"\mathrm{CO_2}",
    "Carbon dioxide": r"\mathrm{CO_2}",

    "H2": r"\mathrm{H_2}",
    "Hydrogen": r"\mathrm{H_2}",

    "Ar": r"\mathrm{Ar}",
    "Argon": r"\mathrm{Ar}",

    "N2": r"\mathrm{N_2}",
    "Nitrogen": r"\mathrm{N_2}",

    "CH4": r"\mathrm{CH_4}",
    "Methane": r"\mathrm{CH_4}",

    "O2": r"\mathrm{O_2}",
    "Oxygen": r"\mathrm{O_2}",

    "H2O": r"\mathrm{H_2O}",
    "Water": r"\mathrm{H_2O}",

    "CO": r"\mathrm{CO}",
    "Carbon monoxide": r"\mathrm{CO}",

    "NO2": r"\mathrm{NO_2}",
    "Nitrogen dioxide": r"\mathrm{NO_2}",

    "NO": r"\mathrm{NO}",
    "Nitric oxide": r"\mathrm{NO}",

    "SO2": r"\mathrm{SO_2}",
    "Sulfur dioxide": r"\mathrm{SO_2}",

    "H2S": r"\mathrm{H_2S}",
    "Hydrogen sulfide": r"\mathrm{H_2S}",

    "C3H8": r"\mathrm{C_3H_8}",
    "Propane": r"\mathrm{C_3H_8}",

    "C2H6": r"\mathrm{C_2H_6}",
    "Ethane": r"\mathrm{C_2H_6}",
}

def component_to_latex(name):
    name = name.strip()
    return component_map.get(name, rf"\mathrm{{{name}}}")

def format_mixture_latex(mixture):
    """
    Convert mixture string like:
        'CO2;Methane'
        'CO2--Methane'
        'CO2 - Methane'
    into a LaTeX-safe mixture string.
    """
    mixture = mixture.replace("—", "--").replace("–", "--")
    mixture = mixture.replace(";", "--")

    components = [c.strip() for c in mixture.split("--") if c.strip()]
    latex_components = [component_to_latex(c) for c in components]

    return r"$" + r"-".join(latex_components) + r"$"

def format_z_latex(mixture, z):
    """
    Format composition as:
        z_{CO2} = 0.9500, z_{CH4} = 0.0500
    with proper chemical LaTeX formatting.
    """
    mixture = mixture.replace("—", "--").replace("–", "--")
    mixture = mixture.replace(";", "--")

    components = [c.strip() for c in mixture.split("--") if c.strip()]

    parts = []
    for comp, zi in zip(components, z):
        comp_ltx = component_to_latex(comp)
        parts.append(rf"z_{{{comp_ltx}}} = {zi:.4f}")

    return r"$" + r",\ ".join(parts) + r"$"

label_map = {
    "temperature": r"$T$ / [K]",
    "pressure": r"$P$ / [bar]",
    "Tc": r"$T_{\mathrm{c}}$ / [K]",
    "Pc": r"$P_{\mathrm{c}}$ / [bar]",
    "P_bubble": r"$P_{\mathrm{bubble}}$ / [bar]",
    "P_dew": r"$P_{\mathrm{dew}}$ / [bar]",
    "liquid_density": r"$\rho_{\mathrm{liq}}$ / [kg m$^{-3}$]",
    "vapor_density": r"$\rho_{\mathrm{vap}}$ / [kg m$^{-3}$]",
    "gamma": r"$\gamma$ / [mN m$^{-1}$]",
    "interfacial_thickness": r"$L_{90-10}$ / [nm]",
}

for comp, comp_tex in component_map.items():
    label_map[f"z_{comp}".lower()] = rf"$z_{{{comp_tex}}}$ / [mol mol$^{{-1}}$]"
    label_map[f"x_{comp}".lower()] = rf"$x_{{{comp_tex}}}$ / [mol mol$^{{-1}}$]"
    label_map[f"y_{comp}".lower()] = rf"$y_{{{comp_tex}}}$ / [mol mol$^{{-1}}$]"
    label_map[f"E_{comp}".lower()] = rf"$E_{{{comp_tex}}}$ / [-]"

# Lowercase
label_map = {k.lower(): v for k, v in label_map.items()}

def label_to_symbol(label_str):
    """
    Converts a label_map value to a clean sympy symbol string.
    e.g. r"$T_{\mathrm{c}}$ / [K]"  -  r"T_{\mathrm{c}}"
    """
    match = re.match(r"\$(.*?)\$", label_str)
    if match:
        return match.group(1)
    return label_str

symbol_map = {k: label_to_symbol(v) for k, v in label_map.items()}

#################### END OF CODE #####################