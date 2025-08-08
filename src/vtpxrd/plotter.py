# file: plotter.py
# copyright_ArnabDutta

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import MaxNLocator

mpl.rcParams['figure.dpi'] = 200

# Numbered menu of common, perceptually good, and familiar colormaps
AVAILABLE_CMAPS: List[str] = [
    "turbo", "viridis", "plasma", "inferno", "magma", "cividis",
    "twilight", "twilight_shifted", "cubehelix", "Spectral",
    "coolwarm", "seismic", "bwr", "PuOr", "PRGn", "PiYG",
    "RdBu", "RdYlBu", "RdYlGn", "YlGnBu", "YlOrRd",
    "terrain", "ocean", "gist_earth", "gnuplot", "rainbow", "jet"
]

def read_xy_folder(data_folder: str) -> Optional[str]:
    """Merge all *.xy (2 col: 2theta,Intensity) in folder into consolidated_data.csv.
    Why: Provide one tidy table for plotting; skip files without a number in name."""
    xy_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".xy")]
    if not xy_files:
        print("❌ No .xy files found in the current directory.")
        return None

    all_frames: List[pd.DataFrame] = []
    for fname in sorted(xy_files):
        fpath = os.path.join(data_folder, fname)
        try:
            df = pd.read_csv(fpath, sep=r"\s+", header=None, names=["2theta", "Intensity"])
            m = re.search(r"(\d+)", fname)  # first number anywhere in filename
            if not m:
                print(f"⚠️ Skipped {fname}: cannot extract a number from filename.")
                continue
            colname = m.group(1)
            all_frames.append(df.rename(columns={"Intensity": colname})[["2theta", colname]])
            print(f"✔ Processed {fname} → column '{colname}'")
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

    if not all_frames:
        print("❌ No valid data columns created from .xy files.")
        return None

    merged = pd.concat(all_frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]  # keep first '2theta'
    cols_numeric = [c for c in merged.columns if c != "2theta"]
    cols_sorted = sorted(cols_numeric, key=lambda x: int(re.match(r"(\d+)", x).group(1)))
    merged = merged[["2theta"] + cols_sorted]

    out_csv = os.path.join(data_folder, "consolidated_data.csv")
    merged.to_csv(out_csv, index=False)
    print(f"✅ Consolidated data saved → {out_csv}")
    return out_csv

def build_colormap_menu() -> Tuple[str, Optional[float]]:
    """Prompt for colormap; optionally reverse and set gamma (contrast)."""
    print("\n🎨 Colormap options:")
    for i, name in enumerate(AVAILABLE_CMAPS, start=1):
        print(f"  {i:2d}. {name}")
    custom_idx = len(AVAILABLE_CMAPS) + 1
    print(f"  {custom_idx:2d}. Custom (enter your own colors)")
    choice_raw = input(f"Choose 1–{custom_idx}: ").strip()

    try:
        choice = int(choice_raw)
    except ValueError:
        print("⚠️ Invalid choice; defaulting to 'turbo'.")
        return "turbo", None

    if choice == custom_idx:
        return "CUSTOM", None

    if not (1 <= choice <= len(AVAILABLE_CMAPS)):
        print("⚠️ Out-of-range; defaulting to 'turbo'.")
        return "turbo", None

    name = AVAILABLE_CMAPS[choice - 1]
    rev = input("Reverse colormap? (y/N): ").strip().lower().startswith("y")
    if rev:
        name = name + "_r"

    gamma_val: Optional[float] = None
    gamma_s = input("Adjust contrast gamma? (e.g., 0.7 softer, 1 none, 1.5 punchier) [Enter to skip]: ").strip()
    if gamma_s:
        try:
            gamma_val = float(gamma_s)
            if gamma_val <= 0:
                print("⚠️ Gamma must be > 0. Ignoring.")
                gamma_val = None
        except ValueError:
            print("⚠️ Not a number. Ignoring gamma.")
            gamma_val = None

    return name, gamma_val

def build_custom_colormap() -> Tuple[LinearSegmentedColormap, Optional[float]]:
    """Custom colormap from user colors."""
    print("\nEnter colors as comma-separated values. Examples:")
    print("  #000000,#4444ff,#00ffff,#ffffff")
    print("  black,orange,yellow")
    raw = input("Colors: ").strip()
    items = [c.strip() for c in raw.split(",") if c.strip()]
    if len(items) < 2:
        print("⚠️ Need at least two colors. Falling back to 'turbo'.")
        return plt.get_cmap("turbo"), None

    try:
        cmap = LinearSegmentedColormap.from_list("user_cmap", items, N=256)
    except ValueError as e:
        print(f"⚠️ Failed to build custom colormap ({e}). Falling back to 'turbo'.")
        return plt.get_cmap("turbo"), None

    rev = input("Reverse custom colormap? (y/N): ").strip().lower().startswith("y")
    if rev:
        cmap = LinearSegmentedColormap.from_list("user_cmap_r", cmap(np.linspace(1, 0, 256)))

    gamma_val: Optional[float] = None
    gamma_s = input("Adjust contrast gamma? (e.g., 0.7, 1, 1.5) [Enter to skip]: ").strip()
    if gamma_s:
        try:
            gamma_val = float(gamma_s)
            if gamma_val <= 0:
                print("⚠️ Gamma must be > 0. Ignoring.")
                gamma_val = None
        except ValueError:
            print("⚠️ Not a number. Ignoring gamma.")
            gamma_val = None

    return cmap, gamma_val

def choose_source_label() -> str:
    """Prompt for X-ray source label."""
    print("\n🧪 X-ray source:")
    print("  1. CuKα")
    print("  2. AgKα")
    print("  3. MoKα")
    print("  4. Custom")
    choice = (input("Choose 1–4 [default 1]: ").strip() or "1")
    if choice == "1":
        return "CuKα"
    if choice == "2":
        return "AgKα"
    if choice == "3":
        return "MoKα"
    custom = input("Enter custom source label (e.g., CoKα, W Lα): ").strip()
    return custom or "CuKα"

def choose_temperature_unit() -> str:
    """Prompt for temperature unit. Returns 'K' or 'C'."""
    print("\n🌡 Temperature unit:")
    print("  1. T(K)")
    print("  2. T(°C)")
    choice = (input("Choose 1–2 [default 1]: ").strip() or "1")
    return "C" if choice == "2" else "K"

def plot_contour(
    data_file: str,
    cmap_in,
    gamma: Optional[float],
    source_label: str,
    temp_unit: str,
    xlim: Tuple[float, float] = (5, 80)
) -> plt.Figure:
    """Plot contour using chosen colormap; apply optional gamma via PowerNorm.
    Why: PowerNorm(gamma) emphasizes low/high intensities without data edits."""
    data = pd.read_csv(data_file)
    x = data.iloc[:, 0].values

    y_raw: List[float] = []
    for col in data.columns[1:]:
        m = re.search(r"(\d+)", col)
        if m:
            y_raw.append(float(m.group(1)))
        else:
            print(f"Skipping invalid column: {col}")

    if not y_raw:
        raise ValueError("No valid numeric columns found for Y axis.")

    # Convert K to °C for labeling/axis if requested
    y_labels = [t - 273.15 for t in y_raw] if temp_unit == "C" else y_raw

    Z = data.iloc[:, 1:].values
    if Z.shape != (len(y_raw), len(x)):
        Z = Z.T  # align shape

    X, Y = np.meshgrid(x, y_labels)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    norm = PowerNorm(gamma=gamma) if gamma and gamma != 1.0 else None
    cp = ax.contourf(X, Y, Z, levels=600, cmap=cmap_in, norm=norm)

    cbar = fig.colorbar(cp, ax=ax, label="Intensity", fraction=0.05, pad=0.02)
    cbar.ax.tick_params(labelsize=8, width=1.5)
    for label in cbar.ax.get_yticklabels():
        label.set_fontweight("bold")

    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.set_xlabel(fr"$\mathbf{{2\theta(\degree)/\mathbf{{{source_label}}}}}$", fontsize=16, fontweight="bold")
    ax.set_ylabel("T(°C)" if temp_unit == "C" else "T(K)", fontsize=16, fontweight="bold")
    ax.set_xlim(*xlim)
    ax.set_ylim(min(y_labels), max(y_labels))
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.5)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    fig.tight_layout()
    return fig

def main():
    cwd = os.getcwd()
    print(f"📂 Current directory: {cwd}")
    csv_path = read_xy_folder(cwd)
    if not csv_path:
        return

    cmap_name, gamma = build_colormap_menu()
    if cmap_name == "CUSTOM":
        cmap, gamma_custom = build_custom_colormap()
        if gamma is None and gamma_custom is not None:
            gamma = gamma_custom
    else:
        try:
            cmap = plt.get_cmap(cmap_name)
        except ValueError:
            print("⚠️ Unknown colormap. Falling back to 'turbo'.")
            cmap = plt.get_cmap("turbo")

    # Optional 2θ range
    xr_input = input("X-range 2θ limits? (format: min,max) [Enter to keep 5,80]: ").strip()
    xlim = (5.0, 80.0)
    if xr_input:
        try:
            xmin_s, xmax_s = xr_input.split(",")
            xlim = (float(xmin_s), float(xmax_s))
        except Exception:
            print("⚠️ Could not parse x-range. Using default 5–80.")

    # Labels/units
    source_label = choose_source_label()
    temp_unit = choose_temperature_unit()

    fig = plot_contour(csv_path, cmap, gamma, source_label, temp_unit, xlim)

    # Save dialog
    base = input("Output file name without extension (default: heatmap_plot): ").strip() or "heatmap_plot"
    fmt = input("Save format (tif/jpg): ").strip().lower()
    while fmt not in {"tif", "jpg"}:
        fmt = input("Please type 'tif' or 'jpg': ").strip().lower()

    out_path = os.path.join(cwd, f"{base}.{fmt}")
    fig.savefig(out_path, dpi=600)
    plt.show()
    print(f"✅ Plot saved → {out_path}")

if __name__ == "__main__":
    main()
