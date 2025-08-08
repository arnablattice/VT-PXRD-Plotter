# file: plotter.py
# copyright_ArnabDutta

import os
import re
import argparse
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, LogNorm
from matplotlib.ticker import MaxNLocator

mpl.rcParams['figure.dpi'] = 200

AVAILABLE_CMAPS: List[str] = [
    "turbo", "viridis", "plasma", "inferno", "magma", "cividis",
    "twilight", "twilight_shifted", "cubehelix", "Spectral",
    "coolwarm", "seismic", "bwr", "PuOr", "PRGn", "PiYG",
    "RdBu", "RdYlBu", "RdYlGn", "YlGnBu", "YlOrRd",
    "terrain", "ocean", "gist_earth", "gnuplot", "rainbow", "jet"
]

# ---------- IO & menus ----------

def read_xy_folder(data_folder: str) -> Optional[str]:
    """Merge all *.xy (2 columns: 2theta, Intensity) into consolidated_data.csv."""
    xy_files = [f for f in os.listdir(data_folder) if f.lower().endswith(".xy")]
    if not xy_files:
        print("❌ No .xy files found in the current directory.")
        return None

    frames: List[pd.DataFrame] = []
    for fname in sorted(xy_files):
        fpath = os.path.join(data_folder, fname)
        try:
            df = pd.read_csv(fpath, sep=r"\s+", header=None, names=["2theta", "Intensity"])
            m = re.search(r"(\d+)", fname)
            if not m:
                print(f"⚠️ Skipped {fname}: cannot extract a number from filename.")
                continue
            colname = m.group(1)
            frames.append(df.rename(columns={"Intensity": colname})[["2theta", colname]])
            print(f"✔ Processed {fname} → column '{colname}'")
        except Exception as e:
            print(f"⚠️ Error processing {fname}: {e}")

    if not frames:
        print("❌ No valid data columns created from .xy files.")
        return None

    merged = pd.concat(frames, axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    cols = [c for c in merged.columns if c != "2theta"]
    cols_sorted = sorted(cols, key=lambda x: int(re.match(r"(\d+)", x).group(1)))
    merged = merged[["2theta"] + cols_sorted]

    out_csv = os.path.join(data_folder, "consolidated_data.csv")
    merged.to_csv(out_csv, index=False)
    print(f"✅ Consolidated data saved → {out_csv}")
    return out_csv


def build_colormap_menu() -> Tuple[str, Optional[float], bool]:
    """Interactive colormap selection. Returns (name_or_CUSTOM, gamma, reversed?)."""
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
        return "turbo", None, False

    if choice == custom_idx:
        return "CUSTOM", None, False

    if not (1 <= choice <= len(AVAILABLE_CMAPS)):
        print("⚠️ Out-of-range; defaulting to 'turbo'.")
        return "turbo", None, False

    name = AVAILABLE_CMAPS[choice - 1]
    rev = input("Reverse colormap? (y/N): ").strip().lower().startswith("y")

    gamma_val: Optional[float] = None
    gamma_s = input("Gamma for PowerNorm (0.7 softer, 1 none, 1.5 punchier) [Enter to skip]: ").strip()
    if gamma_s:
        try:
            gamma_val = float(gamma_s)
            if gamma_val <= 0:
                print("⚠️ Gamma must be > 0. Ignoring.")
                gamma_val = None
        except ValueError:
            print("⚠️ Not a number. Ignoring gamma.")
            gamma_val = None
    return name, gamma_val, rev


def build_custom_colormap() -> Tuple[LinearSegmentedColormap, Optional[float], bool]:
    """Interactive custom colormap from comma-separated colors."""
    print("\nEnter colors as comma-separated values. Examples:")
    print("  #000000,#4444ff,#00ffff,#ffffff")
    print("  black,orange,yellow")
    items = [c.strip() for c in input("Colors: ").strip().split(",") if c.strip()]
    if len(items) < 2:
        print("⚠️ Need at least two colors. Falling back to 'turbo'.")
        return plt.get_cmap("turbo"), None, False
    try:
        cmap = LinearSegmentedColormap.from_list("user_cmap", items, N=256)
    except ValueError as e:
        print(f"⚠️ Failed to build custom colormap ({e}). Falling back to 'turbo'.")
        return plt.get_cmap("turbo"), None, False

    rev = input("Reverse custom colormap? (y/N): ").strip().lower().startswith("y")
    if rev:
        cmap = LinearSegmentedColormap.from_list("user_cmap_r", cmap(np.linspace(1, 0, 256)))

    gamma_val: Optional[float] = None
    gamma_s = input("Gamma for PowerNorm (0.7, 1, 1.5) [Enter to skip]: ").strip()
    if gamma_s:
        try:
            gamma_val = float(gamma_s)
            if gamma_val <= 0:
                print("⚠️ Gamma must be > 0. Ignoring.")
                gamma_val = None
        except ValueError:
            print("⚠️ Not a number. Ignoring gamma.")
            gamma_val = None
    return cmap, gamma_val, rev


def choose_source_label() -> str:
    """Interactive source menu."""
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
    return input("Enter custom source label (e.g., CoKα, W Lα): ").strip() or "CuKα"


def choose_temperature_unit() -> str:
    """Interactive temperature unit selection. Returns 'K' or 'C'."""
    print("\n🌡 Temperature unit:")
    print("  1. T(K)")
    print("  2. T(°C)")
    return "C" if (input("Choose 1–2 [default 1]: ").strip() or "1") == "2" else "K"


# ---------- Plotting ----------

def plot_contour(
    data_file: str,
    cmap_in,
    scale: str = "linear",
    gamma: Optional[float] = None,
    source_label: str = "CuKα",
    temp_unit: str = "K",
    xlim: Tuple[float, float] = (5.0, 80.0),
    levels: int = 600,
) -> plt.Figure:
    """Plot contour with chosen normalization (linear/log/power)."""
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

    y_labels = [t - 273.15 for t in y_raw] if temp_unit == "C" else y_raw

    Z = data.iloc[:, 1:].values
    if Z.shape != (len(y_raw), len(x)):
        Z = Z.T

    # Choose normalization
    norm = None
    if scale == "power":
        if gamma and gamma > 0 and gamma != 1.0:
            norm = PowerNorm(gamma=gamma)
    elif scale == "log":
        # Safe vmin for log scale
        positive = Z[Z > 0]
        vmin = float(np.nanmin(positive)) if positive.size else 1e-9
        vmax = float(np.nanmax(Z)) if Z.size else 1.0
        norm = LogNorm(vmin=max(vmin, 1e-9), vmax=max(vmax, vmin * 10))

    X, Y = np.meshgrid(x, y_labels)

    fig = plt.figure(figsize=(8, 6))
    ax = plt.gca()

    cp = ax.contourf(X, Y, Z, levels=levels, cmap=cmap_in, norm=norm)

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


# ---------- CLI ----------

def parse_args() -> argparse.Namespace:
    """Parse CLI flags for non-interactive usage."""
    p = argparse.ArgumentParser(prog="vtpxrd", description="VT-PXRD heatmap/contour plotter")
    p.add_argument("--non-interactive", action="store_true", help="Run without prompts; use flags only.")
    p.add_argument("--cmap", type=str, help="Matplotlib colormap name (e.g., turbo, viridis).")
    p.add_argument("--reverse", action="store_true", help="Reverse colormap.")
    p.add_argument("--custom-colors", type=str, help="Comma-separated colors to build a custom cmap.")
    p.add_argument("--gamma", type=float, help="Gamma for PowerNorm when --scale=power.")
    p.add_argument("--scale", choices=["linear", "power", "log"], default="linear", help="Intensity scale.")
    p.add_argument("--source", choices=["CuKα", "AgKα", "MoKα", "custom"], help="X-ray source label.")
    p.add_argument("--source-text", type=str, help="Custom source text when --source=custom.")
    p.add_argument("--temp-unit", choices=["K", "C"], help="Temperature unit for Y axis.")
    p.add_argument("--xlim", type=str, help='2θ range "min,max" (e.g., 5,80).')
    p.add_argument("--levels", type=int, default=600, help="Number of contour levels.")
    p.add_argument("--output", type=str, default="heatmap_plot", help="Output filename without extension.")
    p.add_argument("--format", choices=["tif", "jpg"], help="Output format.")
    p.add_argument("--dpi", type=int, default=600, help="Saved figure DPI.")
    return p.parse_args()


def resolve_cmap_from_flags(args: argparse.Namespace):
    """Build colormap from flags or fall back to interactive menus."""
    # Custom colors via flags
    if args.custom_colors:
        items = [c.strip() for c in args.custom_colors.split(",") if c.strip()]
        if len(items) >= 2:
            try:
                cmap = LinearSegmentedColormap.from_list("user_cmap", items, N=256)
            except ValueError:
                cmap = plt.get_cmap("turbo")
        else:
            cmap = plt.get_cmap("turbo")
        if args.reverse:
            cmap = LinearSegmentedColormap.from_list("user_cmap_r", cmap(np.linspace(1, 0, 256)))
        return cmap, args.gamma, args.scale

    # Named cmap via flags
    if args.cmap:
        name = args.cmap + ("_r" if args.reverse else "")
        try:
            return plt.get_cmap(name), args.gamma, args.scale
        except ValueError:
            print("⚠️ Unknown colormap; falling back to 'turbo'.")
            return plt.get_cmap("turbo"), args.gamma, args.scale

    # Interactive path
    name_or_custom, gamma, rev = build_colormap_menu()
    if name_or_custom == "CUSTOM":
        cmap, gamma2, _rev2 = build_custom_colormap()
        gamma = gamma if gamma is not None else gamma2
        return cmap, gamma, "power" if (gamma and gamma != 1.0) else "linear"
    else:
        try:
            cmap = plt.get_cmap(name_or_custom + ("_r" if rev else ""))
        except ValueError:
            cmap = plt.get_cmap("turbo")
        # Ask if they want log scale?
        scale_q = input("Intensity scale? [1] linear, [2] log, [3] power (gamma) : ").strip() or "1"
        scale = {"1": "linear", "2": "log", "3": "power"}.get(scale_q, "linear")
        return cmap, gamma, scale


def main():
    """CLI entry point."""
    args = parse_args()
    cwd = os.getcwd()
    print(f"📂 Current directory: {cwd}")
    csv_path = read_xy_folder(cwd)
    if not csv_path:
        return

    # Colormap / scale
    if args.non_interactive:
        cmap, gamma, scale = resolve_cmap_from_flags(args)
    else:
        cmap, gamma, scale = resolve_cmap_from_flags(args)

    # X-range
    if args.xlim:
        try:
            xmin_s, xmax_s = args.xlim.split(",")
            xlim = (float(xmin_s), float(xmax_s))
        except Exception:
            print("⚠️ Could not parse --xlim; using default 5–80.")
            xlim = (5.0, 80.0)
    elif args.non_interactive:
        xlim = (5.0, 80.0)
    else:
        xr_input = input("X-range 2θ limits? (min,max) [Enter for 5,80]: ").strip()
        if xr_input:
            try:
                a, b = xr_input.split(",")
                xlim = (float(a), float(b))
            except Exception:
                print("⚠️ Could not parse x-range. Using default 5–80.")
                xlim = (5.0, 80.0)
        else:
            xlim = (5.0, 80.0)

    # Source
    if args.source:
        if args.source == "custom":
            source_label = args.source_text or "CuKα"
        else:
            source_label = args.source
    elif args.non_interactive:
        source_label = "CuKα"
    else:
        source_label = choose_source_label()

    # Temp unit
    if args.temp_unit:
        temp_unit = args.temp_unit
    elif args.non_interactive:
        temp_unit = "K"
    else:
        temp_unit = choose_temperature_unit()

    # Plot
    fig = plot_contour(
        csv_path,
        cmap_in=cmap,
        scale=scale,
        gamma=args.gamma,
        source_label=source_label,
        temp_unit=temp_unit,
        xlim=xlim,
        levels=args.levels,
    )

    # Save
    if args.format:
        fmt = args.format
    elif args.non_interactive:
        fmt = "tif"
    else:
        fmt = input("Save format (tif/jpg): ").strip().lower()
        while fmt not in {"tif", "jpg"}:
            fmt = input("Please type 'tif' or 'jpg': ").strip().lower()

    out_base = args.output if args.output else "heatmap_plot"
    out_path = os.path.join(cwd, f"{out_base}.{fmt}")
    fig.savefig(out_path, dpi=args.dpi)
    plt.show()
    print(f"✅ Plot saved → {out_path}")


if __name__ == "__main__":
    main()
