# VT-PXRD-Plotter

[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

A Python-based tool for visualizing and analyzing variable-temperature powder X-ray diffraction (VT-PXRD) data.  
Generates temperature-dependent 2θ plots, heatmaps, and phase evolution profiles from stacked PXRD datasets.

**Author:** Arnab Dutta

---

## Features
- 📊 Generate high-resolution contour/heatmap plots from multiple `.xy` files  
- 🎨 Flexible colormap selection (built-in or custom)  
- 🔬 Support for multiple X-ray sources (CuKα, AgKα, MoKα, custom)  
- 🌡 Temperature axis in Kelvin or Celsius  
- 💾 Export plots to `.tif` or `.jpg` formats  

---

## Installation

Clone the repo (or download ZIP), then create and activate a virtual environment:

```bash
git clone https://github.com/arnablattice/VT-PXRD-Plotter.git
cd VT-PXRD-Plotter
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
