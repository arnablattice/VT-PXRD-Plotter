# setup.py
# Copyright (c) 2025 ArnabDutta

from pathlib import Path
from setuptools import setup, find_packages

README = (Path(__file__).parent / "README.md").read_text(encoding="utf-8") if (Path(__file__).parent / "README.md").exists() else ""

setup(
    name="vt-pxrd-plotter",
    version="1.0.0",
    description="Interactive VT-PXRD heatmap/contour plotter for .xy diffraction files (current directory).",
    long_description=README,
    long_description_content_type="text/markdown",
    author="ArnabDutta",
    license="MIT",
    keywords=["XRD", "PXRD", "diffraction", "materials", "plotting"],
    python_requires=">=3.8",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    include_package_data=True,
    install_requires=[
        "numpy>=1.22",
        "pandas>=1.5",
        "matplotlib>=3.6",
    ],
    entry_points={
        "console_scripts": [
            "vt-pxrd-plotter=vt_pxrd_plotter.vt_pxrd_plotter:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Visualization",
        "Intended Audience :: Science/Research",
    ],
    project_urls={
        "Homepage": "https://github.com/your-org-or-user/vt-pxrd-plotter",
        "Issues": "https://github.com/your-org-or-user/vt-pxrd-plotter/issues",
    },
)
