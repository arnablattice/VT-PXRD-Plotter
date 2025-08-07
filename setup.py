import setuptools
import re

# Versioning
VERSIONFILE = "vtpxrd/__init__.py"  # Ensure your main package folder is named 'vtpxrd'
getversion = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", open(VERSIONFILE, "rt").read(), re.M)
if getversion:
    new_version = getversion.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Setup
setuptools.setup(
    name="vtpxrd",
    version=new_version,
    author="Arnab Dutta",
    author_email="your_email@example.com",  # Replace with your real email
    description="VT-PXRD-Plotter: A tool for plotting and analyzing variable-temperature PXRD data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arnablattice/VT-PXRD-Plotter",
    download_url=f"https://github.com/arnablattice/VT-PXRD-Plotter/archive/refs/tags/v{new_version}.tar.gz",
    packages=setuptools.find_packages(),  # Automatically finds all Python packages
    include_package_data=True,
    license_files=["LICENSE"],
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib'
    ],
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Visualization",
    ],
)
