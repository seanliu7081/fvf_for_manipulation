# EquiHarmony

# Overview
EquiHarmony implements differentiable harmonic transforms for the coordinate systems which express symmetry for 2D and 3D geometries.
While there are a number of existing harmonics implementations, they are tailored to the tasks that they solve, e.g. PDEs, and therefore 
do not integrate well with the widely used equivariant neural networks (e.g. *e3nn* or *escnn*).
To address this shortcoming, EquiHarmony provides differentiable implementations of the various types of harmonics commonly usedin SO(2) 
and SO(3) spaces.
EquiHarmony uses PyTroch primitives to implement these operations, making it fully differentiable.
The design goals of this package were two fold: (1) allow for the efficient integration of the harmonics with the output of *e3nn* or *escnn*
and (2) enable the calculation of the harmonics on both a pre-defined grid of coordinates and at specific coordinates.

# Installation
```
git clone git@github.com:ColinKohler/EquiHarmony.git
cd EquiHarmony 
pip install -e .
```

# Getting Started
EquiHarmony currently implements the following harmonic transforms:
- Circular Harmonics
- Polar Harmonics
- Cylindrical Harmonics
- Spherical Harmonics
- SO(3) Harmonics

Detailed usage of each of these harmonics is provided in a series of notebooks:
- [Circular Harmonics](https://github.com/ColinKohler/EquiHarmony/blob/main/notebooks/CircularHarmonics.ipynb)
- [Polar Harmonics](https://github.com/ColinKohler/EquiHarmony/blob/main/notebooks/PolarHarmonics.ipynb)
- [Cylindrical Harmonics](https://github.com/ColinKohler/EquiHarmony/blob/main/notebooks/CylindricalHarmonics.ipynb)
- [Spherical Harmonics](https://github.com/ColinKohler/EquiHarmony/blob/main/notebooks/SphericalHarmonics.ipynb)
