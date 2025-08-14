---
title: Feature comparison
icon: compare_arrows
katex: true
---

## Input data

- [GSM model](https://gyre.readthedocs.io/en/stable/ref-guide/stellar-models/gsm-file-format.html)

## Direct solver

- Adiabatic pulsation equations
- Coriolis force projected on a single spherical harmonic
- Radial order identification using the Eckart-Scuflaire-Osaki-Takata scheme

## Perturbative

- Chandrasekhar-Milne expansion from a one-dimensional stellar model, up to $P_2$ Legendre polynomial
- Coupling between different degrees of modes through the 
- Toroidal backcoupling through the Coriolis force 

## Not supported

- Automatic regridding (requires improvements to the interpolation methods)
- Non-adiabatic calculations
- Full 2D stellar models
- Traditional approximation of rotation
