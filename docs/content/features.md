---
title: Feature comparison
icon: compare_arrows
katex: true
---

This page gives an overview of the features that StORM supports and for which cases you might want to use other oscillation codes.

## Input data

StORM currently only supports the [GSM](https://gyre.readthedocs.io/en/stable/ref-guide/stellar-models/gsm-file-format.html) format. If you have other stellar model formats that you want to use with StORM, see if [tomso](https://tomso.readthedocs.io/en/stable/) is able to import the format you have and convert it to a GSM. While StORM might be able to read more types of models in the future, this is currently a low priority given tools such as tomso exist.

## Included physics

StORM only solves the adiabatic pulsation equations[^1] with the vacuum boundary conditions[^2]. Any form of energy transport through radiation, conduction, ... is ignored.

Rotation is included on various levels. A more detailed explanation can be [found here](rotation). In short, StORM does the following:
- The main solver includes terms for the Coriolis acceleration up to first order
- Toroidal mode coupling in a perturbative way
- Stellar deformation and resulting mode coupling in a perturbative way

The radial order determination is done with [Eckart-Scuflaire-Osaki-Takata](https://ui.adsabs.harvard.edu/abs/2006PASJ...58..893T/abstract) scheme.

## Some features you might want, but are not supported

- Automatic regridding in case the stellar model resolution is low (manually increasing the number of points is supported as part of the `input` command)
- Non-adiabatic calculations
- Full 2D stellar models (e.g. such as done by ACOR and TOP) and pulsation calculations
- Traditional approximation of rotation for high-order g modes with rotation
- Effects of magnetic fields

These features may be supported in the future.

[^1]: Derivations for the adiabatic pulsation equations can be found in [various](https://ui.adsabs.harvard.edu/abs/2010aste.book.....A/abstract) [textbooks](https://ui.adsabs.harvard.edu/abs/1989nos..book.....U/abstract). An overview of the equations can be found in the [GYRE documentation](https://gyre.readthedocs.io/en/stable/ref-guide/osc-equations.html). Note that StORM uses a modified form of the equations to partial include the effect of the Coriolis force in the main solver, similar to [Soufi et al. (1998)](https://ui.adsabs.harvard.edu/abs/1998A&A...334..911S). Validation of the results can be found in `maxima/oscillation-equations.mac`.
[^2]: This assumes that the pressure goes to zero near the surface of the star. Should the density not go to zero, the outer boundary is treated as having a density discontinuity, immediately dropping to zero outside the star.
