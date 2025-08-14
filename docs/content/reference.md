---
title: CLI reference
icon: terminal
---

This document contains the help content for the `storm` command-line program.

## `input`

Load a stellar model

**Usage:** `input <FILE>`

###### **Arguments:**

* `<FILE>` — Location of the stellar model. The stellar model should be an HDF5 GYRE model file



## `set-rotation-overlay`

Replace the rotation profile of a model

**Usage:** `set-rotation-overlay <FILE>`

###### **Arguments:**

* `<FILE>` — HDF5 file containing the rotation profile. The structure should be the same as the normal input model



## `set-rotation-constant`

Set the rotation profile to a constant

**Usage:** `set-rotation-constant [OPTIONS] <VALUE>`

###### **Arguments:**

* `<VALUE>` — Angular rotation frequency

###### **Options:**

* `--frequency-units <FREQUENCY_UNITS>` — Units of value

  Default value: `dynamical`

  Possible values:
  - `dynamical`:
    Dynamical frequency of the star [sqrt(GM/R^3)]
  - `hertz`:
    Hertz [1/s]
  - `cycles-per-day`:
    Cycles per day [1/d]




## `scan`

Perform a frequency scan

**Usage:** `scan [OPTIONS] <ELL> <M> <LOWER> <UPPER> <STEPS>`

###### **Arguments:**

* `<ELL>` — Spherical degree
* `<M>` — Azimuthal order
* `<LOWER>` — Lower frequency of the scan range
* `<UPPER>` — Upper frequency of the scan range
* `<STEPS>` — Number of scanning steps

###### **Options:**

* `--inverse` — Whether to do steps between lower and upper linear in period (inverse) or in frequency
* `--precision <PRECISION>` — Relative precision required.

   Due to the bracketing method, the actual precision of the result can be a couple of orders of magnitude better. Unless comparing different oscillation codes or methods of computation, a reasonable precision is 1e-8, which is the default.

  Default value: `1e-8`
* `--difference-scheme <DIFFERENCE_SCHEME>` — Difference scheme

  Default value: `magnus2`

  Possible values:
  - `colloc2`:
    Second-order collocation method
  - `colloc4`:
    Fourth-order collocation method
  - `colloc6`:
    Sixth-order collocation method
  - `colloc8`:
    Eight-order collocation method
  - `magnus2`:
    Second-order magnus method
  - `magnus4`:
    Fourth-order magnus method
  - `magnus6`:
    Sixth-order magnus method
  - `magnus8`:
    Eight-order magnus method

* `--frequency-units <FREQUENCY_UNITS>` — Units of lower and upper

  Default value: `dynamical`

  Possible values:
  - `dynamical`:
    Dynamical frequency of the star [sqrt(GM/R^3)]
  - `hertz`:
    Hertz [1/s]
  - `cycles-per-day`:
    Cycles per day [1/d]




## `deform`

Compute the P2 deformation of the stellar model

**Usage:** `deform [OPTIONS] <ROTATION>`

###### **Arguments:**

* `<ROTATION>` — Rotation frequency

   This parameter should match the rotation frequency of the model.

###### **Options:**

* `--frequency-units <FREQUENCY_UNITS>` — Units of rotation

  Default value: `dynamical`

  Possible values:
  - `dynamical`:
    Dynamical frequency of the star [sqrt(GM/R^3)]
  - `hertz`:
    Hertz [1/s]
  - `cycles-per-day`:
    Cycles per day [1/d]




## `post-process`

Compute derived properties from the eigenfunctions

This includes the radial and horizontal displacements, the density and pressure perturbations, mode identification, ...

**Usage:** `post-process`



## `perturb-deformed`

Perturb the mode frequencies and eigenfunctions to match the deformed star

**Usage:** `perturb-deformed <M>`

###### **Arguments:**

* `<M>` — Azimuthal order to do the perturbations for. This will filter the modes than have been obtain with scan to only those that have the same azimuthal order as selected here



## `output`

Write the results to an HDF5 file. Unless --keep-data is passed, this will clear all data except for the input model

**Usage:** `output [OPTIONS] <FILE>`

###### **Arguments:**

* `<FILE>` — The file to write the data to

###### **Options:**

* `--frequency-units <FREQUENCY_UNITS>` — All frequencies will be outputted in these units. This includes the mode frequencies, but also the rotation frequency

  Default value: `dynamical`

  Possible values:
  - `dynamical`:
    Dynamical frequency of the star [sqrt(GM/R^3)]
  - `hertz`:
    Hertz [1/s]
  - `cycles-per-day`:
    Cycles per day [1/d]

* `--properties <PROPERTIES>` — Mode properties to include in the output

   Some of these properties require post-processing to be available. The properties will be available as datasets in the root group, unless specified otherwise.

  Possible values:
  - `frequency`:
    Mode frequency in units given by frequency-units
  - `degree`:
    Spherical degree of the mode
  - `azimuthal-order`:
    Azimuthal order of the mode
  - `radial-order`:
    Radial order of the mode
  - `deformed-frequency`:
    Perturbed frequencies, units are given by frequency-units. It is stored in a subgroup of the deformation group. The name of that subgroup is given by the azimuthal order selected for the perturbative calculations
  - `deformed-eigenvector`:
    Eigenvectors for the perturbed system of equations. These can be used to construct the perturbed eigenfunctions from the actual eigenfunctions. Each solution of the perturbed system is a column of this matrix, while the rows map to one of the eigenfunctions. It is stored as eigenvector in the same group as the deformed-frequency option
  - `coupling-matrix`:
    Coupling matrices L, D, and R for the deformation perturbation. They can be found in the previously mentioned subgroup as l, d, and r respectively

* `--profiles <PROFILES>` — Profiles to include in the output. Requires post-processing.

   These profiles can be found in sub groups of the mode-profiles group. The name of the group is the index in the main solution arrays (e.g. frequency). Values can be separated by commas.

   All profiles are normalized.

  Possible values:
  - `radial-coordinate`:
    Radial coordinate of the points in the other datasets. Since this will be the same for all the modes, it can be found in the root group
  - `y1`:
    Dimensionless perturbation
  - `y2`:
    Dimensionless perturbation
  - `y3`:
    Dimensionless perturbation
  - `y4`:
    Dimensionless perturbation
  - `xi_r`:
    Radial displacement
  - `xi_h`:
    Horizontal displacement
  - `xi_tp`:
    Toroidal displacement (l + 1)
  - `xi_tn`:
    Toroidal displacement (l - 1)
  - `pressure`:
    Pressure perturbation
  - `density`:
    Density perturbation
  - `gravity-potential`:
    Gravity potential perturbation
  - `gravity-acceleration`:
    Gravity acceleration perturbation
  - `divergence`:
    Divergence of the displacement

* `--model-properties <MODEL_PROPERTIES>` — Model properties.

   These properties can be found in the model group. Values can be separated by commas.

  Possible values:
  - `dynamical-frequency`:
    Dynamical frequency of the star (sqrt(GM/R^3)) [1/s]
  - `deformation-beta`:
    The P2 deformation of the stellar structure. This quantity is unitless. This is only available if the deformation command has not been called
  - `deformation-dbeta`:
    The derivative of beta by a [1/cm]. It is stored in the model group. This is only available if the deformation command has not been called
  - `deformation-ddbeta`:
    The second derivative of beta by a [1/cm^2]. It is stored in the model group. This is only available if the deformation command has not been called
  - `deformation-rotation-frequency`:
    Rotation frequency used in for the deformation calculations. This might be replaced by just the model rotation frequency if shellular differential rotation is supported in the deformation calculations. Saved as an attribute, not a dataset

* `--keep-data` — Do not delete current computation results



<hr/>

<small><i>
    This document was generated automatically by
    <a href="https://crates.io/crates/clap-markdown"><code>clap-markdown</code></a>.
</i></small>

