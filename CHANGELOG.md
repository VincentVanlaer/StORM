# Unreleased changes

## Features

* Part of the tidal effect can also be taken into account when deforming stars in binaries. Orbital parameters (eccentricity, mass ratio, and semi-major axis) can be passed to the `deform` command.
* Plain text and old versions of the GYRE stellar model formats are now supported. The model format can be passed to the `input` command. By default the HDF5 file format is assumed.

## Bug fixes

* StORM will no longer crash when no modes are found but perturbation of the frequencies is still attempted.
