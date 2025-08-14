---
title: Example walkthrough
icon: footprint
---

The goal of this walkthrough is to show you some of the functionality of StORM. It is not intended to be a reference for all functionality. For that, have a look at the [CLI reference](../reference).
We will investigate the low-order oscillation modes of a *β* Cep stellar model, although you are free to bring your own stellar model.

## Stellar model

In order to compute oscillation frequencies with StORM, you first need to grab a stellar model.
StORM currently only supports HDF5 stellar models from the [MESA](https://mesastar.org) stellar oscillation code.
See the [GYRE documentation](https://gyre.readthedocs.io/en/stable/ref-guide/stellar-models/gsm-file-format.html) for more details.
Other formats can be transformed to this format using the [tomso python package](https://github.com/warrickball/tomso).
This walkthrough will make use of one of the models used for testing StORM (the `test-model-tams.GSM` model).
You can find this model in the `test-data` folder of the source release, or you can directly [download it]().

## Running StORM

To start StORM, run one of the following three commands in a terminal:

- If StORM is in your `PATH`, you can run it directly as `storm`
- If StORM is not in your `PATH` but you have the binary, run it as `full/path/to/storm`.
- If you are in the source folder of StORM and want to build and run it immediately, run `cargo run --release --bin storm`.

If everything goes well, you should see the following:

```shell
[storm] > 
```

This is the StORM command line interface and this is where you will enter commands to let StORM do the oscillation calculations.
If you want to know what all the available commands are, type in `help` and press enter.
We will cover most of the commands you see in your terminal in this walkthrough.
For an online overview of all of these commands, see the [CLI reference](../reference).

### Loading the stellar model

To load the stellar model, run the command `input test-model-tams.GSM`.
If you want to load a different model, you can of course change the name.
StORM will now load the model and tell you the number of points that the model contains.

```shell
[storm] > input test-model-tams.GSM
Loaded model with 6200 points
```

### Searching for oscillation frequencies

To find oscillation frequencies, StORM scans a region in frequency space at a given step size.
The frequencies of all modes that are found, are iteratively refined until the requested precision is reached.
Each scan must also be repeated for a different combination of spherical degree and azimuthal order.

```shell
# The format of the scan command is `scan degree azimuthal-order lower-frequency upper-frequency number-of-scan-points
[storm] > scan 0 0 1 30 30 --frequency-units=cycles-per-day  
Found 20 modes
```

In order for the modes in the region to be detected, only a single mode may lie between two scan points.
Hence, the scan step size must be small enough for all modes to be detected.

### Computing displacement

The scanning step only determines the frequencies of the modes.
In order to obtain the displacement functions, and derived properties such as the radial order, some post-processing must take place.
To run this post-processing, use the `post-process` command.

```shell
[storm] > post-process
```

### Outputting results

To write the results to a file, use the `output` command.
You must specify which information you want to be outputted.
This is accomplished using various arguments to the `output` command:

```shell
[storm] > output --properties=frequency,degree,radial-order --profiles=radial-coordinate,xi_r,xi_h output.hdf5
```

StORM only outputs HDF5 files, which is a binary file format.
The structure of the generated file is as follows:

```
GROUP "/" {
   DATASET "degree" {
      DATATYPE  H5T_STD_U64LE
      DATASPACE  SIMPLE { ( 20 ) / ( 20 ) }
   }
   DATASET "frequency" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( 20 ) / ( 20 ) }
   }
   GROUP "mode-profiles" {
      GROUP "0" {
         DATASET "xi_h" {
            DATATYPE  H5T_IEEE_F64LE
            DATASPACE  SIMPLE { ( 6200 ) / ( 6200 ) }
         }
         DATASET "xi_r" {
            DATATYPE  H5T_IEEE_F64LE
            DATASPACE  SIMPLE { ( 6200 ) / ( 6200 ) }
         }
      }
      ... (one group for each mode)
   }
   DATASET "radial-coordinate" {
      DATATYPE  H5T_IEEE_F64LE
      DATASPACE  SIMPLE { ( 6200 ) / ( 6200 ) }
   }
   DATASET "radial-order" {
      DATATYPE  H5T_STD_I64LE
      DATASPACE  SIMPLE { ( 20 ) / ( 20 ) }
   }
}
```

This can read using packages such as `h5py` for Python.
