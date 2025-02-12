//! Loading and modifying stellar models

use std::{
    f64::consts::PI,
    path::{Path, PathBuf},
};

use hdf5::{File, H5Type};
use ndarray::Array1;
use thiserror::Error;

// As defined by MESA
const GRAV: f64 = 6.67430e-8;

/// Stellar model used as input for the calculations.
///
/// Loading from the following formats is currently supported:
///
/// - GYRE's HDF5 stellar model
#[derive(Debug, Clone)]
pub struct StellarModel {
    /// Total radius of the model \[cm\]
    pub radius: f64,
    /// Total mass of the model \[g\]
    pub mass: f64,
    /// Radial coordinate \[cm\], ranges from 0 to radius
    pub r_coord: Box<[f64]>,
    /// Mass coordinate \[g\], ranges from 0 to mass
    pub m_coord: Box<[f64]>,
    /// Density \[g/cm^3\]
    pub rho: Box<[f64]>,
    /// Pressure \[Ba\]
    pub p: Box<[f64]>,
    /// First adiabatic exponent \[dimensionless\]
    pub gamma1: Box<[f64]>,
    /// Square of the buoyancy frequency \[s^-2\]
    pub nsqrd: Box<[f64]>,
    /// Angular rotation frequency \[rad/s\]
    pub rot: Box<[f64]>,
    /// Gravitional acceleration \[Ncm^2/g\]
    pub grav: f64,
}

/// Errors that can be returned when loading a stellar model
#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ModelError {
    /// Failed to open the model.
    ///
    /// The first parameter is the file that failed to open, the second parameter is the error the
    /// hdf5 crate returned.
    #[error("Could not open `{0}`")]
    HDF5OpenError(PathBuf, #[source] hdf5::Error),
    /// Failed to read an attribute or a dataset.
    ///
    /// The first parameter is the attribute or dataset that failed to be read. The second
    /// parameter is the error the hdf5 crate returned.
    #[error("Could not read `{0}` from model file")]
    HDF5DataReadError(&'static str, #[source] hdf5::Error),
    /// Length of a dataset does not match the expected length.
    ///
    /// The first parameter is the expected length, the second parameter the actual length, and the
    /// third parameter the name of the dataset.
    #[error("Length mismatch, expected {0} points, got {1} for dataset `{2}`")]
    LengthMismatch(usize, usize, &'static str),
}

fn read_attr<T: H5Type>(file: &File, attr: &'static str) -> Result<T, ModelError> {
    file.attr(attr)
        .and_then(|res| res.read_scalar())
        .map_err(|err| ModelError::HDF5DataReadError(attr, err))
}

fn read_dataset<T: H5Type>(
    file: &File,
    attr: &'static str,
    expected_length: usize,
) -> Result<Array1<T>, ModelError> {
    let res = file
        .dataset(attr)
        .and_then(|res| res.read_1d())
        .map_err(|err| ModelError::HDF5DataReadError(attr, err))?;

    if res.len() != expected_length {
        return Err(ModelError::LengthMismatch(expected_length, res.len(), attr));
    }

    Ok(res)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DimensionlessCoefficients {
    pub v_gamma: f64,
    pub a_star: f64,
    pub u: f64,
    pub c1: f64,
}

impl StellarModel {
    /// Load a stellar model from a GYRE stellar model HDF5 file.
    ///
    /// The format is documented as part of the GYRE
    /// [documentation](https://gyre.readthedocs.io/en/stable/ref-guide/stellar-models/gsm-file-format.html).
    /// Requires at least version 1.00 of the GSM format.
    pub fn from_gsm<P: AsRef<Path>>(file: P) -> Result<StellarModel, ModelError> {
        let input = &hdf5::File::open(file.as_ref())
            .map_err(|err| ModelError::HDF5OpenError(file.as_ref().to_owned(), err))?;
        let n = read_attr(input, "n")?;
        let radius = read_attr(input, "R_star")?;
        let mass = read_attr(input, "M_star")?;
        let r_coord = read_dataset(input, "r", n)?;
        let m_coord = read_dataset(input, "M_r", n)?;
        let rho = read_dataset(input, "rho", n)?;
        let p = read_dataset(input, "P", n)?;
        let gamma1 = read_dataset(input, "Gamma_1", n)?;
        let nsqrd = read_dataset(input, "N2", n)?;
        let rot = read_dataset(input, "Omega_rot", n)?;

        Ok(StellarModel {
            radius,
            mass,
            r_coord: r_coord.to_vec().into(),
            m_coord: m_coord.to_vec().into(),
            rho: rho.to_vec().into(),
            p: p.to_vec().into(),
            gamma1: gamma1.to_vec().into(),
            nsqrd: nsqrd.to_vec().into(),
            rot: rot.to_vec().into(),
            grav: GRAV,
        })
    }

    /// Modify the rotation profile of the model, using an HDF5 file with a single `Omega_rot`
    /// dataset. This mirrors the GSM format.
    pub fn overlay_rot<P: AsRef<Path>>(&mut self, file: P) -> Result<(), ModelError> {
        let input = &hdf5::File::open(file.as_ref())
            .map_err(|err| ModelError::HDF5OpenError(file.as_ref().to_owned(), err))?;
        self.rot = read_dataset(input, "Omega_rot", self.r_coord.len())?
            .to_vec()
            .into();

        Ok(())
    }

    pub(crate) fn dimensionless_coefficients(&self, i: usize) -> DimensionlessCoefficients {
        if i == 0 {
            DimensionlessCoefficients {
                v_gamma: 0.,
                a_star: 0.,
                u: 3.,
                c1: self.mass / self.radius.powi(3) * 3. / (4. * PI * self.rho[0]),
            }
        } else {
            let r_cubed = self.r_coord[i].powi(3);
            DimensionlessCoefficients {
                v_gamma: self.grav * self.m_coord[i] * self.rho[i]
                    / (self.p[i] * self.r_coord[i] * self.gamma1[i]),
                a_star: r_cubed / (self.grav * self.m_coord[i]) * self.nsqrd[i],
                u: 4. * PI * self.rho[i] * r_cubed / self.m_coord[i],
                c1: r_cubed / self.radius.powi(3) * self.mass / self.m_coord[i],
            }
        }
    }
}
