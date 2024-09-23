use std::path::{Path, PathBuf};

use hdf5::{File, H5Type};
use ndarray::Array1;
use thiserror::Error;

// As defined by MESA
const GRAV: f64 = 6.67430e-8;

pub struct StellarModel {
    pub(crate) radius: f64,
    pub(crate) mass: f64,
    pub(crate) r_coord: Array1<f64>,
    pub(crate) m_coord: Array1<f64>,
    pub(crate) rho: Array1<f64>,
    pub(crate) p: Array1<f64>,
    pub(crate) gamma1: Array1<f64>,
    pub(crate) nsqrd: Array1<f64>,
    pub(crate) rot: Array1<f64>,
    // Gravitation constant, to ensure consistency with evolution code
    pub(crate) grav: f64,
}

#[derive(Error, Debug)]
#[non_exhaustive]
pub enum ModelError {
    #[error("Could not open `{0}`")]
    HDF5OpenError(PathBuf, #[source] hdf5::Error),
    #[error("Could not read `{0}` from model file")]
    HDF5DataReadError(&'static str, #[source] hdf5::Error),
    #[error("Length mismatch, expected {0} points, got {1} for dataset `{2}`")]
    LengthMismatch(usize, usize, &'static str),
}

fn read_attr<T: H5Type>(file: &File, attr: &'static str) -> Result<T, ModelError> {
    let res: Result<_, _> = try { file.attr(attr)?.read_scalar()? };

    res.map_err(|err| ModelError::HDF5DataReadError(attr, err))
}

fn read_dataset<T: H5Type>(
    file: &File,
    attr: &'static str,
    expected_length: usize,
) -> Result<Array1<T>, ModelError> {
    let res: Result<_, _> = try { file.dataset(attr)?.read_1d()? };

    let res = res.map_err(|err| ModelError::HDF5DataReadError(attr, err))?;

    if res.len() != expected_length {
        return Err(ModelError::LengthMismatch(expected_length, res.len(), attr));
    }

    Ok(res)
}

impl StellarModel {
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
            r_coord,
            m_coord,
            rho,
            p,
            gamma1,
            nsqrd,
            rot,
            grav: GRAV,
        })
    }

    pub fn overlay_rot<P: AsRef<Path>>(&mut self, file: P) -> Result<(), ModelError> {
        let input = &hdf5::File::open(file.as_ref())
            .map_err(|err| ModelError::HDF5OpenError(file.as_ref().to_owned(), err))?;
        self.rot = read_dataset(input, "Omega_rot", self.r_coord.len())?;

        Ok(())
    }
}
