use std::{
    f64::consts::PI,
    path::{Path, PathBuf},
};

use hdf5::{File, H5Type};
use itertools::Itertools;
use ndarray::{Array1, s};
use thiserror::Error;

use super::{DimensionedProperties, DimensionlessProperties, DiscreteModel, DiscreteModelSegment};

// As defined by MESA
const GRAV: f64 = 6.67430e-8;

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
    /// At least one segment in the model has zero length
    #[error("At least one segment has zero length. Starting point is at `{0}`")]
    ZeroLengthSegment(usize),
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

fn split_segments(r_coord: &[f64]) -> Result<Vec<(usize, usize)>, ModelError> {
    let segment_indices = std::iter::once(0)
        .chain(
            r_coord
                .iter()
                .tuple_windows()
                .enumerate()
                .filter_map(|(idx, (r1, r2))| if r1 == r2 { Some(idx + 1) } else { None }),
        )
        .chain(std::iter::once(r_coord.len()))
        .tuple_windows()
        .collect_vec();

    if let Some(idx) = segment_indices
        .iter()
        .filter_map(|(idx1, idx2)| {
            if *idx1 + 1 == *idx2 {
                Some(*idx1)
            } else {
                None
            }
        })
        .next()
    {
        return Err(ModelError::ZeroLengthSegment(idx));
    }

    Ok(segment_indices)
}

impl DiscreteModel {
    /// Load a stellar model from a GYRE stellar model HDF5 file.
    ///
    /// The format is documented as part of the GYRE
    /// [documentation](https://gyre.readthedocs.io/en/stable/ref-guide/stellar-models/gsm-file-format.html).
    /// Requires at least version 1.00 of the GSM format.
    pub fn from_gsm<P: AsRef<Path>>(file: P) -> Result<DiscreteModel, ModelError> {
        let input = &hdf5::File::open(file.as_ref())
            .map_err(|err| ModelError::HDF5OpenError(file.as_ref().to_owned(), err))?;
        let n = read_attr(input, "n")?;
        let radius: f64 = read_attr(input, "R_star")?;
        let mass: f64 = read_attr(input, "M_star")?;
        let r_coord = &read_dataset(input, "r", n)?;
        let m_coord = &read_dataset(input, "M_r", n)?;
        let rho = &read_dataset(input, "rho", n)?;
        let p = &read_dataset(input, "P", n)?;
        let gamma1 = &read_dataset(input, "Gamma_1", n)?;
        let nsqrd = &read_dataset::<f64>(input, "N2", n)?;
        let rot = &read_dataset(input, "Omega_rot", n)?
            .mapv(|rot: f64| rot / (GRAV * mass / radius.powi(3)).sqrt());

        let mut c1 = r_coord.mapv(|r: f64| r.powi(3)) / radius.powi(3) * mass / m_coord;
        c1[0] = mass / radius.powi(3) * 3. / (4. * PI * rho[0]);
        let mut a_star = r_coord.mapv(|r: f64| r.powi(3)) / (GRAV * m_coord) * nsqrd;
        a_star[0] = 0.;
        let mut v = GRAV * m_coord * rho / (p * r_coord);
        v[0] = 0.;
        let mut u = 4. * PI * rho * r_coord.mapv(|r| r.powi(3)) / m_coord;
        u[0] = 3.;

        let segment_indices = split_segments(r_coord.as_slice().unwrap())?;

        let segments = segment_indices
            .iter()
            .map(|&(idx1, idx2)| DiscreteModelSegment {
                dimensionless: DimensionlessProperties {
                    r_coord: (r_coord / radius).slice(s![idx1..idx2]).to_vec().into(),
                    m_coord: (m_coord / mass).slice(s![idx1..idx2]).to_vec().into(),
                    rho: (rho / mass * radius.powi(3))
                        .slice(s![idx1..idx2])
                        .to_vec()
                        .into(),
                    p: (p / GRAV / mass.powi(2) * radius.powi(4))
                        .slice(s![idx1..idx2])
                        .to_vec()
                        .into(),
                    v: v.slice(s![idx1..idx2]).to_vec().into(),
                    u: u.slice(s![idx1..idx2]).to_vec().into(),
                    gamma1: gamma1.slice(s![idx1..idx2]).to_vec().into(),
                    a_star: a_star.slice(s![idx1..idx2]).to_vec().into(),
                    c1: c1.slice(s![idx1..idx2]).to_vec().into(),
                    rot: rot.slice(s![idx1..idx2]).to_vec().into(),
                },
                metric: None,
            })
            .collect();

        Ok(DiscreteModel {
            segments,
            scale: Some(DimensionedProperties {
                radius,
                mass,
                grav: GRAV,
            }),
            perturbed: None,
        })
    }

    /// Modify the rotation profile of the model, using an HDF5 file with a single `Omega_rot`
    /// dataset. This mirrors the GSM format.
    pub fn overlay_rot<P: AsRef<Path>>(&mut self, file: P) -> Result<(), ModelError> {
        let input = &hdf5::File::open(file.as_ref())
            .map_err(|err| ModelError::HDF5OpenError(file.as_ref().to_owned(), err))?;
        let scale = self.scale.unwrap().freq_scale();
        let rot = read_dataset(
            input,
            "Omega_rot",
            self.segments
                .iter()
                .map(|x| x.dimensionless.r_coord.len())
                .sum(),
        )?
        .mapv(|rot: f64| rot / scale);

        let mut idx = 0;

        for segment in self.segments.iter_mut() {
            let l = segment.dimensionless.rot.len();
            segment.dimensionless.rot = rot.slice(s![idx..(idx + l)]).to_vec().into();
            idx += l;
        }

        Ok(())
    }
}
