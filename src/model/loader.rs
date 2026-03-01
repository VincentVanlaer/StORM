use self::gyre::from_gsm;
use gyre::{from_plain, overlay_rot};
use itertools::Itertools;
use std::{path::Path, str::FromStr};
use thiserror::Error;

use super::{DimensionlessProperties, DiscreteModel, DiscreteModelSegment};

impl DiscreteModel {
    /// Load a stellar model from a path
    pub fn load(path: &Path, format: ModelFormat) -> Result<DiscreteModel, ModelLoadError> {
        let model = match format {
            ModelFormat::Gyre => from_plain(path),
            ModelFormat::GSM => from_gsm(path),
        }
        .map_err(|x| ModelLoadError::InvalidFormat(x.into()))?;

        let props = &model.segments[0].dimensionless;

        let segments = split_segments(&props.r_coord)
            .map_err(|x| ModelLoadError::InvalidFormat(x.into()))?
            .iter()
            .map(|&(idx1, idx2)| DiscreteModelSegment {
                dimensionless: DimensionlessProperties {
                    r_coord: props.r_coord[idx1..idx2].to_vec().into(),
                    m_coord: props.m_coord[idx1..idx2].to_vec().into(),
                    rho: props.rho[idx1..idx2].to_vec().into(),
                    p: props.p[idx1..idx2].to_vec().into(),
                    v: props.v[idx1..idx2].to_vec().into(),
                    u: props.u[idx1..idx2].to_vec().into(),
                    gamma1: props.gamma1[idx1..idx2].to_vec().into(),
                    a_star: props.a_star[idx1..idx2].to_vec().into(),
                    c1: props.c1[idx1..idx2].to_vec().into(),
                    rot: props.rot[idx1..idx2].to_vec().into(),
                },
                metric: None,
            })
            .collect();

        Ok(DiscreteModel {
            segments,
            scale: model.scale,
            perturbed: None,
        })
    }

    /// Modify the rotation profile of the model, using an HDF5 file with a single `Omega_rot`
    /// dataset. This mirrors the GSM format.
    pub fn overlay_rot<P: AsRef<Path>>(&mut self, file: P) -> Result<(), ModelLoadError> {
        overlay_rot(self, file)
    }
}

/// Types of data formats supported for loading
#[non_exhaustive]
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum ModelFormat {
    /// The [MESA - GYRE plain text
    /// format](https://gyre.readthedocs.io/en/stable/ref-guide/stellar-models/mesa-file-format.html)
    Gyre,
    /// The [MESA - GYRE HDF5 format](https://gyre.readthedocs.io/en/stable/ref-guide/stellar-models/gsm-file-format.html)
    GSM,
}

impl FromStr for ModelFormat {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "gyre" => Ok(ModelFormat::Gyre),
            "gsm" => Ok(ModelFormat::GSM),
            _ => Err("Invalid format name"),
        }
    }
}

/// Error when loading a stellar model from a file
#[derive(Error, Debug)]
pub enum ModelLoadError {
    /// Invalid model file
    #[error("Invalid model file")]
    InvalidFormat(#[source] Box<dyn std::error::Error + Send + Sync>),
}

mod gyre {
    // As defined by MESA
    const GRAV: f64 = 6.67430e-8;

    use std::{f64::consts::PI, fs, path::Path};

    use hdf5::{File, H5Type};
    use ndarray::{Array1, s};

    use crate::model::{
        DimensionedProperties, DimensionlessProperties, DiscreteModel, DiscreteModelSegment,
    };

    use super::ModelLoadError;

    pub(super) fn from_gsm(
        path: &Path,
    ) -> Result<DiscreteModel, Box<dyn std::error::Error + Send + Sync>> {
        let input = &hdf5::File::open(path)?;

        let version = read_attr(input, "version")
            .and_then(|v| match v {
                100 => Ok(HDF5FormatVersion::V100),
                110 => Ok(HDF5FormatVersion::V110),
                120 => Ok(HDF5FormatVersion::V120),
                v => return Err(format!("Unknown version {v}").into()),
            })
            .unwrap_or(HDF5FormatVersion::V0);

        let n = read_attr(input, "n")?;
        let radius: f64 = read_attr(input, "R_star")?;
        let mass: f64 = read_attr(input, "M_star")?;

        let r_coord = &read_dataset(input, "r", n)?;

        let m_coord = &match version {
            HDF5FormatVersion::V0 => {
                let w = read_dataset(input, "w", n)?;
                mass / (1. / w + 1.)
            }
            HDF5FormatVersion::V100 | HDF5FormatVersion::V110 | HDF5FormatVersion::V120 => {
                read_dataset(input, "M_r", n)?
            }
        };

        let rho = &read_dataset(input, "rho", n)?;
        let p = &read_dataset(
            input,
            match version {
                HDF5FormatVersion::V0 => "p",
                _ => "P",
            },
            n,
        )?;
        let gamma1 = &read_dataset(input, "Gamma_1", n)?;
        let nsqrd = &read_dataset::<f64>(input, "N2", n)?;
        let rot = &read_dataset(input, "Omega_rot", n)?;

        Ok(convert_gyre(
            radius, mass, r_coord, m_coord, rho, p, gamma1, nsqrd, rot,
        ))
    }

    pub(super) fn from_plain(
        path: &Path,
    ) -> Result<DiscreteModel, Box<dyn std::error::Error + Send + Sync>> {
        let data: String = fs::read(path)?.try_into()?;

        let mut lines = data.lines();

        let mut header = lines
            .next()
            .ok_or("empty file")?
            .split(" ")
            .filter(|p| !p.is_empty());

        let npoints: usize = header.next().ok_or("missing npoints in header")?.parse()?;
        let mass: f64 = header.next().ok_or("missing mass in header")?.parse()?;
        let radius: f64 = header.next().ok_or("missing radius in header")?.parse()?;
        let _: f64 = header
            .next()
            .ok_or("missing luminosity in header")?
            .parse()?;

        let version = match header.next() {
            None => PlainFormatVersion::V1,
            Some(ver) => match ver {
                "19" => PlainFormatVersion::V19,
                "100" => PlainFormatVersion::V100,
                "101" => PlainFormatVersion::V101,
                "120" => PlainFormatVersion::V120,
                _ => return Err("Invalid version format".into()),
            },
        };

        let mut data = vec![
            Vec::with_capacity(npoints);
            match version {
                PlainFormatVersion::V1
                | PlainFormatVersion::V19
                | PlainFormatVersion::V100
                | PlainFormatVersion::V101 => 18,
                PlainFormatVersion::V120 => 19,
            }
        ];

        for line in lines {
            for (idx, val) in line
                .split(|arg| char::is_ascii_whitespace(&arg))
                .filter(|p| !p.is_empty())
                .skip(1)
                .map(|v| v.replace("d", "e").replace("D", "E").parse())
                .enumerate()
            {
                match data.get_mut(idx) {
                    None => return Err("invalid number of columns in line".into()),
                    Some(v) => v.push(val?),
                }
            }
        }

        for v in data.iter() {
            if v.len() != npoints {
                return Err("number of points in header not equal to rows".into());
            }
        }

        let r_coord = &data[0].clone().into();

        let m_coord = match version {
            PlainFormatVersion::V1 | PlainFormatVersion::V19 => {
                let w: &Array1<_> = &data[1].clone().into();
                &(mass / (1. / w + 1.))
            }
            PlainFormatVersion::V100 | PlainFormatVersion::V101 | PlainFormatVersion::V120 => {
                &data[1].clone().into()
            }
        };

        let rho = &data[5].clone().into();
        let p = &data[3].clone().into();
        let nsqrd = &data[7].clone().into();

        let gamma1 = match version {
            PlainFormatVersion::V1 => {
                let cp: &Array1<_> = &data[9].clone().into();
                let cv: &Array1<_> = &data[8].clone().into();
                &(cp / cv)
            }
            PlainFormatVersion::V19
            | PlainFormatVersion::V100
            | PlainFormatVersion::V101
            | PlainFormatVersion::V120 => &data[8].clone().into(),
        };

        let rot = match version {
            PlainFormatVersion::V1 => &vec![0.; npoints].into(),
            PlainFormatVersion::V19 | PlainFormatVersion::V100 | PlainFormatVersion::V101 => {
                &data[17].clone().into()
            }
            PlainFormatVersion::V120 => &data[18].clone().into(),
        };

        Ok(convert_gyre(
            radius, mass, r_coord, m_coord, rho, p, gamma1, nsqrd, rot,
        ))
    }

    pub(super) fn overlay_rot<P: AsRef<Path>>(
        model: &mut DiscreteModel,
        file: P,
    ) -> Result<(), ModelLoadError> {
        let input = &hdf5::File::open(file.as_ref())
            .map_err(|err| ModelLoadError::InvalidFormat(err.into()))?;
        let scale = model.scale.unwrap().freq_scale();
        let rot = read_dataset(
            input,
            "Omega_rot",
            model
                .segments
                .iter()
                .map(|x| x.dimensionless.r_coord.len())
                .sum(),
        )
        .map_err(|err| ModelLoadError::InvalidFormat(err.into()))?
        .mapv(|rot: f64| rot / scale);

        let mut idx = 0;

        for segment in model.segments.iter_mut() {
            let l = segment.dimensionless.rot.len();
            segment.dimensionless.rot = rot.slice(s![idx..(idx + l)]).to_vec().into();
            idx += l;
        }

        Ok(())
    }

    fn convert_gyre(
        radius: f64,
        mass: f64,
        r_coord: &Array1<f64>,
        m_coord: &Array1<f64>,
        rho: &Array1<f64>,
        p: &Array1<f64>,
        gamma1: &Array1<f64>,
        nsqrd: &Array1<f64>,
        rot: &Array1<f64>,
    ) -> DiscreteModel {
        let mut c1 = r_coord.mapv(|r: f64| r.powi(3)) / radius.powi(3) * mass / m_coord;
        c1[0] = mass / radius.powi(3) * 3. / (4. * PI * rho[0]);
        let mut a_star = r_coord.mapv(|r: f64| r.powi(3)) / (GRAV * m_coord) * nsqrd;
        a_star[0] = 0.;
        let mut v = GRAV * m_coord * rho / (p * r_coord);
        v[0] = 0.;
        let mut u = 4. * PI * rho * r_coord.mapv(|r| r.powi(3)) / m_coord;
        u[0] = 3.;

        DiscreteModel {
            segments: vec![DiscreteModelSegment {
                dimensionless: DimensionlessProperties {
                    r_coord: (r_coord / radius).to_vec().into(),
                    m_coord: (m_coord / mass).to_vec().into(),
                    rho: (rho / mass * radius.powi(3)).to_vec().into(),
                    p: (p / GRAV / mass.powi(2) * radius.powi(4)).to_vec().into(),
                    v: v.to_vec().into(),
                    u: u.to_vec().into(),
                    gamma1: gamma1.to_vec().into(),
                    a_star: a_star.to_vec().into(),
                    c1: c1.to_vec().into(),
                    rot: rot
                        .mapv(|rot: f64| rot / (GRAV * mass / radius.powi(3)).sqrt())
                        .to_vec()
                        .into(),
                },
                metric: None,
            }],
            scale: Some(DimensionedProperties {
                radius,
                mass,
                grav: GRAV,
            }),
            perturbed: None,
        }
    }

    fn read_attr<T: H5Type>(
        file: &File,
        attr: &'static str,
    ) -> Result<T, Box<dyn std::error::Error + Send + Sync>> {
        file.attr(attr)
            .and_then(|res| res.read_scalar())
            .map_err(|err| format!("Could not read attribute `{attr}`: {err}").into())
    }

    fn read_dataset<T: H5Type>(
        file: &File,
        attr: &'static str,
        expected_length: usize,
    ) -> Result<Array1<T>, Box<dyn std::error::Error + Send + Sync>> {
        let res = file
            .dataset(attr)
            .and_then(|res| res.read_1d())
            .map_err(|err| format!("Could not read dataset `{attr}`: {err}"))?;

        if res.len() != expected_length {
            return Err(format!(
                "Length mismatch, dataset `{attr}` has length {}, expected {expected_length}",
                res.len()
            )
            .into());
        }

        Ok(res)
    }

    enum PlainFormatVersion {
        V1,
        V19,
        V100,
        V101,
        V120,
    }

    enum HDF5FormatVersion {
        V0,
        V100,
        V110,
        V120,
    }
}

fn split_segments(r_coord: &[f64]) -> Result<Vec<(usize, usize)>, SegmentationError> {
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
        return Err(SegmentationError::ZeroLengthSegment(idx));
    }

    Ok(segment_indices)
}

#[derive(Error, Debug)]
enum SegmentationError {
    /// At least one segment in the model has zero length
    #[error("At least one segment has zero length. Starting point is at `{0}`")]
    ZeroLengthSegment(usize),
}
