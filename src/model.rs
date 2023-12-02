use color_eyre::Result;
use hdf5::File;
use ndarray::Array1;

pub(crate) struct StellarModel {
    pub(crate) radius: f64,
    pub(crate) mass: f64,
    pub(crate) r_coord: Array1<f64>,
    pub(crate) m_coord: Array1<f64>,
    pub(crate) rho: Array1<f64>,
    pub(crate) p: Array1<f64>,
    pub(crate) gamma1: Array1<f64>,
    pub(crate) nsqrd: Array1<f64>,
}

impl StellarModel {
    pub(crate) fn from_gsm(input: &File) -> Result<StellarModel> {
        let r_coord = input.dataset("r")?.read_1d::<f64>()?;
        let m_coord = input.dataset("M_r")?.read_1d::<f64>()?;
        let rho = input.dataset("rho")?.read_1d::<f64>()?;
        let p = input.dataset("P")?.read_1d::<f64>()?;
        let gamma1 = input.dataset("Gamma_1")?.read_1d::<f64>()?;
        let nsqrd = input.dataset("N2")?.read_1d::<f64>()?;

        Ok(StellarModel {
            radius: input.attr("R_star")?.read_scalar()?,
            mass: input.attr("M_star")?.read_scalar()?,
            r_coord,
            m_coord,
            rho,
            p,
            gamma1,
            nsqrd,
        })
    }
}
