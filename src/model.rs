//! Loading and modifying stellar models

pub(crate) trait Model {
    fn len(&self) -> usize;
    fn pos(&self, idx: usize) -> f64;

    fn dimensionless_properties(&self, idx: usize) -> DimensionlessProperties;
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DimensionlessProperties {
    pub v_gamma: f64,
    pub a_star: f64,
    pub u: f64,
    pub c1: f64,
    pub rot: f64,
}

/// GYRE stellar model support
pub mod gsm;
