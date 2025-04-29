//! Loading and modifying stellar models

pub trait Model {
    type ModelPoint;

    fn len(&self) -> usize;
    fn pos(&self, idx: usize) -> f64;

    fn eval(&self, idx: usize) -> Self::ModelPoint;
}

#[derive(Clone, Copy, Debug)]
pub struct DimensionlessProperties {
    pub v_gamma: f64,
    pub a_star: f64,
    pub u: f64,
    pub c1: f64,
    pub rot: f64,
}

pub mod interpolate;

/// GYRE stellar model support
pub mod gsm;
/// Polytrope support
pub mod polytrope;
