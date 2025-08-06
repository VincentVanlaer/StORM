//! Loading and modifying stellar models

#[non_exhaustive]
/// Wrapper struct for potential model properties. Depending on the type of model (MESA, polytrope,
/// ...) the scale field may or may not be filled.
pub struct DiscreteModel {
    /// Dimensionless properties of the model. This field information is always present
    pub dimensionless: DimensionlessProperties,
    /// Scale parameters of the model
    pub scale: Option<DimensionedProperties>,
    /// Metric paramters of the model
    pub metric: Option<PerturbedMetric>,
}

/// Stellar model which can be evaluated at any point. Typically obtained by interpolating a
/// [DiscreteModel]
pub trait ContinuousModel {
    /// Inner radius of the model
    fn inner(&self) -> f64;
    /// Outer radius of the model
    fn outer(&self) -> f64;
    /// Evaluate the model at discrete points given by frational radius
    fn eval(&self, grid: &[f64]) -> DiscreteModel;
}

/// Stellar model used as input for the calculations. All properties are dimensionless. For
/// converting to dimensioned properties, see [DimensionedProperties].
#[derive(Debug, Clone)]
pub struct DimensionlessProperties {
    /// Radial coordinate \[R\]
    pub r_coord: Box<[f64]>,
    /// Mass coordinate \[M\]
    pub m_coord: Box<[f64]>,
    /// Density \[GM/R^3\]
    pub rho: Box<[f64]>,
    /// Pressure \[GM^2/R^4\]
    pub p: Box<[f64]>,
    /// Negative logarithmic derivative of pressure
    pub v: Box<[f64]>,
    /// Logarithmic derivative of mass coordinate
    pub u: Box<[f64]>,
    /// First adiabatic exponent
    pub gamma1: Box<[f64]>,
    /// Difference of pressure and density logarithmic derivatives
    pub a_star: Box<[f64]>,
    /// Inverse average inner density, scaled by overal average density
    pub c1: Box<[f64]>,
    /// Rotation rate as fraction of critical
    pub rot: Box<[f64]>,
}

/// Total radius and mass, and gravitational constant
#[derive(Debug, Clone, Copy)]
pub struct DimensionedProperties {
    /// Total stellar radius \[cm\]
    pub radius: f64,
    /// Total stellar mass \[g\]
    pub mass: f64,
    /// Gravitational acceleration \[Ncm^2/g\]
    pub grav: f64,
}

/// Contains the results of deforming the stellar structure with rotation
#[derive(Debug, Clone)]
pub struct PerturbedMetric {
    /// P2 perturbation
    pub beta: Box<[f64]>,
    /// Derivative of beta
    pub dbeta: Box<[f64]>,
    /// Second derivative of beta
    pub ddbeta: Box<[f64]>,
    /// Rotation frequency
    pub rot: f64,
}

impl DimensionedProperties {
    /// Compute the dynamical frequency of the model
    pub fn freq_scale(&self) -> f64 {
        (self.grav * self.mass / self.radius.powi(3)).sqrt()
    }
}

/// Interpolation of stellar models. Turns a [DiscreteModel] into a [ContinuousModel]
pub mod interpolate;

/// GYRE stellar model support
pub mod gsm;
/// Polytrope support
pub mod polytrope;
