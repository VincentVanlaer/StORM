//! Loading and modifying stellar models

#[non_exhaustive]
/// Wrapper struct for potential model properties. Depending on the type of model (MESA, polytrope,
/// ...) the scale field may or may not be filled.
#[derive(Debug, Clone)]
pub struct DiscreteModel {
    /// Segments of the model
    pub segments: Vec<DiscreteModelSegment>,
    /// Scale parameters of the model
    pub scale: Option<DimensionedProperties>,
    /// Global perturbation parameters
    pub perturbed: Option<PerturbedParameters>,
}

/// Segment of a model
///
/// Models are split in segments to account for density discontinuities
#[derive(Debug, Clone)]
pub struct DiscreteModelSegment {
    /// Dimensionless properties of the model. This field information is always present
    pub dimensionless: DimensionlessProperties,
    /// Metric paramters of the model
    pub metric: Option<PerturbedMetric>,
}

/// A continuous segment of a stellar model.
///
/// The purpose of this struct is the indicate where potential discontinuities in the model are.
/// All properties of the model must be defined from `lower` to `upper`, including the edge points,
/// and must be continuous.
#[derive(Debug, Clone, Copy)]
pub struct Segment {
    /// Lower limit of the segment
    pub lower: f64,
    /// Upper limit of the segment
    pub upper: f64,
}

/// Stellar model which can be evaluated at any point. Typically obtained by interpolating a
/// [DiscreteModel]
pub trait ContinuousModel {
    /// Segments of the model
    fn segments(&self) -> Vec<Segment>;
    /// Evaluate the model at discrete points given by frational radius
    fn eval(&self, segment: usize, points: &[f64]) -> DiscreteModelSegment;
    /// Evaluate the model at discrete points given by frational radius for all segments
    fn eval_all(&self, points: &[&[f64]]) -> Vec<DiscreteModelSegment> {
        (0..self.segments().len())
            .map(|s_idx| self.eval(s_idx, points[s_idx]))
            .collect()
    }
    /// Forward the dimensions, these do not need to be interpolated
    fn dimensions(&self) -> Option<DimensionedProperties>;
}

// The goal of this implementation is to allow unifications of discrete models with continuous model
// with an additional layer of indirection (dyn* would be nice in this case). This effectively
// prevents &mut self and self members
impl<T: ContinuousModel + ?Sized> ContinuousModel for &T {
    fn segments(&self) -> Vec<Segment> {
        (*self).segments()
    }

    fn eval(&self, segment: usize, grid: &[f64]) -> DiscreteModelSegment {
        (*self).eval(segment, grid)
    }

    fn dimensions(&self) -> Option<DimensionedProperties> {
        (*self).dimensions()
    }
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

impl DimensionlessProperties {
    fn zeroed(len: usize) -> DimensionlessProperties {
        DimensionlessProperties {
            r_coord: vec![0.; len].into(),
            m_coord: vec![0.; len].into(),
            rho: vec![0.; len].into(),
            p: vec![0.; len].into(),
            v: vec![0.; len].into(),
            u: vec![0.; len].into(),
            gamma1: vec![0.; len].into(),
            a_star: vec![0.; len].into(),
            c1: vec![0.; len].into(),
            rot: vec![0.; len].into(),
        }
    }
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
    /// P0 perturbation
    pub alpha: Box<[f64]>,
    /// Derivative of alpha
    pub dalpha: Box<[f64]>,
    /// Second derivative of alpha
    pub ddalpha: Box<[f64]>,
    /// P2 perturbation
    pub beta: Box<[f64]>,
    /// Derivative of beta
    pub dbeta: Box<[f64]>,
    /// Second derivative of beta
    pub ddbeta: Box<[f64]>,
}

/// Deformation global parameters
#[derive(Debug, Copy, Clone)]
pub struct PerturbedParameters {
    /// Rotation frequency
    pub rot: f64,
    /// Relative difference in stellar mass due to the deformation
    pub mass_delta: f64,
}

impl PerturbedMetric {
    fn zeroed(len: usize) -> PerturbedMetric {
        PerturbedMetric {
            alpha: vec![0.; len].into(),
            dalpha: vec![0.; len].into(),
            ddalpha: vec![0.; len].into(),
            beta: vec![0.; len].into(),
            dbeta: vec![0.; len].into(),
            ddbeta: vec![0.; len].into(),
        }
    }
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
