use nalgebra::{ComplexField, Const, Dim, DimMul, Dyn, MatrixViewMut, OMatrix};
use std::mem::MaybeUninit;

use super::{Boundary, GridLength, Moments};
use crate::linalg::{OMatrixArray, OwnedArray};
use crate::model::{DimensionlessCoefficients, StellarModel};
use crate::stepper::StepMoments;

/// Spherically symmetric stellar oscillation equations with a full rotation term
///
/// Note that this system of equations only contains a single spherical harmonic, and will
/// therefore not be completely valid in a rotating star.
pub struct Rotating1D {
    components: Vec<ModelPoint>,
    preprocessed: OMatrixArray<f64, Const<4>, Const<4>, Dyn>,
    ell: f64,
    m: f64,
}

type FourMatrix = OMatrix<f64, Const<4>, Const<4>>;

#[derive(Clone, Copy, Debug)]
struct ModelPoint {
    coeff: DimensionlessCoefficients,
    rot: f64,
    x: f64,
}

impl Rotating1D {
    /// Construct this system of equations from a stellar model and selection of the relevant
    /// spherical harmonic.
    pub fn from_model(value: &StellarModel, ell: u64, m: i64) -> Rotating1D {
        let ell = ell as f64;
        let scale = (value.grav * value.mass / value.radius.powi(3)).sqrt();

        let components: Vec<_> = value
            .rot
            .iter()
            .zip(&value.r_coord)
            .enumerate()
            .map(|(i, (&rot, x))| ModelPoint {
                coeff: value.dimensionless_coefficients(i),
                rot: rot / scale,
                x: x / value.radius,
            })
            .collect();

        let mut preprocessed = OMatrixArray::new_with(
            Const {},
            Const {},
            Dyn {
                0: components.len(),
            },
            || 0.,
        );

        for (i, val) in components.iter().enumerate() {
            preprocess::<true, Const<1>, _>(
                Gated::<_, true>::new(val.rot),
                val.coeff.u,
                val.coeff.v_gamma,
                val.coeff.a_star,
                val.coeff.c1,
                ell,
                m as f64,
                preprocessed.index_mut(i),
            );
        }

        Rotating1D {
            components,
            preprocessed,
            ell,
            m: m as f64,
        }
    }
}

struct ModelPointsIterator<'model> {
    model: &'model Rotating1D,
    pos: usize,
    skip: usize,
    subpos: usize,
    total_subpos: usize,
    frequency: f64,
}

impl ModelPointsIterator<'_> {
    fn new(scale: i32, model: &Rotating1D, frequency: f64) -> ModelPointsIterator {
        if scale >= 0 {
            ModelPointsIterator {
                model,
                pos: 1,
                subpos: 0,
                skip: 1,
                total_subpos: 2_usize.pow(scale.unsigned_abs()),
                frequency,
            }
        } else {
            ModelPointsIterator {
                model,
                pos: 1,
                subpos: 0,
                skip: 2_usize.pow(scale.unsigned_abs()),
                total_subpos: 1,
                frequency,
            }
        }
    }
}

impl Iterator for ModelPointsIterator<'_> {
    type Item = (FourMatrix, FourMatrix);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.model.components.is_empty() || (self.pos + self.skip) >= self.model.components.len()
        {
            return None;
        }
        // Bounds check due to potential overflow
        let lower = unsafe { self.model.components.get_unchecked(self.pos) };
        let upper = self.model.components[self.pos + self.skip];

        let delta = (upper.x - lower.x) / (self.total_subpos as f64);
        let sublower = lower.x + delta * (self.subpos as f64);
        let subupper = lower.x + delta * (self.subpos as f64 + 1.);

        let delta = subupper - sublower;

        let lower_a = {
            let mut a = unsafe { self.model.preprocessed.get_unchecked(self.pos) }.clone_owned();
            process::<true, Const<1>, _>(
                Gated::<_, true>::new(lower.rot),
                self.frequency,
                lower.coeff.u,
                lower.coeff.v_gamma,
                lower.coeff.a_star,
                lower.coeff.c1,
                self.model.ell,
                self.model.m,
                a.as_view_mut(),
            );

            a
        } * (delta / lower.x);

        let upper_a = {
            let mut a = unsafe { self.model.preprocessed.get_unchecked(self.pos + self.skip) }
                .clone_owned();
            process::<true, Const<1>, _>(
                Gated::<_, true>::new(upper.rot),
                self.frequency,
                upper.coeff.u,
                upper.coeff.v_gamma,
                upper.coeff.a_star,
                upper.coeff.c1,
                self.model.ell,
                self.model.m,
                a.as_view_mut(),
            );

            a
        } * (delta / upper.x);

        let intercept = lower_a
            + (upper_a - lower_a) * ((self.subpos as f64 + 0.5) / (self.total_subpos as f64));
        let slope = (upper_a - lower_a) * (1. / (self.total_subpos as f64));

        self.subpos += 1;

        if self.subpos == self.total_subpos {
            self.pos += self.skip;
            self.subpos = 0;
        }

        Some((slope, intercept))
    }
}

impl ExactSizeIterator for ModelPointsIterator<'_> {
    fn len(&self) -> usize {
        if self.skip == 1 {
            (self.model.components.len() - 2) * self.total_subpos
        } else {
            (self.model.components.len() - 2) / self.skip
        }
    }
}

struct IterWrapper<'a, G> {
    iter: ModelPointsIterator<'a>,
    wrapped: G,
}

type FourStepMoments<const ORDER: usize> =
    StepMoments<f64, Const<4>, Const<ORDER>, OwnedArray<f64, Const<4>, Const<4>, Const<ORDER>>>;

impl<const ORDER: usize, G: Fn((FourMatrix, FourMatrix)) -> FourStepMoments<ORDER>> Iterator
    for IterWrapper<'_, G>
{
    type Item = FourStepMoments<ORDER>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            None => None,
            Some(r) => Some((self.wrapped)(r)),
        }
    }
}

impl<const ORDER: usize, G: Fn((FourMatrix, FourMatrix)) -> FourStepMoments<ORDER>>
    ExactSizeIterator for IterWrapper<'_, G>
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl GridLength<GridScale> for Rotating1D {
    fn len(&self, grid: &GridScale) -> usize {
        if grid.scale >= 0 {
            (self.components.len() - 2) * 2usize.pow(grid.scale.unsigned_abs())
        } else {
            (self.components.len() - 2) / 2usize.pow(grid.scale.unsigned_abs())
        }
    }
}

impl Moments<f64, GridScale, Const<4>, Const<1>> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &GridScale,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = FourStepMoments<1>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(_s, i)| StepMoments {
                moments: [i].into(),
            },
        }
    }

    fn shape(&self) -> Const<4> {
        Const
    }
}
impl Moments<f64, GridScale, Const<4>, Const<2>> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &GridScale,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = FourStepMoments<2>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(s, i)| StepMoments {
                moments: [i, s].into(),
            },
        }
    }

    fn shape(&self) -> Const<4> {
        Const
    }
}
impl Moments<f64, GridScale, Const<4>, Const<3>> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &GridScale,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = FourStepMoments<3>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(s, i)| StepMoments {
                moments: [i, s, [[0.0; 4]; 4].into()].into(),
            },
        }
    }

    fn shape(&self) -> Const<4> {
        Const
    }
}
impl Moments<f64, GridScale, Const<4>, Const<4>> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &GridScale,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = FourStepMoments<4>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(s, i)| StepMoments {
                moments: [i, s, [[0.0; 4]; 4].into(), [[0.0; 4]; 4].into()].into(),
            },
        }
    }

    fn shape(&self) -> Const<4> {
        Const
    }
}

impl Boundary<f64, Const<4>, Const<2>> for Rotating1D {
    fn inner_boundary(&self, omega: f64) -> OMatrix<f64, Const<2>, Const<4>> {
        let mut b = OMatrix::<f64, Const<2>, Const<4>>::default();
        let props = self.components.first().unwrap();

        inner_bound::<true, Const<1>, _>(
            Gated::<_, true>::new(props.rot),
            omega,
            props.coeff.u,
            props.coeff.v_gamma,
            props.coeff.a_star,
            props.coeff.c1,
            self.ell,
            self.m,
            b.as_view_mut(),
        );

        b
    }

    fn outer_boundary(&self, omega: f64) -> OMatrix<f64, Const<2>, Const<4>> {
        let mut b = OMatrix::<f64, Const<2>, Const<4>>::default();
        let props = self.components.last().unwrap();

        outer_bound::<true, Const<1>, _>(
            Gated::<_, true>::new(props.rot),
            omega,
            props.coeff.u,
            props.coeff.v_gamma,
            props.coeff.a_star,
            props.coeff.c1,
            self.ell,
            self.m,
            b.as_view_mut(),
        );

        b
    }
}

/// Grid scaling for [Rotating1D]
///
/// This is a rough method of scaling the grid, as every interval will be divided a certain number
/// od times, without taking into account where refinement is actually necessary.
pub struct GridScale {
    /// How much times the grid is divided into two sub intervals
    ///
    /// If `scale` equals one, the grid is divide in two, if `scale` equals two, the grid is
    /// divided into four, ...
    pub scale: i32,
}

macro_rules! g {
    ($ff: ident, $e: expr, $alt: expr) => {
        $ff.map(|&$ff| $e, || $alt)
    };
}

pub(crate) struct Gated<T, const ACTIVE: bool> {
    val: MaybeUninit<T>,
}

impl<T> Gated<T, false> {
    pub(crate) fn new() -> Self {
        Gated {
            val: MaybeUninit::uninit(),
        }
    }
}

impl<T> Gated<T, true> {
    pub(crate) fn new(val: T) -> Self {
        Gated {
            val: MaybeUninit::new(val),
        }
    }
}

impl<T, const ACTIVE: bool> Gated<T, ACTIVE> {
    #[inline(always)]
    fn map(&self, f: impl FnOnce(&T) -> T, alt: impl FnOnce() -> T) -> T {
        if ACTIVE {
            // SAFETY: the only way to obtain a Gated object with ACTIVE set to true is the
            // Gated::new which constructs val using MaybeUninit::new
            f(unsafe { self.val.assume_init_ref() })
        } else {
            alt()
        }
    }
}

#[inline(always)]
pub(crate) fn preprocess<const ROTATION: bool, Degrees: Dim, T: ComplexField + Copy>(
    rotation: Gated<T, ROTATION>,
    u: T,
    v_gamma: T,
    a_star: T,
    c1: T,
    ell: T,
    m: T,
    mut output: MatrixViewMut<
        T,
        <Degrees as DimMul<Const<4>>>::Output,
        <Degrees as DimMul<Const<4>>>::Output,
    >,
) where
    Degrees: DimMul<Const<4>>,
{
    assert_eq!(output.shape(), (4, 4));

    let zero = T::from_subset(&0.);
    let one = T::from_subset(&1.);
    let two = T::from_subset(&2.);
    let three = T::from_subset(&3.);
    let lambda = ell * (ell + one);

    *output.index_mut((0, 0)) = v_gamma - one - ell;
    *output.index_mut((0, 1)) = -v_gamma;
    *output.index_mut((0, 2)) = zero;
    *output.index_mut((0, 3)) = zero;

    *output.index_mut((1, 0)) = -a_star;
    *output.index_mut((1, 1)) = a_star - u + three - ell;
    *output.index_mut((1, 2)) = zero;
    *output.index_mut((1, 3)) = -one;

    *output.index_mut((2, 0)) = zero;
    *output.index_mut((2, 1)) = zero;
    *output.index_mut((2, 2)) = three - u - ell;
    *output.index_mut((2, 3)) = one;

    *output.index_mut((3, 0)) = u * a_star;
    *output.index_mut((3, 1)) = u * v_gamma;
    *output.index_mut((3, 2)) = lambda;
    *output.index_mut((3, 3)) = -u + two - ell;
}

#[inline(always)]
pub(crate) fn process<const ROTATION: bool, Degrees: Dim, T: ComplexField + Copy>(
    rotation: Gated<T, ROTATION>,
    freq: T,
    u: T,
    v_gamma: T,
    a_star: T,
    c1: T,
    ell: T,
    m: T,
    mut output: MatrixViewMut<
        T,
        <Degrees as DimMul<Const<4>>>::Output,
        <Degrees as DimMul<Const<4>>>::Output,
    >,
) where
    Degrees: DimMul<Const<4>>,
{
    assert_eq!(output.shape(), (4, 4));

    let zero = T::from_subset(&0.);
    let one = T::from_subset(&1.);
    let two = T::from_subset(&2.);
    let three = T::from_subset(&3.);
    let lambda = ell * (ell + one);

    if ell != zero {
        let rot = g!(rotation, m * rotation, zero);
        let omega_rsq = (lambda * (freq - rot) + two * rot) * (freq - rot);
        let rel_rot = two * rot / (lambda * (freq - rot) + two * rot);

        *output.index_mut((0, 0)) += -lambda * rel_rot;
        *output.index_mut((0, 1)) += lambda.powi(2) / (omega_rsq * c1);
        *output.index_mut((0, 2)) += lambda.powi(2) / (omega_rsq * c1);

        *output.index_mut((1, 1)) += lambda * rel_rot;
        *output.index_mut((1, 2)) += lambda * rel_rot;
        *output.index_mut((1, 0)) +=
            c1 * (freq - rot).powi(2) * (one - (two * rot).powi(2) / omega_rsq);
    } else {
        *output.index_mut((1, 0)) += c1 * freq.powi(2);
    }
}

#[inline]
pub(crate) fn inner_bound<const ROTATION: bool, Degrees: Dim, T: ComplexField + Copy>(
    rotation: Gated<T, ROTATION>,
    freq: T,
    _u: T,
    _v_gamma: T,
    _a_star: T,
    c1: T,
    ell: T,
    m: T,
    mut output: MatrixViewMut<
        T,
        <Degrees as DimMul<Const<2>>>::Output,
        <Degrees as DimMul<Const<4>>>::Output,
    >,
) where
    Degrees: DimMul<Const<4>> + DimMul<Const<2>>,
{
    let zero = T::from_subset(&0.);
    let one = T::from_subset(&1.);
    let two = T::from_subset(&2.);
    let lambda = ell * (ell + one);

    if ell != zero {
        let rot = g!(rotation, m * rotation, zero);
        let omega_rsq = (lambda * (freq - rot) + two * rot) * (freq - rot);
        let rel_rot = two * rot / (lambda * (freq - rot) + two * rot);

        *output.index_mut((0, 0)) =
            c1 * (freq - rot).powi(2) * (one - (two * rot).powi(2) / omega_rsq);
        *output.index_mut((0, 1)) = lambda * rel_rot - ell;
        *output.index_mut((0, 2)) = lambda * rel_rot - ell;
        *output.index_mut((0, 3)) = zero;

        *output.index_mut((1, 0)) = zero;
        *output.index_mut((1, 1)) = zero;
        *output.index_mut((1, 2)) = ell;
        *output.index_mut((1, 3)) = -one;
    } else {
        *output.index_mut((0, 0)) = c1 * freq * freq;
        *output.index_mut((0, 1)) = zero;
        *output.index_mut((0, 2)) = zero;
        *output.index_mut((0, 3)) = zero;

        *output.index_mut((1, 0)) = zero;
        *output.index_mut((1, 1)) = zero;
        *output.index_mut((1, 2)) = zero;
        *output.index_mut((1, 3)) = -one;
    }
}

#[inline]
pub(crate) fn outer_bound<const ROTATION: bool, Degrees: Dim, T: ComplexField + Copy>(
    _rotation: Gated<T, ROTATION>,
    _freq: T,
    u: T,
    _v_gamma: T,
    _a_star: T,
    _c1: T,
    ell: T,
    _m: T,
    mut output: MatrixViewMut<
        T,
        <Degrees as DimMul<Const<2>>>::Output,
        <Degrees as DimMul<Const<4>>>::Output,
    >,
) where
    Degrees: DimMul<Const<4>> + DimMul<Const<2>>,
{
    let zero = T::from_subset(&0.);
    let one = T::from_subset(&1.);

    *output.index_mut((0, 0)) = one;
    *output.index_mut((0, 1)) = -one;
    *output.index_mut((0, 2)) = zero;
    *output.index_mut((0, 3)) = zero;

    *output.index_mut((1, 0)) = u;
    *output.index_mut((1, 1)) = zero;
    *output.index_mut((1, 2)) = ell + one;
    *output.index_mut((1, 3)) = one;
}
