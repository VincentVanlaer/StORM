use nalgebra::{ComplexField, Const, Dim, DimMul, Matrix, StorageMut};
use std::mem::MaybeUninit;

use super::{Boundary, System};
use crate::model::StellarModel;

/// Spherically symmetric stellar oscillation equations with a full rotation term
///
/// Note that this system of equations only contains a single spherical harmonic, and will
/// therefore not be completely valid in a rotating star.
pub struct Rotating1D<'model> {
    model: &'model StellarModel,
    ell: f64,
    m: f64,
}

impl<'model> Rotating1D<'model> {
    pub fn from_model(model: &'model StellarModel, ell: u64, m: i64) -> Self {
        Self {
            model,
            ell: ell as f64,
            m: m as f64,
        }
    }
}

impl Boundary<f64, Const<4>, Const<2>> for Rotating1D<'_> {
    fn inner_boundary(
        &self,
        omega: f64,
        output: &mut Matrix<f64, Const<2>, Const<4>, impl StorageMut<f64, Const<2>, Const<4>>>,
    ) {
        let props = self.model.dimensionless_coefficients(0);

        inner_bound::<true, Const<1>, _>(
            Gated::<_, true>::new(self.model.rot[0]),
            omega,
            props.u,
            props.v_gamma,
            props.a_star,
            props.c1,
            self.ell,
            self.m,
            output,
        );
    }

    fn outer_boundary(
        &self,
        omega: f64,
        output: &mut Matrix<f64, Const<2>, Const<4>, impl StorageMut<f64, Const<2>, Const<4>>>,
    ) {
        let props = self
            .model
            .dimensionless_coefficients(self.model.m_coord.len() - 1);

        outer_bound::<true, Const<1>, _>(
            Gated::<_, true>::new(self.model.rot[self.model.m_coord.len() - 1]),
            omega,
            props.u,
            props.v_gamma,
            props.a_star,
            props.c1,
            self.ell,
            self.m,
            output,
        );
    }

    fn shape_inner(&self) -> Const<2> {
        Const
    }

    fn shape_outer(&self) -> <Const<4> as nalgebra::DimSub<Const<2>>>::Output {
        Const
    }
}

impl System<f64, Const<4>> for Rotating1D<'_> {
    type MoreData = (f64, f64);

    fn shape(&self) -> Const<4> {
        Const
    }

    fn len(&self) -> usize {
        self.model.r_coord.len() - 1
    }

    fn eval(
        &self,
        idx: usize,
        pos: f64,
        delta: f64,
        output: &mut Matrix<f64, Const<4>, Const<4>, impl StorageMut<f64, Const<4>, Const<4>>>,
    ) -> Self::MoreData {
        let left = self.model.dimensionless_coefficients(idx);
        let right = self.model.dimensionless_coefficients(idx + 1);

        macro_rules! itp {
            ($e1: expr, $e2: expr) => {
                $e1 + pos * ($e2 - $e1)
            };
        }

        preprocess::<true, Const<1>, _>(
            Gated::<_, true>::new(itp!(self.model.rot[idx], self.model.rot[idx + 1])),
            itp!(left.u, right.u),
            itp!(left.v_gamma, right.v_gamma),
            itp!(left.a_star, right.a_star),
            itp!(left.c1, right.c1),
            self.ell,
            self.m,
            delta,
            output,
        );

        (
            itp!(self.model.rot[idx], self.model.rot[idx + 1]),
            itp!(left.c1, right.c1),
        )
    }

    #[inline(always)]
    fn update(
        &self,
        freq: f64,
        data: &Self::MoreData,
        delta: f64,
        matrix: &mut Matrix<f64, Const<4>, Const<4>, impl StorageMut<f64, Const<4>, Const<4>>>,
    ) {
        process::<true, Const<1>, _>(
            Gated::<_, true>::new(data.0),
            freq,
            data.1,
            self.ell,
            self.m,
            delta,
            matrix,
        );
    }

    fn pos(&self, idx: usize) -> f64 {
        self.model.r_coord[idx] / self.model.radius
    }
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
pub(crate) fn preprocess<
    const ROTATION: bool,
    Degrees: Dim + DimMul<Const<4>>,
    T: ComplexField + Copy,
>(
    _rotation: Gated<T, ROTATION>,
    u: T,
    v_gamma: T,
    a_star: T,
    _c1: T,
    ell: T,
    _m: T,
    delta: T,
    output: &mut Matrix<
        T,
        <Degrees as DimMul<Const<4>>>::Output,
        <Degrees as DimMul<Const<4>>>::Output,
        impl StorageMut<T, <Degrees as DimMul<Const<4>>>::Output, <Degrees as DimMul<Const<4>>>::Output>,
    >,
) {
    assert_eq!(output.shape(), (4, 4));

    let zero = T::from_subset(&0.);
    let one = T::from_subset(&1.);
    let two = T::from_subset(&2.);
    let three = T::from_subset(&3.);
    let lambda = ell * (ell + one);

    *output.index_mut((0, 0)) = delta * (v_gamma - one - ell);
    *output.index_mut((0, 1)) = -delta * v_gamma;
    *output.index_mut((0, 2)) = zero;
    *output.index_mut((0, 3)) = zero;

    *output.index_mut((1, 0)) = -delta * a_star;
    *output.index_mut((1, 1)) = delta * (a_star - u + three - ell);
    *output.index_mut((1, 2)) = zero;
    *output.index_mut((1, 3)) = -delta;

    *output.index_mut((2, 0)) = zero;
    *output.index_mut((2, 1)) = zero;
    *output.index_mut((2, 2)) = delta * (three - u - ell);
    *output.index_mut((2, 3)) = delta;

    *output.index_mut((3, 0)) = delta * (u * a_star);
    *output.index_mut((3, 1)) = delta * (u * v_gamma);
    *output.index_mut((3, 2)) = delta * lambda;
    *output.index_mut((3, 3)) = delta * (-u + two - ell);
}

#[inline(always)]
pub(crate) fn process<
    const ROTATION: bool,
    Degrees: Dim + DimMul<Const<4>>,
    T: ComplexField + Copy,
>(
    rotation: Gated<T, ROTATION>,
    freq: T,
    c1: T,
    ell: T,
    m: T,
    delta: T,
    output: &mut Matrix<
        T,
        <Degrees as DimMul<Const<4>>>::Output,
        <Degrees as DimMul<Const<4>>>::Output,
        impl StorageMut<T, <Degrees as DimMul<Const<4>>>::Output, <Degrees as DimMul<Const<4>>>::Output>,
    >,
) {
    assert_eq!(output.shape(), (4, 4));

    let zero = T::from_subset(&0.);
    let one = T::from_subset(&1.);
    let two = T::from_subset(&2.);
    let lambda = ell * (ell + one);

    let dc1 = delta * c1;
    let dlambda = delta * lambda;

    if ell != zero {
        let rot = g!(rotation, m * rotation, zero);
        let omega_rsq = (lambda * (freq - rot) + two * rot) * (freq - rot);
        let rel_rot = two * rot / (lambda * (freq - rot) + two * rot);

        *output.index_mut((0, 0)) += -dlambda * rel_rot;
        *output.index_mut((0, 1)) += dlambda.powi(2) / (omega_rsq * dc1);
        *output.index_mut((0, 2)) += dlambda.powi(2) / (omega_rsq * dc1);

        *output.index_mut((1, 1)) += dlambda * rel_rot;
        *output.index_mut((1, 2)) += dlambda * rel_rot;
        *output.index_mut((1, 0)) +=
            dc1 * (freq - rot).powi(2) * (one - (two * rot).powi(2) / omega_rsq);
    } else {
        *output.index_mut((1, 0)) += dc1 * freq.powi(2);
    }
}

#[inline]
pub(crate) fn inner_bound<
    const ROTATION: bool,
    Degrees: Dim + DimMul<Const<4>> + DimMul<Const<2>>,
    T: ComplexField + Copy,
>(
    rotation: Gated<T, ROTATION>,
    freq: T,
    _u: T,
    _v_gamma: T,
    _a_star: T,
    c1: T,
    ell: T,
    m: T,
    output: &mut Matrix<
        T,
        <Degrees as DimMul<Const<2>>>::Output,
        <Degrees as DimMul<Const<4>>>::Output,
        impl StorageMut<T, <Degrees as DimMul<Const<2>>>::Output, <Degrees as DimMul<Const<4>>>::Output>,
    >,
) {
    let zero = T::from_subset(&0.);
    let one = T::from_subset(&1.);
    let two = T::from_subset(&2.);

    if ell != zero {
        let rot = g!(rotation, m * rotation, zero);
        let omega2 = (freq - rot + two * rot / ell) * (freq - rot);

        *output.index_mut((0, 0)) = c1 * omega2;
    } else {
        *output.index_mut((0, 0)) = c1 * freq * freq;
    }

    *output.index_mut((0, 1)) = -ell;
    *output.index_mut((0, 2)) = -ell;
    *output.index_mut((0, 3)) = zero;

    *output.index_mut((1, 0)) = zero;
    *output.index_mut((1, 1)) = zero;
    *output.index_mut((1, 2)) = ell;
    *output.index_mut((1, 3)) = -one;
}

#[inline]
pub(crate) fn outer_bound<
    const ROTATION: bool,
    Degrees: Dim + DimMul<Const<4>> + DimMul<Const<2>>,
    T: ComplexField + Copy,
>(
    _rotation: Gated<T, ROTATION>,
    _freq: T,
    _u: T,
    _v_gamma: T,
    _a_star: T,
    _c1: T,
    ell: T,
    _m: T,
    output: &mut Matrix<
        T,
        <Degrees as DimMul<Const<2>>>::Output,
        <Degrees as DimMul<Const<4>>>::Output,
        impl StorageMut<T, <Degrees as DimMul<Const<2>>>::Output, <Degrees as DimMul<Const<4>>>::Output>,
    >,
) {
    let zero = T::from_subset(&0.);
    let one = T::from_subset(&1.);

    *output.index_mut((0, 0)) = one;
    *output.index_mut((0, 1)) = -one;
    *output.index_mut((0, 2)) = zero;
    *output.index_mut((0, 3)) = zero;

    *output.index_mut((1, 0)) = zero;
    *output.index_mut((1, 1)) = zero;
    *output.index_mut((1, 2)) = ell + one;
    *output.index_mut((1, 3)) = one;
}
