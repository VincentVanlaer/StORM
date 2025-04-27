use nalgebra::{ComplexField, Const, Matrix, StorageMut};

use crate::model::DimensionlessProperties;

use super::System;

/// Spherically symmetric stellar oscillation equations with a full rotation term
///
/// Note that this system of equations only contains a single spherical harmonic, and will
/// therefore not be completely valid in a rotating star.
#[derive(Debug, Clone, Copy)]
pub struct Rotating1D {
    ell: f64,
    m: f64,
}

impl Rotating1D {
    /// Create a new [Rotating1D] with the given spherical degree (`ell`) and azimuthal order (`m`)
    pub fn new(ell: u64, m: i64) -> Self {
        Self {
            ell: ell as f64,
            m: m as f64,
        }
    }
}

impl<T: ComplexField + Copy> System<T> for Rotating1D {
    type ModelPoint = DimensionlessProperties;
    type N = Const<4>;
    type NInner = Const<2>;

    fn shape(&self) -> Self::N {
        Const
    }

    fn shape_inner(&self) -> Self::NInner {
        Const
    }

    fn shape_outer(&self) -> <Self::N as nalgebra::DimSub<Self::NInner>>::Output {
        Const
    }

    fn eval(
        &self,
        point: Self::ModelPoint,
        delta: f64,
        output: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    ) {
        let zero = T::from_subset(&0.);
        let one = T::from_subset(&1.);
        let two = T::from_subset(&2.);
        let three = T::from_subset(&3.);
        let ell = T::from_subset(&self.ell);
        let delta = T::from_subset(&delta);
        let v_gamma = T::from_subset(&point.v_gamma);
        let u = T::from_subset(&point.u);
        let a_star = T::from_subset(&point.a_star);
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
    fn add_frequency(
        &self,
        freq: T,
        point: Self::ModelPoint,
        delta: f64,
        output: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    ) {
        let zero = T::from_subset(&0.);
        let one = T::from_subset(&1.);
        let two = T::from_subset(&2.);
        let ell = T::from_subset(&self.ell);
        let delta = T::from_subset(&delta);
        let rot = T::from_subset(&point.rot);
        let c1 = T::from_subset(&point.c1);
        let lambda = ell * (ell + one);

        let dc1 = delta * c1;
        let dlambda = delta * lambda;

        if ell != zero {
            let rot = T::from_subset(&self.m) * rot;
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

    fn inner_boundary(
        &self,
        frequency: T,
        inner_point: Self::ModelPoint,
        output: &mut Matrix<T, Self::NInner, Self::N, impl StorageMut<T, Self::NInner, Self::N>>,
    ) {
        let zero = T::from_subset(&0.);
        let one = T::from_subset(&1.);
        let two = T::from_subset(&2.);
        let ell = T::from_subset(&self.ell);
        let c1 = T::from_subset(&inner_point.c1);

        if ell != zero {
            let rot = T::from_subset(&(self.m * inner_point.rot));
            let omega2 = (frequency - rot + two * rot / ell) * (frequency - rot);

            *output.index_mut((0, 0)) = c1 * omega2;
        } else {
            *output.index_mut((0, 0)) = c1 * frequency * frequency;
        }

        *output.index_mut((0, 1)) = -ell;
        *output.index_mut((0, 2)) = -ell;
        *output.index_mut((0, 3)) = zero;

        *output.index_mut((1, 0)) = zero;
        *output.index_mut((1, 1)) = zero;
        *output.index_mut((1, 2)) = ell;
        *output.index_mut((1, 3)) = -one;
    }

    fn outer_boundary(
        &self,
        _frequency: T,
        _outer_point: Self::ModelPoint,
        output: &mut Matrix<
            T,
            <Self::N as nalgebra::DimSub<Self::NInner>>::Output,
            Self::N,
            impl StorageMut<T, <Self::N as nalgebra::DimSub<Self::NInner>>::Output, Self::N>,
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
        *output.index_mut((1, 2)) = T::from_subset(&self.ell) + one;
        *output.index_mut((1, 3)) = one;
    }
}
