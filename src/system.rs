//! Systems of equations, interface and implementations
use nalgebra::{ComplexField, Dim, DimSub, Matrix, StorageMut};

pub(crate) trait System<T: ComplexField> {
    type ModelPoint;
    type N: Dim + nalgebra::DimSub<Self::NInner>;
    type NInner: Dim;

    fn shape(&self) -> Self::N;
    fn shape_inner(&self) -> Self::NInner;
    fn shape_outer(&self) -> <Self::N as DimSub<Self::NInner>>::Output;

    fn eval(
        &self,
        point: Self::ModelPoint,
        delta: f64,
        output: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    );

    fn add_frequency(
        &self,
        freq: T,
        point: Self::ModelPoint,
        delta: f64,
        matrix: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    );

    fn inner_boundary(
        &self,
        frequency: T,
        inner_point: Self::ModelPoint,
        output: &mut Matrix<T, Self::NInner, Self::N, impl StorageMut<T, Self::NInner, Self::N>>,
    );

    fn outer_boundary(
        &self,
        frequency: T,
        outer_point: Self::ModelPoint,
        output: &mut Matrix<
            T,
            <Self::N as DimSub<Self::NInner>>::Output,
            Self::N,
            impl StorageMut<T, <Self::N as DimSub<Self::NInner>>::Output, Self::N>,
        >,
    );
}

/// Adiabatic stellar oscillation equations
pub mod adiabatic;

pub(crate) mod discretized;
