//! Systems of equations, interface and implementations
use nalgebra::{ComplexField, Dim, DimSub, Matrix, StorageMut};

pub(crate) trait Boundary<T: ComplexField, N: Dim + nalgebra::DimSub<NInner>, NInner: Dim> {
    fn shape_inner(&self) -> NInner;
    fn shape_outer(&self) -> <N as DimSub<NInner>>::Output;

    fn inner_boundary(
        &self,
        frequency: T,
        output: &mut Matrix<T, NInner, N, impl StorageMut<T, NInner, N>>,
    );
    fn outer_boundary(
        &self,
        frequency: T,
        output: &mut Matrix<
            T,
            <N as DimSub<NInner>>::Output,
            N,
            impl StorageMut<T, <N as DimSub<NInner>>::Output, N>,
        >,
    );
}

pub(crate) trait System<T: ComplexField, N: Dim> {
    type MoreData;

    fn shape(&self) -> N;
    fn len(&self) -> usize;
    fn pos(&self, idx: usize) -> f64;

    fn eval(
        &self,
        idx: usize,
        pos: f64,
        delta: f64,
        output: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
    ) -> Self::MoreData;

    fn update(
        &self,
        freq: T,
        data: &Self::MoreData,
        delta: f64,
        matrix: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
    );
}

#[expect(private_bounds)]
pub trait BoundedSystem<T: ComplexField, N: Dim + nalgebra::DimSub<NInner>, NInner: Dim>:
    Boundary<T, N, NInner> + System<T, N>
{
}

impl<
    T: ComplexField,
    N: Dim + nalgebra::DimSub<NInner>,
    NInner: Dim,
    S: Boundary<T, N, NInner> + System<T, N>,
> BoundedSystem<T, N, NInner> for S
{
}

/// Adiabatic stellar oscillation equations
pub mod adiabatic;

pub(crate) mod filler;
