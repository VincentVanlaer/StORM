//! Systems of equations, interface and implementations
use crate::{
    linalg::storage::{ArrayAllocator, OwnedArray},
    stepper::StepMoments,
};
use nalgebra::{DefaultAllocator, Dim, DimName, DimSub, Field, OMatrix, Scalar};

pub(crate) trait Moments<T: Field + Scalar, G: ?Sized, N: Dim, Order: DimName>
where
    DefaultAllocator: ArrayAllocator<N, N, Order>,
{
    fn evaluate_moments(
        &self,
        grid: &G,
        frequency: T,
    ) -> impl ExactSizeIterator<Item = StepMoments<T, N, Order, OwnedArray<T, N, N, Order>>>;

    fn shape(&self) -> N;
}

pub(crate) trait GridLength<G: ?Sized> {
    fn len(&self, grid: &G) -> usize;
}

pub(crate) trait Boundary<T: Field + Scalar, N: Dim + nalgebra::DimSub<NInner>, NInner: Dim>
where
    DefaultAllocator: nalgebra::allocator::Allocator<NInner, N>,
    DefaultAllocator: nalgebra::allocator::Allocator<<N as DimSub<NInner>>::Output, N>,
{
    fn inner_boundary(&self, frequency: T) -> OMatrix<T, NInner, N>;
    fn outer_boundary(&self, _frequency: T) -> OMatrix<T, <N as DimSub<NInner>>::Output, N>;
}

#[expect(private_bounds)]
/// Represents a system of equations with boundary conditions.
///
/// This trait is still a bit in flux, and hence the parent traits are private, making it not
/// possible for downstream crates to implement or call any methods on this trait. In the future,
/// this trait may become fully public.
///
/// This trait has many generic parameters, to allow for as much optimizations as possible. The
/// generic parameters mean the following:
///
/// - `T`: the base numeric type, e.g. f64
/// - `G`: the gridding method. This needs to be a generic method to allow for full inlining of the
///   main loop
/// - `N`: number of equations per point in the system of equations, used for loop unrolling
/// - `N_INNER`: number of inner boundary conditioms
/// - `ORDER`: order of stepping method. The lower the order, the less information needs to be
///   computed.
pub trait System<
    T: Field + Scalar,
    G: ?Sized,
    N: Dim + nalgebra::DimSub<NInner>,
    NInner: Dim,
    Order: DimName,
>: Moments<T, G, N, Order> + Boundary<T, N, NInner> + GridLength<G> where
    DefaultAllocator: ArrayAllocator<N, N, Order>,
    DefaultAllocator: nalgebra::allocator::Allocator<NInner, N>,
    DefaultAllocator: nalgebra::allocator::Allocator<<N as DimSub<NInner>>::Output, N>,
{
}

impl<
    T: Field + Scalar,
    G: ?Sized,
    N: Dim + nalgebra::DimSub<NInner>,
    NInner: Dim,
    Order: DimName,
    U,
> System<T, G, N, NInner, Order> for U
where
    U: Moments<T, G, N, Order> + Boundary<T, N, NInner> + GridLength<G>,
    DefaultAllocator: ArrayAllocator<N, N, Order>,
    DefaultAllocator: nalgebra::allocator::Allocator<NInner, N>,
    DefaultAllocator: nalgebra::allocator::Allocator<<N as DimSub<NInner>>::Output, N>,
{
}

/// Adiabatic stellar oscillation equations
pub mod adiabatic;
