use std::marker::PhantomData;

use nalgebra::{ComplexField, DefaultAllocator, Dim, DimName, DimSub, Dyn, Matrix, StorageMut};

use crate::{
    linalg::storage::{ArrayAllocator, MatrixArray},
    stepper::{ExplicitStepper, ImplicitStepper},
};

use super::{Boundary, BoundedSystem};

pub(crate) struct InterpolatedSystem<
    'a,
    T: ComplexField,
    N: Dim + DimSub<NInner>,
    NInner: Dim,
    Stor,
    Stepper: ImplicitStepper,
    Sys: BoundedSystem<T, N, NInner>,
> {
    scheme: Stepper,
    system: &'a Sys,
    more_data: Vec<Sys::MoreData>,
    matrices: MatrixArray<T, N, N, Dyn, Stor>,
    delta: Vec<f64>,
    _phantom: PhantomData<NInner>,
}

impl<
    'a,
    T: ComplexField,
    N: Dim + DimSub<NInner>,
    NInner: Dim,
    Stepper: ImplicitStepper,
    Sys: BoundedSystem<T, N, NInner>,
>
    InterpolatedSystem<
        'a,
        T,
        N,
        NInner,
        <DefaultAllocator as ArrayAllocator<N, N, Dyn>>::Buffer<T>,
        Stepper,
        Sys,
    >
where
    DefaultAllocator: ArrayAllocator<N, N, Dyn> + ArrayAllocator<N, N, Stepper::Points>,
{
    pub(crate) fn construct(system: &'a Sys, scheme: Stepper) -> Self {
        let n = system.shape();
        let point_locations = scheme.points();

        let mut matrices =
            MatrixArray::new_with(n, n, Dyn(point_locations.len() * system.len()), || {
                T::zero()
            });
        let mut more_data = Vec::with_capacity(point_locations.len() * system.len());
        let mut deltas = Vec::with_capacity(system.len());
        let mut prev_point = None;

        for i in 1..(system.len() + 1) {
            if let Some(prev) = prev_point {
                let next_point = system.pos(i).ln();
                let delta = next_point - prev;
                deltas.push(delta);

                if i == 2 {
                    deltas.push(delta);
                }

                prev_point = Some(next_point)
            } else {
                prev_point = Some(system.pos(i).ln())
            }
        }

        for i in 0..system.len() {
            for j in 0..point_locations.len() {
                let idx = i * point_locations.len() + j;
                more_data.push(system.eval(
                    i,
                    point_locations[j],
                    deltas[i],
                    &mut matrices.index_mut(idx),
                ));
            }
        }

        Self {
            scheme,
            system,
            more_data,
            matrices,
            delta: deltas,
            _phantom: PhantomData,
        }
    }
}

pub(crate) trait DiscretizedSystem<T: ComplexField, N: Dim + nalgebra::DimSub<NInner>, NInner: Dim>:
    Boundary<T, N, NInner>
{
    fn shape(&self) -> N;
    fn len(&self) -> usize;

    fn fill_implicit(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        right: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
    );
}

pub(crate) trait ExplicitDiscretizedSystem<
    T: ComplexField,
    N: Dim + nalgebra::DimSub<NInner>,
    NInner: Dim,
>: DiscretizedSystem<T, N, NInner>
{
    fn fill_explicit(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
    );
}

impl<
    T: ComplexField,
    N: Dim + nalgebra::DimSub<NInner>,
    NInner: Dim,
    Stepper: ImplicitStepper,
    Sys: BoundedSystem<T, N, NInner>,
> Boundary<T, N, NInner>
    for InterpolatedSystem<
        '_,
        T,
        N,
        NInner,
        <DefaultAllocator as ArrayAllocator<N, N, Dyn>>::Buffer<T>,
        Stepper,
        Sys,
    >
where
    DefaultAllocator: ArrayAllocator<N, N, Dyn> + ArrayAllocator<N, N, Stepper::Points>,
{
    fn inner_boundary(
        &self,
        frequency: T,
        output: &mut Matrix<T, NInner, N, impl StorageMut<T, NInner, N>>,
    ) {
        self.system.inner_boundary(frequency, output);
    }

    fn outer_boundary(
        &self,
        frequency: T,
        output: &mut Matrix<
            T,
            <N as nalgebra::DimSub<NInner>>::Output,
            N,
            impl StorageMut<T, <N as nalgebra::DimSub<NInner>>::Output, N>,
        >,
    ) {
        self.system.outer_boundary(frequency, output);
    }

    fn shape_inner(&self) -> NInner {
        self.system.shape_inner()
    }

    fn shape_outer(&self) -> <N as DimSub<NInner>>::Output {
        self.system.shape_outer()
    }
}

impl<
    T: ComplexField,
    N: Dim + nalgebra::DimSub<NInner>,
    NInner: Dim,
    Stepper: ImplicitStepper,
    Sys: BoundedSystem<T, N, NInner>,
> DiscretizedSystem<T, N, NInner>
    for InterpolatedSystem<
        '_,
        T,
        N,
        NInner,
        <DefaultAllocator as ArrayAllocator<N, N, Dyn>>::Buffer<T>,
        Stepper,
        Sys,
    >
where
    DefaultAllocator: ArrayAllocator<N, N, Dyn> + ArrayAllocator<N, N, Stepper::Points>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, N>,
{
    fn shape(&self) -> N {
        self.system.shape()
    }

    fn len(&self) -> usize {
        self.system.len()
    }

    #[inline(always)]
    fn fill_implicit(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        right: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
    ) {
        let points_per_step = Stepper::Points::dim();
        let mut matrix_array = MatrixArray::new_with(
            self.system.shape(),
            self.system.shape(),
            Stepper::Points::name(),
            || T::zero(),
        );

        for i in 0..points_per_step {
            matrix_array
                .index_mut(i)
                .copy_from(&self.matrices.index(step * points_per_step + i));

            self.system.update(
                frequency.clone(),
                &self.more_data[step * points_per_step + i],
                self.delta[step],
                &mut matrix_array.index_mut(i),
            );
        }

        self.scheme.apply(left, right, &matrix_array);
    }
}

impl<
    T: ComplexField,
    N: Dim + nalgebra::DimSub<NInner>,
    NInner: Dim,
    Stepper: ExplicitStepper,
    Sys: BoundedSystem<T, N, NInner>,
> ExplicitDiscretizedSystem<T, N, NInner>
    for InterpolatedSystem<
        '_,
        T,
        N,
        NInner,
        <DefaultAllocator as ArrayAllocator<N, N, Dyn>>::Buffer<T>,
        Stepper,
        Sys,
    >
where
    DefaultAllocator: ArrayAllocator<N, N, Dyn> + ArrayAllocator<N, N, Stepper::Points>,
    DefaultAllocator: nalgebra::allocator::Allocator<N, N>,
{
    #[inline(always)]
    fn fill_explicit(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
    ) {
        let points_per_step = Stepper::Points::dim();
        let mut matrix_array = MatrixArray::new_with(
            self.system.shape(),
            self.system.shape(),
            Stepper::Points::name(),
            || T::zero(),
        );

        for i in 0..points_per_step {
            matrix_array
                .index_mut(i)
                .copy_from(&self.matrices.index(step * points_per_step + i));

            self.system.update(
                frequency.clone(),
                &self.more_data[step * points_per_step + i],
                self.delta[step],
                &mut matrix_array.index_mut(i),
            );
        }

        self.scheme.apply(left, &matrix_array);
    }
}
