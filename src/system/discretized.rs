use nalgebra::{
    ComplexField, DefaultAllocator, Dim, DimName, DimSub, Dyn, Matrix, StorageMut,
    allocator::Allocator,
};

use crate::{
    linalg::storage::{ArrayAllocator, ArrayStorage, MatrixArray},
    model::interpolate::InterpolatingModel,
    stepper::{ExplicitStepper, ImplicitStepper},
};

use super::System;

pub(crate) trait DiscretizedSystem<T: ComplexField> {
    type N: Dim + DimSub<Self::NInner>;
    type NInner: Dim;

    fn len(&self) -> usize;
    fn shape(&self) -> Self::N;
    fn shape_inner(&self) -> Self::NInner;
    fn shape_outer(&self) -> <Self::N as DimSub<Self::NInner>>::Output;

    fn inner_boundary(
        &self,
        frequency: T,
        output: &mut Matrix<T, Self::NInner, Self::N, impl StorageMut<T, Self::NInner, Self::N>>,
    );

    fn outer_boundary(
        &self,
        frequency: T,
        output: &mut Matrix<
            T,
            <Self::N as DimSub<Self::NInner>>::Output,
            Self::N,
            impl StorageMut<T, <Self::N as DimSub<Self::NInner>>::Output, Self::N>,
        >,
    );

    fn fill(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
        right: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    );
}

pub(crate) trait ExplicitDiscretizedSystem<T: ComplexField>: DiscretizedSystem<T> {
    fn fill_explicit(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    );
}

pub(crate) struct DiscretizedSystemImpl<T: ComplexField + Copy, S: System<T>, Stepper, SArray> {
    stepper: Stepper,
    system: S,
    matrices: MatrixArray<T, S::N, S::N, Dyn, SArray>,
    interpolated_points: Vec<S::ModelPoint>,
    inner: S::ModelPoint,
    outer: S::ModelPoint,
    delta: Vec<f64>,
}

impl<T: ComplexField + Copy, Stepper: ImplicitStepper, S: System<T>>
    DiscretizedSystemImpl<
        T,
        S,
        Stepper,
        <DefaultAllocator as ArrayAllocator<S::N, S::N, Dyn>>::Buffer<T>,
    >
where
    DefaultAllocator: ArrayAllocator<S::N, S::N, Dyn>,
    S::ModelPoint: Copy,
{
    pub(crate) fn new(
        model: &impl InterpolatingModel<ModelPoint = impl Into<S::ModelPoint>>,
        stepper: Stepper,
        system: S,
    ) -> Self {
        let n = System::<T>::shape(&system);
        let point_locations = stepper.points();
        let steps = model.len() - 1;
        let points = point_locations.len() * steps;

        let mut matrices = MatrixArray::new_with(n, n, Dyn(points), || T::zero());
        let mut interpolated_points = Vec::with_capacity(points);
        let mut delta = Vec::with_capacity(steps);

        for i in 1..steps {
            let left = model.pos(i).ln();
            let right = model.pos(i + 1).ln();
            delta.push(right - left);

            if i == 1 {
                delta.push(right - left);
            }
        }

        for i in 0..steps {
            for j in 0..point_locations.len() {
                let idx = i * point_locations.len() + j;
                let interpolated_point = model.eval(i, point_locations[j]).into();

                system.eval(interpolated_point, delta[i], &mut matrices.index_mut(idx));

                interpolated_points.push(interpolated_point);
            }
        }

        Self {
            stepper,
            system,
            interpolated_points,
            matrices,
            delta,
            inner: model.eval_exact(0).into(),
            outer: model.eval_exact(steps).into(),
        }
    }
}

impl<
    T: ComplexField + Copy,
    S: System<T>,
    Stepper: ImplicitStepper,
    SArray: ArrayStorage<T, S::N, S::N, Dyn>,
> DiscretizedSystem<T> for DiscretizedSystemImpl<T, S, Stepper, SArray>
where
    DefaultAllocator: ArrayAllocator<S::N, S::N, Stepper::Points> + Allocator<S::N, S::N>,
    S::ModelPoint: Copy,
{
    type N = S::N;
    type NInner = S::NInner;

    fn len(&self) -> usize {
        self.delta.len()
    }

    fn shape(&self) -> Self::N {
        self.system.shape()
    }

    fn shape_inner(&self) -> Self::NInner {
        self.system.shape_inner()
    }

    fn shape_outer(&self) -> <Self::N as DimSub<Self::NInner>>::Output {
        self.system.shape_outer()
    }

    fn inner_boundary(
        &self,
        frequency: T,
        output: &mut Matrix<T, Self::NInner, Self::N, impl StorageMut<T, Self::NInner, Self::N>>,
    ) {
        self.system.inner_boundary(frequency, self.inner, output)
    }

    fn outer_boundary(
        &self,
        frequency: T,
        output: &mut Matrix<
            T,
            <Self::N as DimSub<Self::NInner>>::Output,
            Self::N,
            impl StorageMut<T, <Self::N as DimSub<Self::NInner>>::Output, Self::N>,
        >,
    ) {
        self.system.outer_boundary(frequency, self.outer, output)
    }

    #[inline(always)]
    fn fill(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
        right: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    ) {
        let points_per_step = Stepper::Points::dim();
        let mut matrix_array =
            MatrixArray::new_with(self.shape(), self.shape(), Stepper::Points::name(), || {
                T::zero()
            });

        for i in 0..points_per_step {
            matrix_array
                .index_mut(i)
                .copy_from(&self.matrices.index(step * points_per_step + i));

            self.system.add_frequency(
                frequency,
                self.interpolated_points[step * points_per_step + i],
                self.delta[step],
                &mut matrix_array.index_mut(i),
            );
        }

        self.stepper.apply(left, right, &matrix_array);
    }
}

impl<
    T: ComplexField + Copy,
    S: System<T>,
    Stepper: ExplicitStepper,
    SArray: ArrayStorage<T, S::N, S::N, Dyn>,
> ExplicitDiscretizedSystem<T> for DiscretizedSystemImpl<T, S, Stepper, SArray>
where
    DefaultAllocator: ArrayAllocator<S::N, S::N, Stepper::Points> + Allocator<S::N, S::N>,
    S::ModelPoint: Copy,
{
    #[inline(always)]
    fn fill_explicit(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    ) {
        let points_per_step = Stepper::Points::dim();
        let mut matrix_array =
            MatrixArray::new_with(self.shape(), self.shape(), Stepper::Points::name(), || {
                T::zero()
            });

        for i in 0..points_per_step {
            matrix_array
                .index_mut(i)
                .copy_from(&self.matrices.index(step * points_per_step + i));

            self.system.add_frequency(
                frequency,
                self.interpolated_points[step * points_per_step + i],
                self.delta[step],
                &mut matrix_array.index_mut(i),
            );
        }

        self.stepper.apply(left, &matrix_array);
    }
}
