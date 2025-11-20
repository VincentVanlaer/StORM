use itertools::Itertools;
use nalgebra::{
    ComplexField, DefaultAllocator, Dim, DimName, DimSub, Dyn, Matrix, StorageMut,
    allocator::Allocator,
};

use crate::{
    linalg::storage::{ArrayAllocator, ArrayStorage, MatrixArray},
    model::{ContinuousModel, DiscreteModelSegment},
    stepper::ImplicitStepper,
};

use super::System;

pub(crate) trait DiscretizedSystem<T: ComplexField> {
    type N: Dim + DimSub<Self::NInner>;
    type NInner: Dim;

    fn len_segments(&self) -> usize;
    fn len_steps(&self, segment: usize) -> usize;
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

    fn transition_segment(
        &self,
        segment: usize,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
        right: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    );

    fn fill(
        &self,
        segment: usize,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
        right: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    );
}

pub(crate) struct DiscretizedSystemImpl<T: ComplexField + Copy, S: System<T>, Stepper, SArray> {
    stepper: Stepper,
    system: S,
    segments: Vec<DiscretizedSegment<T, S, SArray>>,
}

struct DiscretizedSegment<T: ComplexField + Copy, S: System<T>, SArray> {
    matrices: MatrixArray<T, S::N, S::N, Dyn, SArray>,
    interpolated_points: Vec<S::ModelPoint>,
    inner: S::ModelPoint,
    outer: S::ModelPoint,
    delta: Vec<f64>,
    points: Vec<f64>,
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
    Vec<S::ModelPoint>: for<'a> From<&'a DiscreteModelSegment>,
{
    pub(crate) fn new(
        model: &(impl ContinuousModel + ?Sized),
        stepper: Stepper,
        system: S,
        solving_grid: &[&[f64]],
    ) -> Self {
        let n = System::<T>::shape(&system);
        let point_locations = stepper.points();
        let segments = model.segments();

        assert_eq!(segments.len(), solving_grid.len());

        let mut discretized_segments = Vec::new();

        for (idx, grid) in solving_grid.iter().enumerate() {
            let steps = grid.len() - 1;
            let points = point_locations.len() * steps;

            let mut matrices = MatrixArray::new_with(n, n, Dyn(points), || T::zero());
            let delta: Vec<_> = grid
                .iter()
                .tuple_windows()
                .map(|(x1, x2)| x2 - x1)
                .collect();
            let xs: Vec<_> = grid
                .iter()
                .zip(delta.iter())
                .flat_map(|(&x, &delta)| point_locations.iter().map(move |&p| x + p * delta))
                .collect();

            let interpolated_points: Vec<S::ModelPoint> = (&model.eval(idx, &xs)).into();

            for i in 0..steps {
                for j in 0..point_locations.len() {
                    let index = i * point_locations.len() + j;
                    system.eval(
                        interpolated_points[index],
                        delta[i] / xs[index],
                        &mut matrices.index_mut(index),
                    );
                }
            }

            discretized_segments.push(DiscretizedSegment {
                matrices,
                interpolated_points,
                inner: Into::<Vec<S::ModelPoint>>::into(
                    &model.eval(idx, &[model.segments()[idx].lower]),
                )[0],
                outer: Into::<Vec<S::ModelPoint>>::into(
                    &model.eval(idx, &[model.segments()[idx].upper]),
                )[0],
                delta,
                points: xs,
            })
        }

        Self {
            stepper,
            system,
            segments: discretized_segments,
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

    fn len_segments(&self) -> usize {
        self.segments.len()
    }

    fn len_steps(&self, segment: usize) -> usize {
        self.segments[segment].delta.len()
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
        self.system
            .inner_boundary(frequency, self.segments[0].inner, output)
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
        self.system
            .outer_boundary(frequency, self.segments.last().unwrap().outer, output)
    }

    fn transition_segment(
        &self,
        segment: usize,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
        right: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    ) {
        self.system.double_point(
            self.segments[segment].outer,
            self.segments[segment + 1].inner,
            left,
            right,
        );
    }

    #[inline(always)]
    fn fill(
        &self,
        segment: usize,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
        right: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    ) {
        let segment = &self.segments[segment];
        let points_per_step = Stepper::Points::dim();
        let mut matrix_array =
            MatrixArray::new_with(self.shape(), self.shape(), Stepper::Points::name(), || {
                T::zero()
            });

        for i in 0..points_per_step {
            matrix_array
                .index_mut(i)
                .copy_from(&segment.matrices.index(step * points_per_step + i));

            self.system.add_frequency(
                frequency,
                segment.interpolated_points[step * points_per_step + i],
                segment.delta[step] / segment.points[step * points_per_step + i],
                &mut matrix_array.index_mut(i),
            );
        }

        self.stepper.apply(left, right, &matrix_array);
    }
}
