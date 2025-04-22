use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, DimName, DimSub, Dyn, Matrix, StorageMut,
};

use crate::{
    linalg::storage::{ArrayAllocator, ArrayStorage, MatrixArray, UnsizedMatrixArray},
    model::{DimensionlessProperties, Model},
    stepper::{ExplicitStepper, ImplicitStepper},
};

use super::{System, adiabatic::Rotating1D};

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

pub(crate) struct DiscretizedRotating1D<T: ComplexField + Copy, Stepper, SArray> {
    stepper: Stepper,
    system: Rotating1D,
    matrices:
        MatrixArray<T, <Rotating1D as System<T>>::N, <Rotating1D as System<T>>::N, Dyn, SArray>,
    interpolated_points: Vec<<Rotating1D as System<T>>::ModelPoint>,
    inner: <Rotating1D as System<T>>::ModelPoint,
    outer: <Rotating1D as System<T>>::ModelPoint,
    delta: Vec<f64>,
}

impl<T: ComplexField + Copy, Stepper: ImplicitStepper>
    DiscretizedRotating1D<T, Stepper, UnsizedMatrixArray<T, Const<4>, Const<4>, Dyn>>
{
    pub(crate) fn new(model: &impl Model, stepper: Stepper, system: Rotating1D) -> Self {
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
            let left = model.dimensionless_properties(i);
            let right = model.dimensionless_properties(i + 1);

            for j in 0..point_locations.len() {
                let idx = i * point_locations.len() + j;
                let interpolated_point = interpolate_linear(left, right, point_locations[j]);

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
            inner: model.dimensionless_properties(0),
            outer: model.dimensionless_properties(steps),
        }
    }
}

fn interpolate_linear(
    point1: DimensionlessProperties,
    point2: DimensionlessProperties,
    x: f64,
) -> DimensionlessProperties {
    macro_rules! interp {
        ($e: ident) => {
            point1.$e + x * (point2.$e - point1.$e)
        };
    }

    DimensionlessProperties {
        v_gamma: interp!(v_gamma),
        a_star: interp!(a_star),
        u: interp!(u),
        c1: interp!(c1),
        rot: interp!(rot),
    }
}

impl<
    T: ComplexField + Copy,
    Stepper: ImplicitStepper,
    SArray: ArrayStorage<T, Const<4>, Const<4>, Dyn>,
> DiscretizedSystem<T> for DiscretizedRotating1D<T, Stepper, SArray>
where
    DefaultAllocator: ArrayAllocator<Const<4>, Const<4>, Stepper::Points>,
{
    type N = <Rotating1D as System<T>>::N;
    type NInner = <Rotating1D as System<T>>::NInner;

    fn len(&self) -> usize {
        self.delta.len()
    }

    fn shape(&self) -> Self::N {
        System::<T>::shape(&self.system)
    }

    fn shape_inner(&self) -> Self::NInner {
        System::<T>::shape_inner(&self.system)
    }

    fn shape_outer(&self) -> <Self::N as DimSub<Self::NInner>>::Output {
        System::<T>::shape_outer(&self.system)
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

    fn fill(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
        right: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    ) {
        let points_per_step = Stepper::Points::dim();
        let mut matrix_array = MatrixArray::new_with(
            System::<T>::shape(&self.system),
            System::<T>::shape(&self.system),
            Stepper::Points::name(),
            || T::zero(),
        );

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
    Stepper: ExplicitStepper,
    SArray: ArrayStorage<T, Const<4>, Const<4>, Dyn>,
> ExplicitDiscretizedSystem<T> for DiscretizedRotating1D<T, Stepper, SArray>
where
    DefaultAllocator: ArrayAllocator<Const<4>, Const<4>, Stepper::Points>,
{
    fn fill_explicit(
        &self,
        step: usize,
        frequency: T,
        left: &mut Matrix<T, Self::N, Self::N, impl StorageMut<T, Self::N, Self::N>>,
    ) {
        let points_per_step = Stepper::Points::dim();
        let mut matrix_array = MatrixArray::new_with(
            System::<T>::shape(&self.system),
            System::<T>::shape(&self.system),
            Stepper::Points::name(),
            || T::zero(),
        );

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
