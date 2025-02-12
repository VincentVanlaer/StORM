use nalgebra::{
    allocator::Allocator, ComplexField, Const, DefaultAllocator, Dim, Matrix, Scalar, ViewStorage,
};

use crate::linalg::{assign_matrix, commutator, ArrayStorage, Exp, MatrixArray};

pub(crate) struct StepMoments<T, N, Order, S> {
    pub moments: MatrixArray<T, N, N, Order, S>,
}

pub(crate) struct Step<T, N, S> {
    steps: MatrixArray<T, N, N, Const<2>, S>,
}

impl<T: Scalar, N: Dim, S: ArrayStorage<T, N, N, Const<2>>> Step<T, N, S> {
    pub(crate) fn new(steps: MatrixArray<T, N, N, Const<2>, S>) -> Self {
        Step { steps }
    }

    pub(crate) fn left(&self) -> Matrix<T, N, N, ViewStorage<T, N, N, Const<1>, N>> {
        self.steps.index(0)
    }

    pub(crate) fn right(&self) -> Matrix<T, N, N, ViewStorage<T, N, N, Const<1>, N>> {
        self.steps.index(1)
    }
}

pub(crate) trait Stepper<T, N, Order> {
    fn step<S1: ArrayStorage<T, N, N, Order>, S2: ArrayStorage<T, N, N, Const<2>>>(
        &self,
        step_input: StepMoments<T, N, Order, S1>,
        step: &mut Step<T, N, S2>,
    );
}

pub(crate) struct Magnus2 {}

impl<T: ComplexField, N: Dim> Stepper<T, N, Const<1>> for Magnus2
where
    DefaultAllocator: Allocator<N, N>,
{
    fn step<S1: ArrayStorage<T, N, N, Const<1>>, S2: ArrayStorage<T, N, N, Const<2>>>(
        &self,
        mut step_input: StepMoments<T, N, Const<1>, S1>,
        step: &mut Step<T, N, S2>,
    ) {
        let mut omega = step_input.moments.index_mut(0);

        omega.exp();

        assign_matrix(&mut step.steps.index_mut(0), omega);
        step.steps.index_mut(1).fill_diagonal(T::from_subset(&-1.));
    }
}

pub(crate) struct Magnus4 {}

impl<T: ComplexField, N: Dim> Stepper<T, N, Const<2>> for Magnus4
where
    DefaultAllocator: Allocator<N, N>,
{
    fn step<S1: ArrayStorage<T, N, N, Const<2>>, S2: ArrayStorage<T, N, N, Const<2>>>(
        &self,
        step_input: StepMoments<T, N, Const<2>, S1>,
        step: &mut Step<T, N, S2>,
    ) {
        let b1 = &step_input.moments.index(0);
        let b2 = &step_input.moments.index(1);

        let mut omega = b1 - commutator(b1, b2) * T::from_subset(&(1.0 / 12.0));

        omega.exp();

        assign_matrix(&mut step.steps.index_mut(0), omega);
        step.steps.index_mut(1).fill_diagonal(T::from_subset(&-1.));
    }
}

pub(crate) struct Magnus6 {}

impl<T: ComplexField, N: Dim> Stepper<T, N, Const<3>> for Magnus6
where
    DefaultAllocator: Allocator<N, N>,
{
    fn step<S1: ArrayStorage<T, N, N, Const<3>>, S2: ArrayStorage<T, N, N, Const<2>>>(
        &self,
        step_input: StepMoments<T, N, Const<3>, S1>,
        step: &mut Step<T, N, S2>,
    ) {
        let b1 = &step_input.moments.index(0);
        let b2 = &step_input.moments.index(1);
        let b3 = &step_input.moments.index(2);

        let c1 = &commutator(b1, b2);
        let c2 = &(commutator(b1, b3 * T::from_subset(&(2.)) + c1) * T::from_subset(&(-1. / 60.)));

        let mut omega = b1
            + b3 * T::from_subset(&(1. / 12.))
            + commutator(b1 * T::from_subset(&(-20.)) - b3 + c1, b2 + c2)
                * T::from_subset(&(1. / 240.));

        omega.exp();

        assign_matrix(&mut step.steps.index_mut(0), omega);
        step.steps.index_mut(1).fill_diagonal(T::from_subset(&-1.));
    }
}

pub(crate) struct Magnus8 {}

impl<T: ComplexField, N: Dim> Stepper<T, N, Const<4>> for Magnus8
where
    DefaultAllocator: Allocator<N, N>,
{
    fn step<S1: ArrayStorage<T, N, N, Const<4>>, S2: ArrayStorage<T, N, N, Const<2>>>(
        &self,
        step_input: StepMoments<T, N, Const<4>, S1>,
        step: &mut Step<T, N, S2>,
    ) {
        let b1 = &step_input.moments.index(0);
        let b2 = &step_input.moments.index(1);
        let b3 = &step_input.moments.index(2);
        let b4 = &step_input.moments.index(2);

        let s1 = &(commutator(
            b1 + b3 * T::from_subset(&(1. / 28.)),
            b2 + b4 * T::from_subset(&(3. / 28.)),
        ) * T::from_subset(&(-1. / 28.)));
        let r1 =
            &(commutator(b1, b3 * T::from_subset(&(-1. / 14.)) + s1) * T::from_subset(&(1. / 3.)));
        let s2 = &commutator(
            b1 + b3 * T::from_subset(&(1. / 28.)) + s1,
            b2 + b4 * T::from_subset(&(3. / 28.)) + r1,
        );
        let s2_prime = &commutator(b2, s1);
        let r2 = &commutator(
            b1 + s1 * T::from_subset(&(5. / 4.)),
            b3 * T::from_subset(&(2.0)) + s2 + s2_prime * T::from_subset(&(0.5)),
        );
        let s3 = &(commutator(
            b1 + b3 * T::from_subset(&(1. / 12.))
                + s1 * T::from_subset(&(-7. / 3.))
                + s2 * T::from_subset(&(-1. / 6.)),
            b2 * T::from_subset(&(-9.))
                + b4 * T::from_subset(&(-9. / 4.))
                + r1 * T::from_subset(&(63.))
                + r2,
        ));

        let mut omega = b1
            + b3 * T::from_subset(&(1. / 12.))
            + s2 * T::from_subset(&(-7. / 120.))
            + s3 * T::from_subset(&(1. / 360.));

        omega.exp();

        assign_matrix(&mut step.steps.index_mut(0), omega);
        step.steps.index_mut(1).fill_diagonal(T::from_subset(&-1.));
    }
}

// Collocation points

// Actual value is 0.288675134594812882254
const C2_1: f64 = -0.288_675_134_594_812_87;
const C2_2: f64 = 0.288_675_134_594_812_87;

// General method for deriving these calculations
//
// 1. Start from an n-order Gauss-Legendre collocation method and obtain the Butcher tableau
// 2. Fill in f'(x, y) = A(x)y
// 3. In general, it seems that the equations can always be transformed to
//
//    kᵢ = Aᵢ(y₀ + ∑ⱼcᵢⱼkⱼ) => -Aᵢ(y₀ + y₁) / 2 = Aᵢ(kᵢ + ∑ⱼ(cᵢⱼ - bⱼ/ 2) kⱼ)
//      where cᵢᵢ - bᵢ/ 2 = 0
//
//    NOTE that all kᵢ are vectors with the same dimensions as A(x), and Aᵢ≡ hA(xᵢ)
//
// 4. This gives a system of equations that looks like this in terms of structure (excluding the
//    c and b constants)
//
//    │𝟙  A₁ A₁││k₁│  │A₁│
//    │A₂ 𝟙  A₂││k₂│= │A₂│(y₀ + y₁) / 2
//    │A₃ A₃ 𝟙 ││k₃│  │A₃│
//
//    NOTE that this is a matrix of matrices. From the view of solving this system we take the A
//    matrices as constants
//
// 5. Solve the system of equations for kᵢ using gaussian elimination. This will require n - 1 matrix
//    inversions.
// 6. Fill the result into the final equation of the collocation step. This gives us an implicit
//    linear equation for y₀ and y₁, which is what we need. In principle it is possible to solve
//    for y₁ at this point, but that is not necessary
//
// See also:
//   https://en.wikipedia.org/wiki/Runge-Kutta_methods
//   https://en.wikipedia.org/wiki/Gauss-Legendre_method

pub(crate) struct Colloc2 {}

impl<T: ComplexField, N: Dim> Stepper<T, N, Const<1>> for Colloc2
where
    DefaultAllocator: Allocator<N, N>,
{
    #[inline(always)]
    fn step<S1: ArrayStorage<T, N, N, Const<1>>, S2: ArrayStorage<T, N, N, Const<2>>>(
        &self,
        step_input: StepMoments<T, N, Const<1>, S1>,
        step: &mut Step<T, N, S2>,
    ) {
        let b1 = &step_input.moments.index(0);

        let jump = &(b1 * T::from_subset(&(0.5)));
        let eye = &(Matrix::identity_generic(b1.shape_generic().0, b1.shape_generic().1));

        assign_matrix(&mut step.steps.index_mut(0), jump + eye);
        assign_matrix(&mut step.steps.index_mut(1), jump - eye);
    }
}

pub(crate) struct Colloc4 {}

impl<T: ComplexField, N: Dim> Stepper<T, N, Const<2>> for Colloc4
where
    DefaultAllocator: Allocator<N, N>,
{
    #[inline(always)]
    fn step<S1: ArrayStorage<T, N, N, Const<2>>, S2: ArrayStorage<T, N, N, Const<2>>>(
        &self,
        step_input: StepMoments<T, N, Const<2>, S1>,
        step: &mut Step<T, N, S2>,
    ) {
        let c1 = T::from_subset(&C2_1);
        let c2 = T::from_subset(&C2_2);
        let one_fourth = T::from_subset(&0.25);

        // Moments
        let m1 = &step_input.moments.index(0);
        let m2 = &step_input.moments.index(1);

        // Collocation points
        let a1 = &(m1 + m2 * c1.clone());
        let a2 = &(m1 + m2 * c2.clone());

        let eye = &(Matrix::identity_generic(m1.shape_generic().0, m1.shape_generic().1));

        let a2a1 = &(a2 * a1);

        // This operation might be more efficient by directly solving Ax = By, rather than
        // computing the inverse of A and the multiplying that with B
        let k2 = &((eye + a2a1 * T::from_subset(&(1. / 12.)))
            .try_inverse()
            .unwrap()
            * (a2 + a2a1 * c2));
        let k1 = &(a1 + a1 * k2 * c1);

        let jump = (k1 + k2) * one_fourth;

        assign_matrix(&mut step.steps.index_mut(0), &jump + eye);
        assign_matrix(&mut step.steps.index_mut(1), &jump - eye);
    }
}
