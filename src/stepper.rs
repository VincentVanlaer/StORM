use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, DimName, Matrix, StorageMut, allocator::Allocator,
};

use crate::linalg::storage::{ArrayStorage, Exp, MatrixArray, assign_matrix, commutator};

pub(crate) trait ImplicitStepper {
    type Points: DimName;

    fn points(&self) -> Vec<f64>;

    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        right: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>;
}

pub(crate) trait ExplicitStepper {
    type Points: DimName;

    fn points(&self) -> Vec<f64>;

    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>;
}

impl<S: ExplicitStepper> ImplicitStepper for S {
    type Points = S::Points;

    fn points(&self) -> Vec<f64> {
        self.points()
    }

    #[inline(always)]
    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        right: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        right.fill_with_identity();

        self.apply(left, values)
    }
}

pub(crate) struct Euler {}

impl ExplicitStepper for Euler {
    type Points = Const<1>;

    fn points(&self) -> Vec<f64> {
        [0.].into()
    }

    #[inline(always)]
    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) {
        left.copy_from(&values.index(0));
    }
}

// Collocation points

// Actual value is 0.288675134594812882254
const C2_1: f64 = -0.288_675_134_594_812_87;
const C2_2: f64 = 0.288_675_134_594_812_87;

const C3_1: f64 = -0.387_298_334_620_741_7;
const C3_2: f64 = 0.;
const C3_3: f64 = 0.387_298_334_620_741_7;

const C4_1: f64 = -0.861_136_311_594_052_6 / 2.;
const C4_2: f64 = -0.339_981_043_584_856_26 / 2.;
const C4_3: f64 = 0.339_981_043_584_856_26 / 2.;
const C4_4: f64 = 0.861_136_311_594_052_6 / 2.;

// General method for deriving these calculations
//
// 1. Start from an n-order Gauss-Legendre collocation method and obtain the Butcher tableau
// 2. Fill in f'(x, y) = A(x)y
// 3. In general, it seems that the equations can always be transformed to
//
//    kᵢ = Aᵢ(y₀ + ∑ⱼcᵢⱼkⱼ) => -Aᵢ(y₀ + y₁) / 2 = kᵢ + Aᵢ∑ⱼ(cᵢⱼ - bⱼ/ 2) k
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

impl ImplicitStepper for Colloc2 {
    type Points = Const<1>;

    fn points(&self) -> Vec<f64> {
        [0.5].into()
    }

    #[inline(always)]
    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        right: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        let b1 = &values.index(0);

        let jump = &(b1 * T::from_subset(&(0.5)));
        let eye = &(Matrix::identity_generic(b1.shape_generic().0, b1.shape_generic().1));

        assign_matrix(left, jump + eye);
        assign_matrix(right, jump - eye);
    }
}

pub(crate) struct Colloc4 {}

impl ImplicitStepper for Colloc4 {
    type Points = Const<2>;

    fn points(&self) -> Vec<f64> {
        [0.5 + C2_1, 0.5 + C2_2].into()
    }

    #[inline(always)]
    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        right: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        let c1 = T::from_subset(&C2_1);
        let c2 = T::from_subset(&C2_2);
        let one_fourth = T::from_subset(&0.25);

        // Collocation points
        let a1 = &values.index(0);
        let a2 = &values.index(1);

        let eye = &(Matrix::identity_generic(a1.shape_generic().0, a1.shape_generic().1));

        let a2a1 = &(a2 * a1);

        // This operation might be more efficient by directly solving Ax = By, rather than
        // computing the inverse of A and the multiplying that with B
        let k2 = &((eye + a2a1 * T::from_subset(&(1. / 12.)))
            .try_inverse()
            .unwrap()
            * (a2 + a2a1 * c2));
        let k1 = &(a1 + a1 * k2 * c1);

        let jump = (k1 + k2) * one_fourth;

        assign_matrix(left, &jump + eye);
        assign_matrix(right, &jump - eye);
    }
}

pub(crate) struct Colloc6 {}

macro_rules! sc {
    ($e: expr) => {
        T::from_subset(&$e)
    };
}

impl ImplicitStepper for Colloc6 {
    type Points = Const<3>;

    fn points(&self) -> Vec<f64> {
        [0.5 + C3_1, 0.5 + C3_2, 0.5 + C3_3].into()
    }

    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        right: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        let d12 = -2. / 3. * C3_1;
        let d13 = -1. / 3. * C3_1;
        let d21 = 5. / 12. * C3_1;
        let d23 = -5. / 12. * C3_1;
        let d31 = 1. / 3. * C3_1;
        let d32 = 2. / 3. * C3_1;

        // Collocation points
        let a1 = &values.index(0);
        let a2 = &values.index(1);
        let a3 = &values.index(2);

        let a2a1 = &(a2 * a1);
        let a3a1 = &(a3 * a1);
        let eye = &(Matrix::identity_generic(a1.shape_generic().0, a1.shape_generic().1));

        let inv1 = &(eye - a2a1 * sc!(d12 * d21)).try_inverse().unwrap();

        let o2 = inv1 * (a2 - a2a1 * sc!(d21));
        let l2 = inv1 * (a2 * sc!(d23) - a2a1 * sc!(d13 * d21));

        let f3 = a3 * sc!(d32) - a3a1 * sc!(d12 * d31);
        let o3 = a3 - a3a1 * sc!(d31) - &f3 * &o2;
        let l3 = eye - a3a1 * sc!(d13 * d31) - &f3 * &l2;

        let k3 = l3.try_inverse().unwrap() * o3;
        let k2 = o2 - l2 * &k3;
        let k1 = a1 * (eye - &k2 * sc!(d12) - &k3 * sc!(d13));

        let jump = k1 * sc!(5. / 36.) + k2 * sc!(2. / 9.) + k3 * sc!(5. / 36.);

        assign_matrix(left, &jump + eye);
        assign_matrix(right, &jump - eye);
    }
}

pub(crate) struct Colloc8 {}

const D4_12: f64 = 1.89640468800635328970119143171e-1;
const D4_13: f64 = 1.5040882602623181114167713212e-1;
const D4_14: f64 = 9.05188609701591475001769691551e-2;

const D4_21: f64 = -1.01154406215504607307419557782e-1;
const D4_23: f64 = 1.90916717318107430880885119114e-1;
const D4_24: f64 = 8.02282106898253088278673182198e-2;

const D4_31: f64 = -8.02282106898253088278673182198e-2;
const D4_32: f64 = -1.90916717318107430880885119114e-1;
const D4_34: f64 = 1.01154406215504607307419557782e-1;

const D4_41: f64 = -9.05188609701591475001769691551e-2;
const D4_42: f64 = -1.5040882602623181114167713212e-1;
const D4_43: f64 = -1.89640468800635328970119143171e-1;

impl ImplicitStepper for Colloc8 {
    type Points = Const<4>;

    fn points(&self) -> Vec<f64> {
        [0.5 + C4_1, 0.5 + C4_2, 0.5 + C4_3, 0.5 + C4_4].into()
    }

    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        right: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        // Collocation points
        let a1 = &values.index(0);
        let a2 = &values.index(1);
        let a3 = &values.index(2);
        let a4 = &values.index(3);

        let a2a1 = &(a2 * a1);
        let a3a1 = &(a3 * a1);
        let a4a1 = &(a4 * a1);
        let eye = &(Matrix::identity_generic(a1.shape_generic().0, a1.shape_generic().1));

        let u12 = a1 * sc!(D4_12);
        let u13 = a1 * sc!(D4_13);
        let u14 = a1 * sc!(D4_14);

        let d2 = eye - a2a1 * sc!(D4_21 * D4_12);
        let inv2 = &d2.try_inverse().unwrap();

        let u23 = inv2 * (a2 * sc!(D4_23) - a2a1 * sc!(D4_21 * D4_13));
        let u24 = inv2 * (a2 * sc!(D4_24) - a2a1 * sc!(D4_21 * D4_14));
        let r2 = inv2 * (a2 - a2a1 * sc!(D4_21));

        let l32 = a3 * sc!(D4_32) - a3a1 * sc!(D4_31 * D4_12);
        let d3 = eye - a3a1 * sc!(D4_31 * D4_13) - &l32 * &u23;
        let inv3 = &d3.try_inverse().unwrap();

        let u34 = inv3 * (a3 * sc!(D4_34) - a3a1 * sc!(D4_31 * D4_14) - &l32 * &u24);
        let r3 = inv3 * (a3 - a3a1 * sc!(D4_31) - &l32 * &r2);

        let l42 = a4 * sc!(D4_42) - a4a1 * sc!(D4_41 * D4_12);
        let l43 = a4 * sc!(D4_43) - a4a1 * sc!(D4_41 * D4_13) - &l42 * &u23;
        let d4 = eye - a4a1 * sc!(D4_41 * D4_14) - &l42 * &u24 - &l43 * &u34;
        let inv4 = &d4.try_inverse().unwrap();

        let r4 = inv4 * (a4 - a4a1 * sc!(D4_41) - l42 * &r2 - l43 * &r3);

        let k4 = r4;
        let k3 = r3 - u34 * &k4;
        let k2 = r2 - u24 * &k4 - u23 * &k3;
        let k1 = a1 - u14 * &k4 - u13 * &k3 - u12 * &k2;

        let jump = k1 * sc!(W1 / 2.) + k2 * sc!(W2 / 2.) + k3 * sc!(W2 / 2.) + k4 * sc!(W1 / 2.);
        assign_matrix(left, &jump + eye);
        assign_matrix(right, &jump - eye);
    }
}

pub(crate) struct Magnus2 {}

impl ExplicitStepper for Magnus2 {
    type Points = Const<1>;

    fn points(&self) -> Vec<f64> {
        [0.5].into()
    }

    #[inline(always)]
    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        let mut omega = values.index(0).clone_owned();

        omega.exp();

        assign_matrix(left, omega);
    }
}

pub(crate) struct Magnus4 {}

impl ExplicitStepper for Magnus4 {
    type Points = Const<2>;

    fn points(&self) -> Vec<f64> {
        [0.5 + C2_1, 0.5 + C2_2].into()
    }

    #[inline(always)]
    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        let a1 = &values.index(0);
        let a2 = &values.index(1);

        let b1 = &((a1 + a2) * T::from_subset(&0.5));
        let b2 = &((a2 - a1) * T::from_subset(&(0.5 * C2_2)));

        assign_matrix(left, b1 - commutator(b1, b2));

        left.exp();
    }
}

pub(crate) struct Magnus6 {}

impl ExplicitStepper for Magnus6 {
    type Points = Const<3>;

    fn points(&self) -> Vec<f64> {
        [0.5 + C3_1, 0.5 + C3_2, 0.5 + C3_3].into()
    }

    #[inline(always)]
    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        let a1 = &values.index(0);
        let a2 = &values.index(1);
        let a3 = &values.index(2);
        let ten_three = T::from_subset(&(10. / 3.));

        let b1 = a2;
        let b2 = &((a3 - a1) * (T::from_subset(&C3_3) * ten_three.clone()));
        let b3 = &((a3 - a2 * T::from_subset(&2.) + a1) * ten_three);

        let c1 = &commutator(b1, b2);
        let c2 = &(commutator(b1, b3 * T::from_subset(&(2.)) + c1) * T::from_subset(&(-1. / 60.)));

        assign_matrix(
            left,
            b1 + b3 * T::from_subset(&(1. / 12.))
                + commutator(b1 * T::from_subset(&(-20.)) - b3 + c1, b2 + c2)
                    * T::from_subset(&(1. / 240.)),
        );

        left.exp();
    }
}

pub(crate) struct Magnus8 {}

const W1: f64 = 0.347_854_845_137_453_85 / 2.;
const W2: f64 = 0.652_145_154_862_546_1 / 2.;

const B1_A1S: f64 = 9. / 4. * W1 - 15. * W1 * C4_1 * C4_1;
const B1_A2S: f64 = 9. / 4. * W2 - 15. * W2 * C4_2 * C4_2;

const B2_A1A: f64 = 75. * W1 * C4_4 - 420. * W1 * C4_4 * C4_4 * C4_4;
const B2_A2A: f64 = 75. * W2 * C4_3 - 420. * W2 * C4_3 * C4_3 * C4_3;

const B3_A1S: f64 = -15. * W1 + 180. * W1 * C4_1 * C4_1;
// const B3_A1S: f64 = -B3_A1S

const B4_A1A: f64 = -420. * W1 * C4_4 + 2800. * W1 * C4_4 * C4_4 * C4_4;
const B4_A2A: f64 = -420. * W2 * C4_3 + 2800. * W2 * C4_3 * C4_3 * C4_3;

impl ExplicitStepper for Magnus8 {
    type Points = Const<4>;

    fn points(&self) -> Vec<f64> {
        [0.5 + C4_1, 0.5 + C4_2, 0.5 + C4_3, 0.5 + C4_4].into()
    }

    #[inline(always)]
    fn apply<T: ComplexField, N: Dim>(
        &self,
        left: &mut Matrix<T, N, N, impl StorageMut<T, N, N>>,
        values: &MatrixArray<T, N, N, Self::Points, impl ArrayStorage<T, N, N, Self::Points>>,
    ) where
        DefaultAllocator: Allocator<N, N>,
    {
        let a1 = &values.index(0);
        let a2 = &values.index(1);
        let a3 = &values.index(2);
        let a4 = &values.index(3);

        let a1s = &(a1 + a4);
        let a1a = &(a4 - a1);
        let a2s = &(a2 + a3);
        let a2a = &(a3 - a2);

        let b1 = &(a1s * T::from_subset(&B1_A1S) + a2s * T::from_subset(&B1_A2S));
        let b2 = &(a1a * T::from_subset(&B2_A1A) + a2a * T::from_subset(&B2_A2A));
        let b3 = &((a1s - a2s) * T::from_subset(&B3_A1S));
        let b4 = &(a1a * T::from_subset(&B4_A1A) + a2a * T::from_subset(&B4_A2A));

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

        assign_matrix(
            left,
            b1 + b3 * T::from_subset(&(1. / 12.))
                + s2 * T::from_subset(&(-7. / 120.))
                + s3 * T::from_subset(&(1. / 360.)),
        );

        left.exp();
    }
}
