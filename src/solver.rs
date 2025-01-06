use nalgebra::{ComplexField, Const, DefaultAllocator, Dim, DimName, Matrix, Scalar};
use num::{One, Zero};

use crate::{
    linalg::{ArrayAllocator, MatrixArray},
    stepper::{Step, Stepper},
    system::System,
};

pub(crate) struct UpperResult<T> {
    data: Box<[T]>,
    n: usize,
    n_systems: usize,
}

pub(crate) fn determinant<
    T: DeterminantField,
    N: Dim + nalgebra::DimSub<NInner> + nalgebra::DimMul<Const<2>> + nalgebra::DimAdd<NInner>,
    NInner: Dim,
    Order: DimName,
    G: ?Sized,
    I: System<T, G, N, NInner, Order>,
    S: Stepper<T, N, Order>,
>(
    system: &I,
    stepper: &S,
    grid: &G,
    frequency: T,
) -> T
where
    DefaultAllocator: DeterminantAllocs<N, NInner, Order>,
{
    determinant_inner(system, stepper, grid, frequency, &mut ())
}

pub(crate) fn determinant_with_upper<
    T: DeterminantField,
    N: Dim + nalgebra::DimSub<NInner> + nalgebra::DimMul<Const<2>> + nalgebra::DimAdd<NInner>,
    NInner: Dim,
    Order: DimName,
    G: ?Sized,
    I: System<T, G, N, NInner, Order>,
    S: Stepper<T, N, Order>,
>(
    system: &I,
    stepper: &S,
    grid: &G,
    frequency: T,
    upper: &mut UpperResult<T>,
) -> T
where
    DefaultAllocator: DeterminantAllocs<N, NInner, Order>,
{
    assert_eq!(upper.n, system.shape().value());
    assert_eq!(upper.n_systems, system.len(grid));

    determinant_inner(system, stepper, grid, frequency, upper)
}

fn gauss(lower: usize, upper: usize) -> usize {
    (upper - lower + 1) * (upper + lower) / 2
}

// Data storage of UpperResult and backwards substitution example. The different indices i, j, and
// k refer to the variables in UpperResult::eigenvectors. The goal is to determine the indices in
// the top row, as those give which element of the eigenvector needs to be multiplied with the
// upper block triangular matrix. The data is indexed row by row, starting from the top left. Note
// that this is *back* substitution so the direction of this iterator is reversed.
//
// 6 5 4 3 2 1 0 9 8 7 6 5 4 3 2 1
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ i   k                j
// 1 * * * * * * *                 ┃ 3 ┓    * 6 5 4 3 2 1 0
// 0 1 * * * * * *                 ┃ 2 ┣ 2    * 5 4 3 2 1 0
// 0 0 1 * * * * *                 ┃ 1 ┃        * 4 3 2 1 0
// 0 0 0 1 * * * *                 ┃ 0 ┛          * 3 2 1 0
// 0 0 0 0 1 * * * * * * *         ┃ 3 ┓            * 6 5 4 3 2 1 0
// 0 0 0 0 0 1 * * * * * *         ┃ 2 ┣ 1            * 5 4 3 2 1 0
//         0 0 1 * * * * *         ┃ 1 ┃                * 4 3 2 1 0
//         0 0 0 1 * * * *         ┃ 0 ┛                  * 3 2 1 0
//         0 0 0 0 1 * * * * * * * ┃ 3 ┓                    * 6 5 4 3 2 1 0
//         0 0 0 0 0 1 * * * * * * ┃ 2 ┣ 0                    * 5 4 3 2 1 0
//                 0 0 1 * * * * * ┃ 1 ┃                        * 4 3 2 1 0
//                 0 0 0 1 * * * * ┃ 0 ┛                          * 3 2 1 0
//                 0 0 0 0 1 * * * ┃ 3 ┓                            * 2 1 0
//                 0 0 0 0 0 1 * * ┃ 2 ┣ special case                 * 1 0
//                         0 0 1 * ┃ 1 ┃ has less data to the           * 0
//                         0 0 0 ^ ┃ 0 ┛ right due to edge of matrix      *

impl<T: DeterminantField> UpperResult<T> {
    pub(crate) fn new(matrix_size: usize, points: usize) -> UpperResult<T> {
        UpperResult {
            data: vec![
                T::zero();
                points * gauss(matrix_size, 2 * matrix_size - 1) + gauss(1, matrix_size - 1)
            ]
            .into_boxed_slice(),
            n: matrix_size,
            n_systems: points,
        }
    }

    pub(crate) fn eigenvectors(&self) -> Vec<T> {
        let mut eigenvectors = vec![T::zero(); self.n_systems * self.n];
        let len = eigenvectors.len();

        eigenvectors[len - 1] = T::one();
        let mut data_iter = self.data.iter().rev();

        for i in 1..self.n {
            let mut next_val = T::zero();
            for j in 0..i {
                next_val -= *data_iter.next().unwrap() * eigenvectors[eigenvectors.len() - j - 1]
            }
            eigenvectors[len - i - 1] = next_val;
        }

        for k in 0..(self.n_systems - 1) {
            for i in 0..self.n {
                let mut next_val = T::zero();
                for j in 0..(i + self.n) {
                    next_val -= *data_iter.next().unwrap()
                        * eigenvectors[eigenvectors.len() - j - k * self.n - 1]
                }
                eigenvectors[len - i - (k + 1) * self.n - 1] = next_val;
            }
        }

        eigenvectors
    }
}

trait SetUpperResult<T> {
    fn set(&mut self, point: usize, k: usize, i: usize, val: T);
}

impl<T> SetUpperResult<T> for UpperResult<T> {
    fn set(&mut self, point: usize, k: usize, i: usize, val: T) {
        if point != self.n_systems {
            self.data[point * gauss(self.n, 2 * self.n - 1)
                + gauss(2 * self.n - 1 - k, 2 * self.n - 1)
                - (2 * self.n - 1 - k)
                + (i - 1)] = val;
        } else {
            self.data[point * gauss(self.n, 2 * self.n - 1) + gauss(self.n - 1 - k, self.n - 1)
                - (self.n - 1 - k)
                + (i - 1)] = val;
        }
    }
}

impl<T> SetUpperResult<T> for () {
    fn set(&mut self, _point: usize, _k: usize, _i: usize, _val: T) {}
}

pub(crate) trait DeterminantField = ComplexField<RealField: Zero> + Scalar + One + Zero + Copy;

pub(crate) trait DeterminantAllocs<
    N: Dim + nalgebra::DimSub<NInner> + nalgebra::DimMul<Const<2>> + nalgebra::DimAdd<NInner>,
    NInner: Dim,
    Order: DimName,
> = ArrayAllocator<N, N, Order>
    + ArrayAllocator<N, N, Const<2>>
    + nalgebra::allocator::Allocator<N, NInner>
    + nalgebra::allocator::Allocator<N, <N as nalgebra::DimSub<NInner>>::Output>
    + nalgebra::allocator::Allocator<
        <N as nalgebra::DimMul<Const<2>>>::Output,
        <N as nalgebra::DimAdd<NInner>>::Output,
    > + nalgebra::allocator::Allocator<<N as nalgebra::DimMul<Const<2>>>::Output, Const<1>>;

fn determinant_inner<
    T: DeterminantField,
    N: Dim + nalgebra::DimSub<NInner> + nalgebra::DimMul<Const<2>> + nalgebra::DimAdd<NInner>,
    NInner: Dim,
    Order: DimName,
    G: ?Sized,
    I: System<T, G, N, NInner, Order>,
    S: Stepper<T, N, Order>,
>(
    system: &I,
    stepper: &S,
    grid: &G,
    frequency: T,
    upper: &mut impl SetUpperResult<T>,
) -> T
where
    DefaultAllocator: DeterminantAllocs<N, NInner, Order>,
{
    let iterator = system.evaluate_moments(grid, frequency);
    let total_steps = iterator.len();
    assert_eq!(total_steps, system.len(grid));
    let outer_boundary = system.outer_boundary(frequency);
    let inner_boundary = system.inner_boundary(frequency);

    let n_inner = inner_boundary.shape().1;
    let n = system.shape().value();

    let mut bands = Matrix::from_element_generic(
        system.shape().mul(Const::<2>),
        system.shape().add(inner_boundary.shape_generic().1),
        T::zero(),
    );

    let mut step = Step::new(MatrixArray::new_with(
        system.shape(),
        system.shape(),
        Const::<2>,
        || T::zero(),
    ));

    for i in 0..n_inner {
        for j in 0..n {
            *bands.index_mut((j, i)) = *inner_boundary.index((j, i));
        }
    }

    let mut det = T::one();
    for (n_step, moments) in iterator.enumerate() {
        stepper.step(moments, &mut step);
        assert!(n_step < total_steps);
        for r in 0..n {
            for c in 0..n {
                *bands.index_mut((c, r + 2)) = *step.left().index((r, c));
                *bands.index_mut((c + n, r + 2)) = *step.right().index((r, c));
            }
        }

        for k in 0..n {
            let mut max_idx = 0;
            let mut max_val: T::RealField = T::RealField::zero();

            for i in k..(n + n_inner) {
                if bands.index((k, i)).abs() > max_val {
                    max_idx = i;
                    max_val = bands.index((k, i)).abs();
                }
            }

            let pivot;

            if max_idx != k {
                pivot = bands.column(max_idx).into_owned();
                bands.set_column(max_idx, &bands.column(k).into_owned());
                det *= -T::one();
            } else {
                pivot = bands.column(k).into_owned();
            }

            det *= pivot[k];

            debug_assert!(det.is_finite());

            for i in (k + 1)..(2 * n) {
                upper.set(n_step, k, i - k, pivot[i] / pivot[k]);
            }

            for i in (k + 1)..(n + n_inner) {
                let m = *bands.index((k, i)) / pivot[k];
                for j in 0..(2 * n) {
                    *bands.index_mut((j, i)) -= pivot[j] * m;
                }
            }
        }

        for i in 0..n_inner {
            for j in 0..n {
                *bands.index_mut((j, i)) = *bands.index((j + n, i + n));
                *bands.index_mut((j + n, i)) = T::zero();
            }
        }
    }

    // Outer boundary
    for r in 0..(n - n_inner) {
        for c in 0..n {
            *bands.index_mut((c, n_inner + r)) = *outer_boundary.index((c, r));
        }
    }

    for k in 0..(n - 1) {
        let mut max_idx = 0;
        let mut max_val: T::RealField = T::RealField::zero();

        for i in k..n {
            if bands.index((k, i)).abs() > max_val {
                max_idx = i;
                max_val = bands.index((k, i)).abs();
            }
        }

        if max_idx != k {
            bands.swap_columns(max_idx, k);
            det *= -T::one();
        }

        let pivot = *bands.index((k, k));

        for j in 0..n {
            *bands.index_mut((j, k)) /= pivot;
        }

        for i in (k + 1)..n {
            upper.set(total_steps, k, i - k, *bands.index((i, k)));
        }

        debug_assert!(pivot.is_finite(), "det = {det}, pivot = {pivot}");

        det *= pivot;

        for i in (k + 1)..n {
            let m = *bands.index((k, i));
            for j in 0..n {
                let res = *bands.index((j, k));
                *bands.index_mut((j, i)) -= res * m;
            }
        }
    }

    det * *bands.index((n - 1, n - 1))
}
