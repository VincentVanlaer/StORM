use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, DimAdd, DimMul, DimSub, Matrix, OMatrix,
    RawStorage, RawStorageMut, allocator::Allocator,
};
use num_traits::Zero;

use crate::system::discretized::DiscretizedSystem;

pub(crate) struct UpperResult<T> {
    data: Box<[T]>,
    n: usize,
    n_systems: usize,
    column_pivot: Vec<(usize, usize)>,
}

pub(crate) fn determinant<
    T: ComplexField + Copy,
    System: DiscretizedSystem<T, N: DimMul<Const<2>> + DimAdd<System::NInner>>,
>(
    system: &System,
    frequency: T,
) -> T
where
    DefaultAllocator: DeterminantAllocs<System::N, System::NInner>,
{
    determinant_inner(system, frequency, &mut ())
}

pub(crate) fn determinant_with_upper<
    T: ComplexField + Copy,
    System: DiscretizedSystem<T, N: DimMul<Const<2>> + DimAdd<System::NInner>>,
>(
    system: &System,
    frequency: T,
    upper: &mut UpperResult<T>,
) -> T
where
    DefaultAllocator: DeterminantAllocs<System::N, System::NInner>,
{
    assert_eq!(upper.n, system.shape().value());
    assert_eq!(upper.n_systems, system.len());

    determinant_inner(system, frequency, upper)
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

impl<T: ComplexField + Copy> UpperResult<T> {
    pub(crate) fn new(matrix_size: usize, points: usize) -> UpperResult<T> {
        UpperResult {
            data: vec![
                T::zero();
                points * gauss(matrix_size, 2 * matrix_size - 1) + gauss(1, matrix_size - 1)
            ]
            .into_boxed_slice(),
            n: matrix_size,
            n_systems: points,
            column_pivot: Vec::with_capacity(matrix_size),
        }
    }

    pub(crate) fn eigenvectors(&self) -> Vec<T> {
        let mut eigenvectors = vec![T::zero(); (self.n_systems + 1) * self.n];
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

        for k in 0..self.n_systems {
            for i in 0..self.n {
                let mut next_val = T::zero();
                for j in 0..(i + self.n) {
                    next_val -= *data_iter.next().unwrap()
                        * eigenvectors[eigenvectors.len() - j - k * self.n - 1]
                }
                eigenvectors[len - i - (k + 1) * self.n - 1] = next_val;
            }
        }

        for &(c1, c2) in self.column_pivot.iter().rev() {
            (eigenvectors[c1], eigenvectors[c2]) = (eigenvectors[c2], eigenvectors[c1]);
        }

        eigenvectors
    }
}

trait SetUpperResult<T> {
    fn set(&mut self, point: usize, k: usize, i: usize, val: T);
    fn column_pivot(&mut self, c1: usize, c2: usize);
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

    fn column_pivot(&mut self, c1: usize, c2: usize) {
        self.column_pivot.push((c1, c2));
    }
}

impl<T> SetUpperResult<T> for () {
    fn set(&mut self, _point: usize, _k: usize, _i: usize, _val: T) {}
    fn column_pivot(&mut self, _c1: usize, _c2: usize) {}
}

pub(crate) trait DeterminantAllocs<N: Dim + DimSub<NInner> + DimMul<Const<2>> + DimAdd<NInner>, NInner: Dim> =
    Allocator<N, N>
        + Allocator<NInner, N>
        + Allocator<<N as DimSub<NInner>>::Output, N>
        + Allocator<<N as DimMul<Const<2>>>::Output, <N as DimAdd<NInner>>::Output>
        + Allocator<<N as DimMul<Const<2>>>::Output, Const<1>>;

fn determinant_inner<
    T: ComplexField + Copy,
    System: DiscretizedSystem<T, N: DimMul<Const<2>> + DimAdd<System::NInner>>,
>(
    system: &System,
    frequency: T,
    upper: &mut impl SetUpperResult<T>,
) -> T
where
    DefaultAllocator: DeterminantAllocs<System::N, System::NInner>,
{
    let outer_boundary = {
        let mut outer_boundary = OMatrix::zeros_generic(system.shape_outer(), system.shape());
        system.outer_boundary(frequency, &mut outer_boundary);
        outer_boundary
    };
    let inner_boundary = {
        let mut inner_boundary = OMatrix::zeros_generic(system.shape_inner(), system.shape());
        system.inner_boundary(frequency, &mut inner_boundary);
        inner_boundary
    };

    let n_inner = inner_boundary.shape().0;
    let n = system.shape().value();

    let mut bands = Matrix::from_element_generic(
        system.shape().mul(Const::<2>),
        system.shape().add(inner_boundary.shape_generic().0),
        T::zero(),
    );

    let mut left: OMatrix<T, System::N, System::N> =
        OMatrix::zeros_generic(system.shape(), system.shape());
    let mut right: OMatrix<T, System::N, System::N> =
        OMatrix::zeros_generic(system.shape(), system.shape());

    for i in 0..n_inner {
        for j in 0..n {
            *bands.index_mut((j, i)) = *inner_boundary.index((i, j));
        }
    }

    let mut det = T::one();
    let mut n_step = 0;

    // In order to keep the algebraic equations at the inner boundary local, we need to use column
    // pivotting in the first iteration of the loop
    if n_step != system.len() {
        system.fill(n_step, frequency, &mut left, &mut right);

        for r in 0..n {
            for c in 0..n {
                *bands.index_mut((c, r + n_inner)) = *left.index((r, c));
                *bands.index_mut((c + n, r + n_inner)) = *right.index((r, c));
            }
        }

        for k in 0..n {
            let pivot;

            if k < n_inner {
                let mut max_idx = k;
                let mut max_val: T::RealField = unsafe { bands.get_unchecked((k, k)) }.abs();

                for i in (k + 1)..n {
                    if unsafe { bands.get_unchecked((i, k)) }.abs() > max_val {
                        max_idx = i;
                        max_val = bands.index((i, k)).abs();
                    }
                }

                pivot = unsafe { *bands.get_unchecked((max_idx, k)) };

                if max_idx != k {
                    upper.column_pivot(max_idx, k);
                    for i in k..(n + n_inner) {
                        unsafe {
                            let c1 = *bands.get_unchecked((max_idx, i));
                            let c2 = *bands.get_unchecked((k, i));

                            *bands.get_unchecked_mut((max_idx, i)) = c2;
                            *bands.get_unchecked_mut((k, i)) = c1;
                        }
                    }
                    det *= -T::one();
                }
            } else {
                let mut max_idx = k;
                let mut max_val: T::RealField = unsafe { bands.get_unchecked((k, k)) }.abs();

                for i in (k + 1)..(n + n_inner) {
                    if unsafe { bands.get_unchecked((k, i)) }.abs() > max_val {
                        max_idx = i;
                        max_val = bands.index((k, i)).abs();
                    }
                }

                pivot = unsafe { *bands.get_unchecked((k, max_idx)) };

                if max_idx != k {
                    for i in 0..(2 * n) {
                        unsafe {
                            let r1 = *bands.get_unchecked((i, max_idx));
                            let r2 = *bands.get_unchecked((i, k));

                            *bands.get_unchecked_mut((i, max_idx)) = r2;
                            *bands.get_unchecked_mut((i, k)) = r1;
                        }
                    }
                    det *= -T::one();
                }
            }

            let mut pivot_row =
                Matrix::from_element_generic(system.shape().mul(Const::<2>), Const::<1>, T::zero());

            for i in 0..(2 * n) {
                unsafe {
                    *pivot_row.get_unchecked_mut(i) = *bands.get_unchecked((i, k));
                }
            }

            det *= pivot;

            debug_assert!(
                det.is_finite() && det != T::zero(),
                "det = {det}, pivot = {pivot}, n = {n_step}"
            );

            let pinv = T::one() / pivot;

            for i in (k + 1)..(2 * n) {
                upper.set(n_step, k, i - k, unsafe {
                    *pivot_row.get_unchecked(i) * pinv
                });
            }

            for i in (k + 1)..(n + n_inner) {
                let m = unsafe { *bands.get_unchecked((k, i)) * pinv };
                for j in 0..(2 * n) {
                    // PERF: VFNMADD231PD
                    *unsafe { bands.get_unchecked_mut((j, i)) } -=
                        *unsafe { pivot_row.get_unchecked(j) } * m;
                }
            }
        }

        for i in 0..n_inner {
            for j in 0..n {
                *bands.index_mut((j, i)) = *bands.index((j + n, i + n));
                *bands.index_mut((j + n, i)) = T::zero();
            }
        }

        n_step += 1;
    }

    while n_step < system.len() {
        system.fill(n_step, frequency, &mut left, &mut right);

        for r in 0..n {
            for c in 0..n {
                *bands.index_mut((c, r + n_inner)) = *left.index((r, c));
                *bands.index_mut((c + n, r + n_inner)) = *right.index((r, c));
            }
        }

        for k in 0..n {
            let mut max_idx = k;
            let mut max_val: T::RealField = unsafe { bands.get_unchecked((k, k)) }.abs();

            for i in (k + 1)..(n + n_inner) {
                if unsafe { bands.get_unchecked((k, i)) }.abs() > max_val {
                    max_idx = i;
                    max_val = bands.index((k, i)).abs();
                }
            }

            // PERF: This needs to be loaded first before we rewrite and construct pivot_row. While
            // it is possible to read the pivot element from pivot_row, the compiler will then
            // spill pivot_row to the stack, so it can actually mov the data with the offset k,
            // which it can't do while everything is in registers.
            let pivot = unsafe { *bands.get_unchecked((k, max_idx)) };
            let mut pivot_row =
                Matrix::from_element_generic(system.shape().mul(Const::<2>), Const::<1>, T::zero());

            if max_idx != k {
                for i in 0..(2 * n) {
                    unsafe {
                        *pivot_row.get_unchecked_mut(i) = *bands.get_unchecked((i, max_idx));
                        *bands.get_unchecked_mut((i, max_idx)) = *bands.get_unchecked((i, k));
                    }
                }
                det *= -T::one();
            } else {
                for i in 0..(2 * n) {
                    unsafe {
                        *pivot_row.get_unchecked_mut(i) = *bands.get_unchecked((i, k));
                    }
                }
            }

            det *= pivot;

            debug_assert!(
                pivot.is_finite() && det.is_finite(),
                "det = {det}, pivot = {pivot}, n = {n_step}"
            );

            let pinv = T::one() / pivot;

            for i in (k + 1)..(2 * n) {
                upper.set(n_step, k, i - k, unsafe {
                    *pivot_row.get_unchecked(i) * pinv
                });
            }

            let mut ridx = (k + 1) * 2 * n;

            for _ in (k + 1)..(n + n_inner) {
                let m = *unsafe { bands.data.get_unchecked_linear(ridx + k) } * pinv;
                for j in 0..(2 * n) {
                    // PERF: VFNMADD231PD
                    *unsafe { bands.data.get_unchecked_linear_mut(ridx + j) } -=
                        *unsafe { pivot_row.get_unchecked(j) } * m;
                }

                ridx += 2 * n;
            }
        }

        for i in 0..n_inner {
            for j in 0..n {
                *bands.index_mut((j, i)) = *bands.index((j + n, i + n));
                *bands.index_mut((j + n, i)) = T::zero();
            }
        }

        n_step += 1;
    }

    // Outer boundary
    for r in 0..(n - n_inner) {
        for c in 0..n {
            *bands.index_mut((c, n_inner + r)) = *outer_boundary.index((r, c));
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
            upper.set(n_step, k, i - k, *bands.index((i, k)));
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
