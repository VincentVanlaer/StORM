use nalgebra::{
    ComplexField, Const, DefaultAllocator, Dim, DimAdd, DimMul, DimSub, Matrix, OMatrix,
    RawStorage, RawStorageMut, allocator::Allocator,
};

use crate::system::discretized::{DiscretizedSystem, ExplicitDiscretizedSystem};

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
    DefaultAllocator: Allocator<System::N, System::N>
        + Allocator<System::NInner, System::N>
        + Allocator<<System::N as DimSub<System::NInner>>::Output, System::N>
        + Allocator<
            <System::N as DimMul<Const<2>>>::Output,
            <System::N as DimAdd<System::NInner>>::Output,
        > + Allocator<<System::N as DimMul<Const<2>>>::Output, Const<1>>,
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
    DefaultAllocator: Allocator<System::N, System::N>
        + Allocator<System::NInner, System::N>
        + Allocator<<System::N as DimSub<System::NInner>>::Output, System::N>
        + Allocator<
            <System::N as DimMul<Const<2>>>::Output,
            <System::N as DimAdd<System::NInner>>::Output,
        > + Allocator<<System::N as DimMul<Const<2>>>::Output, Const<1>>,
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

macro_rules! sweep {
    ($bands: ident, $upper: ident, $n_step: ident, $rows: expr, $cols: expr, $idx: ident, $pivot: ident, $pivot_row: ident) => {
        let pinv = T::one() / $pivot;

        for i in ($idx + 1)..$cols {
            $upper.set($n_step, $idx, i - $idx, unsafe {
                *$pivot_row.get_unchecked(i) * pinv
            });
        }

        sweep!($bands, $rows, $cols, $idx, $pivot, $pivot_row);
    };
    ($bands: ident, $rows: expr, $cols: expr, $idx: ident, $pivot: ident, $pivot_row: ident) => {
        let pinv = T::one() / $pivot;
        let mut ridx = ($idx + 1) * $bands.shape().0;

        for _ in ($idx + 1)..$rows {
            let m = *unsafe { $bands.data.get_unchecked_linear(ridx + $idx) } * pinv;
            for j in 0..$cols {
                // PERF: VFNMADD231PD
                *unsafe { $bands.data.get_unchecked_linear_mut(ridx + j) } -=
                    *unsafe { $pivot_row.get_unchecked(j) } * m;
            }

            ridx += $bands.shape().0;
        }
    };
}

macro_rules! column_pivot {
    ($bands: ident, $upper: ident, $det: ident, $rows: expr, $cols: expr, $idx: ident) => {{
        let mut max_idx = $idx;
        let mut max_val: T::RealField = unsafe { $bands.get_unchecked(($idx, $idx)) }.abs();

        for i in ($idx + 1)..$cols {
            if unsafe { $bands.get_unchecked((i, $idx)) }.abs() > max_val {
                max_idx = i;
                max_val = $bands.index((i, $idx)).abs();
            }
        }

        let pivot = unsafe { *$bands.get_unchecked((max_idx, $idx)) };
        let mut pivot_row =
            Matrix::from_element_generic($bands.shape_generic().0, Const::<1>, T::zero());

        if max_idx != $idx {
            $upper.column_pivot(max_idx, $idx);
            for i in $idx..$rows {
                unsafe {
                    let c1 = *$bands.get_unchecked((max_idx, i));
                    let c2 = *$bands.get_unchecked(($idx, i));

                    *$bands.get_unchecked_mut((max_idx, i)) = c2;
                    *$bands.get_unchecked_mut(($idx, i)) = c1;
                }
            }
            $det *= -T::one();
        }

        for i in 0..$cols {
            unsafe {
                *pivot_row.get_unchecked_mut(i) = *$bands.get_unchecked((i, $idx));
            }
        }

        (pivot, pivot_row)
    }};
}
macro_rules! row_pivot {
    ($bands: ident, $det: ident, $rows: expr, $cols: expr, $idx: ident) => {{
        let mut max_idx = $idx;
        let mut max_val: T::RealField = unsafe { $bands.get_unchecked(($idx, $idx)) }.abs();

        for i in ($idx + 1)..$rows {
            if unsafe { $bands.get_unchecked(($idx, i)) }.abs() > max_val {
                max_idx = i;
                max_val = $bands.index(($idx, i)).abs();
            }
        }

        // PERF: This needs to be loaded first before we rewrite and construct pivot_row. While
        // it is possible to read the pivot element from pivot_row, the compiler will then
        // spill pivot_row to the stack, so it can actually mov the data with the offset k,
        // which it can't do while everything is in registers.
        let pivot = unsafe { *$bands.get_unchecked(($idx, max_idx)) };
        let mut pivot_row =
            Matrix::from_element_generic($bands.shape_generic().0, Const::<1>, T::zero());

        if max_idx != $idx {
            for i in 0..$cols {
                unsafe {
                    *pivot_row.get_unchecked_mut(i) = *$bands.get_unchecked((i, max_idx));
                    *$bands.get_unchecked_mut((i, max_idx)) = *$bands.get_unchecked((i, $idx));
                }
            }
            $det *= -T::one();
        } else {
            for i in 0..$cols {
                unsafe {
                    *pivot_row.get_unchecked_mut(i) = *$bands.get_unchecked((i, $idx));
                }
            }
        }

        (pivot, pivot_row)
    }};
}

fn determinant_inner<
    T: ComplexField + Copy,
    System: DiscretizedSystem<T, N: DimMul<Const<2>> + DimAdd<System::NInner>>,
>(
    system: &System,
    frequency: T,
    upper: &mut impl SetUpperResult<T>,
) -> T
where
    DefaultAllocator: Allocator<System::N, System::N>
        + Allocator<System::NInner, System::N>
        + Allocator<<System::N as DimSub<System::NInner>>::Output, System::N>
        + Allocator<
            <System::N as DimMul<Const<2>>>::Output,
            <System::N as DimAdd<System::NInner>>::Output,
        > + Allocator<<System::N as DimMul<Const<2>>>::Output, Const<1>>,
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
            let (pivot, pivot_row) = if k < n_inner {
                column_pivot!(bands, upper, det, n + n_inner, 2 * n, k)
            } else {
                row_pivot!(bands, det, n + n_inner, 2 * n, k)
            };

            det *= pivot;

            debug_assert!(
                det.is_finite() && det != T::zero(),
                "det = {det}, pivot = {pivot}, n = {n_step}"
            );

            sweep!(
                bands,
                upper,
                n_step,
                n + n_inner,
                2 * n,
                k,
                pivot,
                pivot_row
            );
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
            let (pivot, pivot_row) = row_pivot!(bands, det, n + n_inner, 2 * n, k);

            det *= pivot;

            debug_assert!(
                pivot.is_finite() && det.is_finite(),
                "det = {det}, pivot = {pivot}, n = {n_step}"
            );

            sweep!(
                bands,
                upper,
                n_step,
                n + n_inner,
                2 * n,
                k,
                pivot,
                pivot_row
            );
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
        let (pivot, pivot_row) = row_pivot!(bands, det, n, n, k);

        det *= pivot;

        debug_assert!(
            pivot.is_finite() && det.is_finite(),
            "det = {det}, pivot = {pivot}, n = {n_step}"
        );

        sweep!(bands, upper, n_step, n, n, k, pivot, pivot_row);
    }

    det * *bands.index((n - 1, n - 1))
}

pub(crate) fn determinant_explicit<
    T: ComplexField + Copy,
    System: ExplicitDiscretizedSystem<T, N: DimMul<Const<2>> + DimAdd<System::NInner>>,
>(
    system: &System,
    frequency: T,
) -> T
where
    DefaultAllocator: Allocator<System::N, System::N>
        + Allocator<System::NInner, System::N>
        + Allocator<<System::N as DimSub<System::NInner>>::Output, System::N>
        + Allocator<
            <System::N as DimMul<Const<2>>>::Output,
            <System::N as DimAdd<System::NInner>>::Output,
        > + Allocator<<System::N as DimMul<Const<2>>>::Output, Const<1>>,
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
    let mut accum: OMatrix<T, System::N, System::N> =
        OMatrix::identity_generic(system.shape(), system.shape());
    let mut accum2: OMatrix<T, System::N, System::N> =
        OMatrix::zeros_generic(system.shape(), system.shape());

    for i in 0..n_inner {
        for j in 0..n {
            *bands.index_mut((j, i)) = *inner_boundary.index((i, j));
        }
    }

    let mut det = T::one();

    for step in 0..system.len() {
        system.fill_explicit(step, frequency, &mut left);

        accum2.clone_from(&accum);

        accum.gemm(T::one(), &left, &accum2, T::zero());
    }

    for r in 0..n {
        for c in 0..n {
            *bands.index_mut((c, r + n_inner)) = *accum.index((r, c));
            *bands.index_mut((c + n, r + n_inner)) = if r == c { -T::one() } else { T::zero() };
        }
    }

    for k in 0..n {
        let (pivot, pivot_row) = row_pivot!(bands, det, n + n_inner, 2 * n, k);

        det *= pivot;

        debug_assert!(
            det.is_finite() && det != T::zero(),
            "det = {det}, pivot = {pivot}"
        );

        sweep!(bands, n + n_inner, 2 * n, k, pivot, pivot_row);
    }

    for i in 0..n_inner {
        for j in 0..n {
            *bands.index_mut((j, i)) = *bands.index((j + n, i + n));
            *bands.index_mut((j + n, i)) = T::zero();
        }
    }

    // Outer boundary
    for r in 0..(n - n_inner) {
        for c in 0..n {
            *bands.index_mut((c, n_inner + r)) = *outer_boundary.index((r, c));
        }
    }

    for k in 0..(n - 1) {
        let (pivot, pivot_row) = row_pivot!(bands, det, n, n, k);

        det *= pivot;

        debug_assert!(
            pivot.is_finite() && det.is_finite(),
            "det = {det}, pivot = {pivot}"
        );

        sweep!(bands, n, n, k, pivot, pivot_row);
    }

    det * *bands.index((n - 1, n - 1))
}
