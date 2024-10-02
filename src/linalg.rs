use blas::dgemm;
use lapack::dgeev;
use std::{
    mem::{self, transmute_copy},
    ops::{Add, AddAssign, DivAssign, Index, IndexMut, Mul, Sub, SubAssign},
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut},
};

use num::{Float, One, Zero};

#[derive(Copy, Clone, Debug)]
#[repr(align(64))]
pub(crate) struct Matrix<T, const ROWS: usize, const COLUMNS: usize> {
    pub data: [[T; ROWS]; COLUMNS],
}

impl<T, const N: usize, const M: usize> Matrix<T, N, M> {
    pub(crate) fn as_slice(&mut self) -> &[T] {
        let slice = slice_from_raw_parts(self.data.as_ptr() as *const T, N * M);
        unsafe { &*slice }
    }

    pub(crate) fn as_slice_mut(&mut self) -> &mut [T] {
        let slice = slice_from_raw_parts_mut(self.data.as_mut_ptr() as *mut T, N * M);
        unsafe { &mut *slice }
    }
}

impl<T: Zero + One + Copy, const N: usize> Matrix<T, N, N> {
    pub(crate) fn eye() -> Matrix<T, N, N> {
        let mut data = [[T::zero(); N]; N];

        for i in 0..N {
            data[i][i] = T::one();
        }

        data.into()
    }
}

pub(crate) trait Matmul<T> {
    fn matmul(self, rhs: T) -> T;
}

impl<T: Zero + Copy + Mul<Output = T> + AddAssign, const N: usize> Matmul<Matrix<T, N, N>>
    for Matrix<T, N, N>
{
    #[inline(always)]
    fn matmul(self, rhs: Matrix<T, N, N>) -> Matrix<T, N, N> {
        let mut result = Matrix {
            data: [[T::zero(); N]; N],
        };

        for i in 0..N {
            for j in 0..N {
                let scalar = rhs[i][j];

                for k in 0..N {
                    result[i][k] += scalar * self[j][k];
                }
            }
        }

        result
    }
}

impl<T: Float + SubAssign + DivAssign + Copy, const N: usize> Matrix<T, N, N> {
    #[inline(always)]
    pub(crate) fn inv(mut self) -> Result<Matrix<T, N, N>, i32> {
        let mut inv = Self::eye();

        for i in 0..N {
            let mut max_idx = 0;
            let mut max_val = T::zero();

            for j in i..N {
                if self.data[j][i].abs() > max_val.abs() {
                    max_idx = j;
                    max_val = self.data[j][i];
                }
            }

            if max_idx != i {
                (self.data[i], self.data[max_idx]) = (self.data[max_idx], self.data[i]);
                (inv[i], inv[max_idx]) = (inv[max_idx], inv[i]);
            }

            for j in 0..N {
                self.data[i][j] /= max_val;
                inv[i][j] /= max_val;
            }

            for j in (i + 1)..N {
                let m = self.data[j][i];
                for k in 0..N {
                    self.data[j][k] -= self.data[i][k] * m;
                    inv.data[j][k] -= inv.data[i][k] * m;
                }
            }
        }

        for i in 0..N {
            for j in (i + 1)..N {
                let m = self.data[N - j - 1][N - i - 1];
                for k in 0..N {
                    self.data[N - j - 1][k] -= self.data[N - i - 1][k] * m;
                    inv.data[N - j - 1][k] -= inv.data[N - i - 1][k] * m;
                }
            }
        }

        Ok(inv)
    }
}

pub(crate) fn commutator<T: Copy, const N: usize>(
    a: Matrix<T, N, N>,
    b: Matrix<T, N, N>,
) -> Matrix<T, N, N>
where
    Matrix<T, N, N>: Matmul<Matrix<T, N, N>> + Sub<Matrix<T, N, N>, Output = Matrix<T, N, N>>,
{
    a.matmul(b) - b.matmul(a)
}

impl<T, const ROWS: usize, const COLUMNS: usize> Index<usize> for Matrix<T, ROWS, COLUMNS> {
    type Output = [T; ROWS];

    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> IndexMut<usize> for Matrix<T, ROWS, COLUMNS> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Copy + Zero, const N: usize, const M: usize> Add<Matrix<T, N, M>> for Matrix<T, N, M>
where
    T: Add<T, Output = T>,
{
    type Output = Matrix<T, N, M>;

    #[inline(always)]
    fn add(self, rhs: Matrix<T, N, M>) -> Self::Output {
        let mut result = Matrix {
            data: [[T::zero(); N]; M],
        };

        for i in 0..M {
            for j in 0..N {
                result[i][j] = self[i][j] + rhs[i][j];
            }
        }

        result
    }
}

impl<T: Copy + Zero, const N: usize, const M: usize> Sub<Matrix<T, N, M>> for Matrix<T, N, M>
where
    T: Sub<T, Output = T>,
{
    type Output = Matrix<T, N, M>;

    #[inline(always)]
    fn sub(self, rhs: Matrix<T, N, M>) -> Self::Output {
        let mut result = Matrix {
            data: [[T::zero(); N]; M],
        };

        for i in 0..M {
            for j in 0..N {
                result[i][j] = self[i][j] - rhs[i][j];
            }
        }

        result
    }
}

impl<
        T: Copy + Zero + Mul<C, Output = T> + Mul<T, Output = T>,
        C: Copy,
        const N: usize,
        const M: usize,
    > Mul<C> for Matrix<T, N, M>
{
    type Output = Matrix<T, N, M>;

    fn mul(self, rhs: C) -> Self::Output {
        let mut result = Matrix {
            data: [[T::zero(); N]; M],
        };

        for i in 0..M {
            for j in 0..N {
                result[i][j] = self[i][j] * rhs;
            }
        }

        result
    }
}

impl<const N: usize> Matrix<f64, N, N> {
    pub(crate) fn exp(&mut self, step: f64) {
        let lapack_n: i32 = N.try_into().unwrap();
        let mut wr = [0.0; N];
        let mut wi = [0.0; N];
        let mut vl = Matrix {
            data: [[0.0; N]; N],
        };
        let mut vr = Matrix {
            data: [[0.0; N]; N],
        };
        let mut info = 0;
        let mut work = Matrix {
            data: [[0.0; 4]; N],
        }; // work = 4 * SIZE(A, 1)

        unsafe {
            dgeev(
                b'V',
                b'V',
                lapack_n,
                self.as_slice_mut(),
                lapack_n,
                &mut wr,
                &mut wi,
                vl.as_slice_mut(),
                lapack_n,
                vr.as_slice_mut(),
                lapack_n,
                work.as_slice_mut(),
                4 * lapack_n,
                &mut info,
            )
        };

        if info != 0 {
            panic!("dgeev failed")
        }

        let mut i_iter = 0..N;

        while let Some(i) = i_iter.next() {
            if wi[i] != 0.0 {
                let mut sum = Complex64::new(0.0, 0.0);
                for j in 0..N {
                    let a = Complex64::new(vr[i][j], vr[i + 1][j]);
                    let b = Complex64::new(vl[i][j], -vl[i + 1][j]);
                    sum += a * b;
                }

                for j in 0..N {
                    let mut b = Complex64::new(vl[i][j], -vl[i + 1][j]);
                    b /= sum;
                    vl[i][j] = b.re;
                    vl[i + 1][j] = b.im;
                }
                i_iter.next();
            } else {
                let mut sum = 0.;
                for j in 0..N {
                    sum += vl[i][j] * vr[i][j];
                }

                for j in 0..N {
                    vl[i][j] /= sum;
                }
            }
        }

        let mut i_iter = 0..N;

        while let Some(i) = i_iter.next() {
            if wi[i] != 0.0 {
                let eig = Complex64::new(wr[i] * step, wi[i] * step).exp();

                for j in 0..N {
                    let v = eig * Complex64::new(vl[i][j], vl[i + 1][j]);

                    vl[i][j] = 2. * v.re;
                    vl[i + 1][j] = -2. * v.im;
                }

                i_iter.next();
            } else {
                for j in 0..N {
                    vl[i][j] *= (step * wr[i]).exp();
                }
            }
        }

        unsafe {
            dgemm(
                b'N',
                b'T',
                lapack_n,
                lapack_n,
                lapack_n,
                1.,
                vr.as_slice(),
                lapack_n,
                vl.as_slice(),
                lapack_n,
                0.,
                self.as_slice_mut(),
                lapack_n,
            )
        }
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> From<[T; ROWS * COLUMNS]>
    for Matrix<T, ROWS, COLUMNS>
{
    fn from(data: [T; ROWS * COLUMNS]) -> Self {
        let res = Self {
            data: unsafe { transmute_copy(&data) },
        };

        mem::forget(data);

        res
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> From<[[T; ROWS]; COLUMNS]>
    for Matrix<T, ROWS, COLUMNS>
{
    fn from(data: [[T; ROWS]; COLUMNS]) -> Self {
        Self { data }
    }
}

impl<T, const ROWS: usize, const COLUMNS: usize> From<Matrix<T, ROWS, COLUMNS>>
    for [T; ROWS * COLUMNS]
{
    fn from(m: Matrix<T, ROWS, COLUMNS>) -> Self {
        let res = unsafe { transmute_copy(&m.data) };

        mem::forget(m);

        res
    }
}

#[cfg(test)]
mod test_inv {
    use crate::linalg::Matrix;

    fn compare_floats(f1: f64, f2: f64) -> bool {
        u64::abs_diff(f1.to_bits(), f2.to_bits()) <= 2
    }

    fn compare_matrices<const N: usize, const M: usize>(
        m1: Matrix<f64, N, M>,
        m2: Matrix<f64, N, M>,
    ) {
        for i in 0..N {
            for j in 0..M {
                if !compare_floats(m1[i][j], m2[i][j]) {
                    panic!("Matrices do not match\n{m1:?}\n{m2:?}");
                }
            }
        }
    }

    #[test]
    fn inverse_eye() {
        compare_matrices(
            Matrix::<f64, 4, 4>::eye(),
            Matrix::<f64, 4, 4>::eye().inv().unwrap(),
        )
    }

    #[test]
    fn inverse_2x2() {
        compare_matrices(
            Matrix::<_, 2, 2>::from([[2., 2.], [3., 2.]]),
            Matrix::<_, 2, 2>::from([[-1., 1.], [3. / 2., -1.]])
                .inv()
                .unwrap(),
        )
    }

    #[test]
    fn inverse_4x4() {
        compare_matrices(
            Matrix::<_, 4, 4>::from([
                [1., 2., 0., 0.],
                [0., 1., 2., 0.],
                [0., 0., 1., 2.],
                [0., 0., 0., 1.],
            ])
            .inv()
            .unwrap(),
            Matrix::<_, 4, 4>::from([
                [1., -2., 4., -8.],
                [0., 1., -2., 4.],
                [0., 0., 1., -2.],
                [0., 0., 0., 1.],
            ]),
        )
    }
}
