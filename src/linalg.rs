use blas::dgemm;
use lapack::{dgeev, dgetrf, dgetri};
use std::{
    mem::{self, transmute_copy},
    ops::{Add, AddAssign, Index, IndexMut, Mul, Sub},
    ptr::{slice_from_raw_parts, slice_from_raw_parts_mut},
};

use num::{complex::Complex64, One, Zero};

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

impl<T: Zero + Copy, const N: usize> Matmul<Matrix<T, N, N>> for Matrix<T, N, N>
where
    T: Mul<Output = T> + AddAssign,
{
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

const fn usize_to_i32<const N: usize>() -> i32 {
    match N {
        _ if N > (i32::MAX as usize) => {
            panic!("N is larger than i32::MAX")
        }
        _ => N as i32,
    }
}

impl<const N: usize> Matrix<f64, N, N> {
    const AS_I32_SIZE: i32 = usize_to_i32::<N>();
    const AS_I32_SIZE_WORK: i32 = usize_to_i32::<N>()
        .checked_mul(usize_to_i32::<N>())
        .unwrap();

    pub(crate) fn inv(mut self) -> Result<Matrix<f64, N, N>, i32> {
        let mut ipiv = [0i32; N];
        let mut work = Matrix {
            data: [[0.0; N]; N],
        };
        let mut info: i32 = 0;

        unsafe {
            dgetrf(
                Self::AS_I32_SIZE,
                Self::AS_I32_SIZE,
                self.as_slice_mut(),
                Self::AS_I32_SIZE,
                &mut ipiv,
                &mut info,
            )
        };

        match info {
            i if i < 0 => {
                panic!("Invalid argument {i} to dgetrf")
            }
            i if i > 0 => return Err(i),
            _ => {}
        };

        unsafe {
            dgetri(
                Self::AS_I32_SIZE,
                self.as_slice_mut(),
                Self::AS_I32_SIZE,
                &ipiv,
                work.as_slice_mut(),
                Self::AS_I32_SIZE_WORK,
                &mut info,
            )
        };

        match info {
            i if i < 0 => {
                panic!("Invalid argument {i} to dgetri")
            }
            i if i > 0 => return Err(i),
            _ => {}
        };

        Ok(self)
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

impl<T: Copy + Zero + Mul<C, Output = T>, C: Copy, const N: usize, const M: usize> Mul<C>
    for Matrix<T, N, M>
where
    T: Mul<T, Output = T>,
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
