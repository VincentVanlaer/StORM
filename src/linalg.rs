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

const T18_COEFFICIENTS: [[f64; 5]; 5] = [
    [
        0.,
        -0.100_365_581_030_144_62,
        -0.008_029_246_482_411_57,
        -0.000_892_138_498_045_73,
        0.,
    ],
    [
        0.,
        0.397_849_749_499_645_1,
        1.367_837_784_604_117_2,
        0.498_289_622_525_382_67,
        -0.000_637_898_194_594_723_3,
    ],
    [
        -10.967_639_605_296_206,
        1.680_158_138_789_062,
        0.057_177_984_647_886_55,
        -0.006_982_101_224_880_520_6,
        0.00003349750170860705,
    ],
    [
        -0.090_431_683_239_081_06,
        -0.067_640_451_907_138_19,
        0.067_596_130_177_045_97,
        0.029_555_257_042_931_552,
        -0.00001391802575160607,
    ],
    [
        0.,
        0.,
        -0.092_336_461_936_711_86,
        -0.016_936_493_900_208_172,
        -0.00001400867981820361,
    ],
];

impl<const N: usize> Matrix<f64, N, N> {
    pub(crate) fn exp(&mut self, step: f64) {
        *self = *self * step;

        let norm = self
            .data
            .iter()
            .map(|&x| x.iter().map(|&x| x.abs()).sum())
            .reduce(f64::max)
            .expect("at least one item");

        let (mantissa, mut exponent, _) = norm.integer_decode();

        exponent += i16::try_from(mantissa.next_power_of_two().leading_zeros())
            .expect("max 64 leading zeroes");

        if exponent > 0 {
            *self = *self * (2.).powi((-exponent).into());
        }

        let eye = Self::eye();
        let a2 = self.matmul(*self);
        let a3 = a2.matmul(*self);
        let a6 = a3.matmul(a3);

        let b1 = eye * T18_COEFFICIENTS[0][0]
            + *self * T18_COEFFICIENTS[0][1]
            + a2 * T18_COEFFICIENTS[0][2]
            + a3 * T18_COEFFICIENTS[0][3]
            + a6 * T18_COEFFICIENTS[0][4];
        let b2 = eye * T18_COEFFICIENTS[1][0]
            + *self * T18_COEFFICIENTS[1][1]
            + a2 * T18_COEFFICIENTS[1][2]
            + a3 * T18_COEFFICIENTS[1][3]
            + a6 * T18_COEFFICIENTS[1][4];
        let b3 = eye * T18_COEFFICIENTS[2][0]
            + *self * T18_COEFFICIENTS[2][1]
            + a2 * T18_COEFFICIENTS[2][2]
            + a3 * T18_COEFFICIENTS[2][3]
            + a6 * T18_COEFFICIENTS[2][4];
        let b4 = eye * T18_COEFFICIENTS[3][0]
            + *self * T18_COEFFICIENTS[3][1]
            + a2 * T18_COEFFICIENTS[3][2]
            + a3 * T18_COEFFICIENTS[3][3]
            + a6 * T18_COEFFICIENTS[3][4];
        let b5 = eye * T18_COEFFICIENTS[4][0]
            + *self * T18_COEFFICIENTS[4][1]
            + a2 * T18_COEFFICIENTS[4][2]
            + a3 * T18_COEFFICIENTS[4][3]
            + a6 * T18_COEFFICIENTS[4][4];

        let a9 = b1.matmul(b5) + b4;

        *self = b2 + (b3 + a9).matmul(a9);

        if exponent > 0 {
            for _ in 0..exponent {
                *self = self.matmul(*self);
            }
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
