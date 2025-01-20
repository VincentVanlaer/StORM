use std::{borrow::Borrow, marker::PhantomData, mem::MaybeUninit, ops::Add};

use nalgebra::{
    allocator::Allocator,
    base::Matrix,
    uninit::{InitStatus, Uninit},
    ComplexField, Const, DefaultAllocator, Dim, Dyn, Field, RealField, Scalar, Storage, StorageMut,
    ViewStorage, ViewStorageMut,
};
use num::Float;
use simba::scalar::SupersetOf;

pub(crate) struct MatrixArray<T, R, C, L, S> {
    data: S,
    _phantom: PhantomData<(T, R, C, L)>,
}

pub type OwnedArray<T, R, C, L> = <DefaultAllocator as ArrayAllocator<R, C, L>>::Buffer<T>;
pub type OMatrixArray<T, R, C, L> = MatrixArray<T, R, C, L, OwnedArray<T, R, C, L>>;

pub(crate) trait ArrayAllocator<R: Dim, C: Dim, L: Dim> {
    type Buffer<T: Scalar>: ArrayStorage<T, R, C, L>;
    type BufferUninit<T: Scalar>: ArrayStorage<MaybeUninit<T>, R, C, L>;

    fn allocate_uninit<T: Scalar>(nrows: R, ncols: C, length: L) -> Self::BufferUninit<T>;

    unsafe fn assume_init<T: Scalar>(uninit: Self::BufferUninit<T>) -> Self::Buffer<T>;
}

pub(crate) unsafe trait ArrayStorage<T, R, C, L> {
    fn ptr(&self) -> *const T;
    fn ptr_mut(&mut self) -> *mut T;
    fn shape(&self) -> (R, C, L);
    unsafe fn get_unchecked_mut(&mut self, r: usize, c: usize, i: usize) -> &mut T;
}

#[repr(transparent)]
pub(crate) struct SizedMatrixArray<T, const R: usize, const C: usize, const L: usize>(
    pub [[[T; R]; C]; L],
);

unsafe impl<T, const R: usize, const C: usize, const L: usize>
    ArrayStorage<T, Const<R>, Const<C>, Const<L>> for SizedMatrixArray<T, R, C, L>
{
    fn ptr(&self) -> *const T {
        self.0.as_ptr() as *const T
    }

    fn ptr_mut(&mut self) -> *mut T {
        self.0.as_ptr() as *mut T
    }

    fn shape(&self) -> (Const<R>, Const<C>, Const<L>) {
        (Const, Const, Const)
    }

    unsafe fn get_unchecked_mut(&mut self, r: usize, c: usize, i: usize) -> &mut T {
        self.0
            .get_unchecked_mut(i)
            .get_unchecked_mut(c)
            .get_unchecked_mut(r)
    }
}

impl<const R: usize, const C: usize, const L: usize> ArrayAllocator<Const<R>, Const<C>, Const<L>>
    for DefaultAllocator
{
    type Buffer<T: Scalar> = SizedMatrixArray<T, R, C, L>;
    type BufferUninit<T: Scalar> = SizedMatrixArray<MaybeUninit<T>, R, C, L>;

    fn allocate_uninit<T: Scalar>(_: Const<R>, _: Const<C>, _: Const<L>) -> Self::BufferUninit<T> {
        let array: [[[MaybeUninit<T>; R]; C]; L] = unsafe { MaybeUninit::uninit().assume_init() };
        SizedMatrixArray(array)
    }

    unsafe fn assume_init<T: Scalar>(uninit: Self::BufferUninit<T>) -> Self::Buffer<T> {
        SizedMatrixArray((&uninit as *const _ as *const [_; L]).read())
    }
}

pub(crate) struct UnsizedMatrixArray<T, R: Dim, C: Dim, L: Dim> {
    data: core::ptr::Unique<T>,
    rows: R,
    columns: C,
    length: L,
}

impl<T, R: Dim, C: Dim, L: Dim> UnsizedMatrixArray<T, R, C, L> {
    fn new_with(rows: R, columns: C, length: L, f: impl FnMut() -> T) -> Self {
        let mut data = Vec::new();
        let l = rows.value() * columns.value() * length.value();
        data.reserve_exact(l);
        data.resize_with(l, f);

        let data: &mut [T] = Box::<[T]>::leak(data.into_boxed_slice());

        assert_eq!(data.len(), rows.value() * columns.value() * length.value());

        UnsizedMatrixArray {
            data: core::ptr::Unique::new(data.as_mut_ptr()).unwrap(),
            rows,
            columns,
            length,
        }
    }
}

impl<T, R: Dim, C: Dim, L: Dim> Drop for UnsizedMatrixArray<T, R, C, L> {
    fn drop(&mut self) {
        unsafe {
            drop(Box::<[T]>::from_raw(core::slice::from_raw_parts_mut(
                self.data.as_ptr(),
                self.rows.value() * self.columns.value() * self.length.value(),
            )))
        };
    }
}

unsafe impl<T, R: Dim, C: Dim, L: Dim> ArrayStorage<T, R, C, L> for UnsizedMatrixArray<T, R, C, L> {
    fn ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    fn ptr_mut(&mut self) -> *mut T {
        self.data.as_ptr()
    }

    fn shape(&self) -> (R, C, L) {
        (self.rows, self.columns, self.length)
    }

    unsafe fn get_unchecked_mut(&mut self, r: usize, c: usize, i: usize) -> &mut T {
        &mut *self
            .data
            .as_ptr()
            .add(r + c * self.rows.value() + i * self.columns.value() * self.rows.value())
    }
}

macro_rules! impl_dyn {
    ($r: ty, $c: ty, $l: ty, $($($gen: ident) *: $bound: ident),*) => {
        impl<$($($gen) *: $bound),*> ArrayAllocator<$r, $c, $l> for DefaultAllocator {
            type Buffer<T: Scalar> = UnsizedMatrixArray<T, $r, $c, $l>;
            type BufferUninit<T: Scalar> = UnsizedMatrixArray<MaybeUninit<T>, $r, $c, $l>;

            fn allocate_uninit<T: Scalar>(
                nrows: $r,
                ncols: $c,
                length: $l,
            ) -> Self::BufferUninit<T> {
                UnsizedMatrixArray::new_with(nrows, ncols, length, MaybeUninit::uninit)
            }

            unsafe fn assume_init<T: Scalar>(uninit: Self::BufferUninit<T>) -> Self::Buffer<T> {
                let UnsizedMatrixArray {
                    data,
                    rows,
                    columns,
                    length,
                } = uninit;

                core::mem::forget(uninit);

                UnsizedMatrixArray {
                    data: data.cast(),
                    rows,
                    columns,
                    length,
                }
            }
        }
    };
}

impl_dyn!(Dyn, C, L, C: Dim, L: Dim);
impl_dyn!(Const<R>, Dyn, L, const R: usize, L: Dim);
impl_dyn!(Const<R>, Const<C>, Dyn, const R: usize, const C: usize);

impl<T: Scalar, R: Dim, C: Dim, L: Dim>
    MatrixArray<T, R, C, L, <DefaultAllocator as ArrayAllocator<R, C, L>>::Buffer<T>>
where
    DefaultAllocator: ArrayAllocator<R, C, L>,
{
    pub(crate) fn new_with(rows: R, columns: C, length: L, mut f: impl FnMut() -> T) -> Self {
        let mut data =
            <DefaultAllocator as ArrayAllocator<R, C, L>>::allocate_uninit(rows, columns, length);

        for i in 0..length.value() {
            for c in 0..columns.value() {
                for r in 0..rows.value() {
                    Uninit::init(unsafe { data.get_unchecked_mut(r, c, i) }, f())
                }
            }
        }

        MatrixArray {
            data: unsafe { <DefaultAllocator as ArrayAllocator<_, _, _>>::assume_init(data) },
            _phantom: PhantomData {},
        }
    }
}

impl<T, R, C, L, S: ArrayStorage<T, R, C, L>> MatrixArray<T, R, C, L, S> {
    pub(crate) fn shape(&self) -> (R, C, L) {
        self.data.shape()
    }
}

impl<T: Scalar, R: Dim, C: Dim, L: Dim, S: ArrayStorage<T, R, C, L>> MatrixArray<T, R, C, L, S> {
    pub(crate) unsafe fn get_unchecked(
        &self,
        index: usize,
    ) -> nalgebra::Matrix<T, R, C, ViewStorage<T, R, C, Const<1>, R>> {
        let (rows, columns, length) = self.data.shape();
        let matrix_size = columns.value() * rows.value();
        let data = unsafe {
            core::slice::from_raw_parts(
                self.data.ptr(),
                columns.value() * rows.value() * length.value(),
            )
        };

        nalgebra::Matrix::from_data(unsafe {
            ViewStorage::from_raw_parts(
                data.get_unchecked(index * matrix_size..(index + 1) * matrix_size)
                    .as_ptr(),
                (rows, columns),
                (Const::<1> {}, rows),
            )
        })
    }

    pub(crate) unsafe fn get_unchecked_mut(
        &mut self,
        index: usize,
    ) -> nalgebra::Matrix<T, R, C, ViewStorageMut<T, R, C, Const<1>, R>> {
        let (rows, columns, length) = self.data.shape();
        let matrix_size = columns.value() * rows.value();
        let data = unsafe {
            core::slice::from_raw_parts_mut(
                self.data.ptr_mut(),
                columns.value() * rows.value() * length.value(),
            )
        };

        nalgebra::Matrix::from_data(unsafe {
            ViewStorageMut::from_raw_parts(
                data.get_unchecked_mut(index * matrix_size..(index + 1) * matrix_size)
                    .as_mut_ptr(),
                (rows, columns),
                (Const::<1> {}, rows),
            )
        })
    }

    pub(crate) fn index(
        &self,
        index: usize,
    ) -> nalgebra::Matrix<T, R, C, ViewStorage<T, R, C, Const<1>, R>> {
        assert!(index < self.data.shape().2.value());

        unsafe { self.get_unchecked(index) }
    }

    pub(crate) fn index_mut(
        &mut self,
        index: usize,
    ) -> nalgebra::Matrix<T, R, C, ViewStorageMut<T, R, C, Const<1>, R>> {
        assert!(index < self.data.shape().2.value());

        unsafe { self.get_unchecked_mut(index) }
    }
}

impl<const L: usize, T: Scalar + Clone, N: Dim, M: Dim, S: Storage<T, N, M>>
    From<[Matrix<T, N, M, S>; L]> for OMatrixArray<T, N, M, Const<L>>
where
    DefaultAllocator: ArrayAllocator<N, M, Const<L>>,
{
    fn from(value: [Matrix<T, N, M, S>; L]) -> Self {
        let shape = if L == 0 {
            (N::from_usize(0), M::from_usize(0))
        } else {
            value[0].shape_generic()
        };

        let mut data = <DefaultAllocator as ArrayAllocator<_, _, _>>::allocate_uninit(
            shape.0, shape.1, Const::<L>,
        );

        for i in 0..L {
            for c in 0..shape.1.value() {
                for r in 0..shape.1.value() {
                    Uninit::init(
                        unsafe { data.get_unchecked_mut(r, c, i) },
                        value[i].index((r, c)).clone(),
                    )
                }
            }
        }

        MatrixArray {
            data: unsafe { <DefaultAllocator as ArrayAllocator<_, _, _>>::assume_init(data) },
            _phantom: PhantomData {},
        }
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

pub(crate) trait Exp {
    fn exp(&mut self);
}

pub(crate) fn assign_matrix<T: Scalar, N: Dim, M: Dim>(
    m1: &mut nalgebra::Matrix<T, N, M, impl StorageMut<T, N, M>>,
    m2: nalgebra::Matrix<T, N, M, impl Storage<T, N, M>>,
) {
    assert_eq!(m1.shape_generic(), m2.shape_generic());

    for (e1, e2) in m1.iter_mut().zip(m2.into_iter()) {
        *e1 = e2.clone()
    }
}

impl<T: ComplexField, N: Dim, S: StorageMut<T, N, N>> Exp for nalgebra::Matrix<T, N, N, S>
where
    DefaultAllocator: Allocator<N, N>,
{
    // Matrix exponential using the scaling and squaring algorithm, with an 18th order Taylor
    // expansion as approximation. The bounds on the scaling where chosen such that the expansion
    // is accurate to machine precision for a 64-bit floating point number (see Blanes, Kopylov,
    // and Seydao ̆glu, 'Efficient scaling and squaring method for the matrix exponential', 2024).
    // For the constants used in the approximation, see Eqn. (13) of Bader, Blanes and Casas
    // (2018), 'An improved algorithm to compute the exponential of a matrix'.
    //
    // Rational for choosing this algorithm:
    //
    // - Scaling and squaring is computationally simple, compared to e.g. the eigenvalue method in
    //   GYRE
    // - Numerical stability is also better compared to other iterative methods for small changes
    //   in the frequency
    // - The choice for the t18 approximation instead of the more commonly considered Padé
    //   approximations comes from the balance between the cost of a matrix inverse vs. a matrix
    //   multiplication. Blanes et al. (2024) assumes that a matrix inverse is ~1.33 times the
    //   cost of a matrix multiplication. For the small matrices we are dealing with, that is not
    //   the case. While the matrix multiplication is done branchless, this cannot be done for the
    //   matrix inverse. The number of branches is O(N^2), and hence will become less important
    //   for large matrices.
    //
    //   NOTE: nalgebra does non-branchy inverses for matrices up to size 4, and one inverse can be
    //   left to be dealt with by the determinant calculation, which makes Padé methods free
    //   compared to Taylor methods
    fn exp(&mut self) {
        assert_eq!(self.nrows(), self.ncols());

        let norm: f64 = self
            .column_iter()
            .map(|x| {
                x.iter().map(|x| x.clone().abs()).fold(
                    <T::RealField as SupersetOf<f64>>::from_subset(&0.),
                    <T::RealField as Add<T::RealField>>::add,
                )
            })
            .reduce(<T::RealField as RealField>::max)
            .expect("at least one item")
            .to_subset()
            .expect("should be real");

        let (mantissa, mut exponent, _) = norm.integer_decode();

        exponent += i16::try_from(mantissa.next_power_of_two().leading_zeros())
            .expect("max 64 leading zeroes");

        if exponent > 0 {
            *self *= T::from_subset(&f64::powi(2., (-exponent).into()));
        }

        let eye = &nalgebra::Matrix::<T, N, N, _>::identity_generic(
            self.shape_generic().0,
            self.shape_generic().1,
        );
        let a2 = &(&*self * &*self);
        let a3 = &(a2 * &*self);
        let a6 = &(a3 * a3);

        let b1 = eye * T::from_subset(&(T18_COEFFICIENTS[0][0]))
            + &*self * T::from_subset(&(T18_COEFFICIENTS[0][1]))
            + a2 * T::from_subset(&(T18_COEFFICIENTS[0][2]))
            + a3 * T::from_subset(&(T18_COEFFICIENTS[0][3]))
            + a6 * T::from_subset(&(T18_COEFFICIENTS[0][4]));
        let b2 = eye * T::from_subset(&(T18_COEFFICIENTS[1][0]))
            + &*self * T::from_subset(&(T18_COEFFICIENTS[1][1]))
            + a2 * T::from_subset(&(T18_COEFFICIENTS[1][2]))
            + a3 * T::from_subset(&(T18_COEFFICIENTS[1][3]))
            + a6 * T::from_subset(&(T18_COEFFICIENTS[1][4]));
        let b3 = eye * T::from_subset(&(T18_COEFFICIENTS[2][0]))
            + &*self * T::from_subset(&(T18_COEFFICIENTS[2][1]))
            + a2 * T::from_subset(&(T18_COEFFICIENTS[2][2]))
            + a3 * T::from_subset(&(T18_COEFFICIENTS[2][3]))
            + a6 * T::from_subset(&(T18_COEFFICIENTS[2][4]));
        let b4 = eye * T::from_subset(&(T18_COEFFICIENTS[3][0]))
            + &*self * T::from_subset(&(T18_COEFFICIENTS[3][1]))
            + a2 * T::from_subset(&(T18_COEFFICIENTS[3][2]))
            + a3 * T::from_subset(&(T18_COEFFICIENTS[3][3]))
            + a6 * T::from_subset(&(T18_COEFFICIENTS[3][4]));
        let b5 = eye * T::from_subset(&(T18_COEFFICIENTS[4][0]))
            + &*self * T::from_subset(&(T18_COEFFICIENTS[4][1]))
            + a2 * T::from_subset(&(T18_COEFFICIENTS[4][2]))
            + a3 * T::from_subset(&(T18_COEFFICIENTS[4][3]))
            + a6 * T::from_subset(&(T18_COEFFICIENTS[4][4]));

        let a9 = &(b1 * b5 + b4);

        assign_matrix(self, b2 + (b3 + a9) * a9);

        if exponent > 0 {
            for _ in 0..exponent {
                assign_matrix(self, &*self * &*self);
            }
        }
    }
}

pub(crate) fn commutator<T: Scalar + Field, N: Dim, S1: Storage<T, N, N>, S2: Storage<T, N, N>>(
    m1: impl Borrow<Matrix<T, N, N, S1>>,
    m2: impl Borrow<Matrix<T, N, N, S2>>,
) -> Matrix<T, N, N, <DefaultAllocator as Allocator<N, N>>::Buffer<T>>
where
    DefaultAllocator: nalgebra::allocator::Allocator<N, N>,
{
    let m1 = m1.borrow();
    let m2 = m2.borrow();
    m1 * m2 - m2 * m1
}

#[cfg(test)]
mod tests {
    use nalgebra::{Const, Dyn};

    use super::OMatrixArray;

    #[test]
    fn test_unsized_matrix_array() {
        let matrix_array =
            OMatrixArray::new_with(Const::<4> {}, Const::<4> {}, Dyn { 0: 600 }, || 1.);

        assert_eq!(matrix_array.index(10)[(2, 2)], 1.);
    }
}
