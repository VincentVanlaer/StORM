use std::{marker::PhantomData, mem::MaybeUninit, ptr::NonNull};

use nalgebra::{
    Const, DefaultAllocator, Dim, Dyn, Scalar, Storage, StorageMut, ViewStorage, ViewStorageMut,
    base::Matrix,
    uninit::{InitStatus, Uninit},
};

pub(crate) struct MatrixArray<T, R, C, L, S> {
    data: S,
    _phantom: PhantomData<(T, R, C, L)>,
}

pub(crate) type OwnedArray<T, R, C, L> = <DefaultAllocator as ArrayAllocator<R, C, L>>::Buffer<T>;
pub(crate) type OMatrixArray<T, R, C, L> = MatrixArray<T, R, C, L, OwnedArray<T, R, C, L>>;

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
        unsafe {
            self.0
                .get_unchecked_mut(i)
                .get_unchecked_mut(c)
                .get_unchecked_mut(r)
        }
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
        SizedMatrixArray(unsafe { (&uninit as *const _ as *const [_; L]).read() })
    }
}

pub(crate) struct UnsizedMatrixArray<T, R: Dim, C: Dim, L: Dim> {
    // If at some point this needs to support Send & Sync, then this needs to be upgraded into
    // something equivalent to core::ptr::Unique
    //
    // See also https://users.rust-lang.org/t/where-did-unique-t-go/68807 for more information
    // about concerns regarding dropchk
    data: NonNull<T>,
    rows: R,
    columns: C,
    length: L,
}

unsafe impl<T: Send, R: Dim, C: Dim, L: Dim> Send for UnsizedMatrixArray<T, R, C, L> {}
unsafe impl<T: Send, R: Dim, C: Dim, L: Dim> Sync for UnsizedMatrixArray<T, R, C, L> {}

impl<T, R: Dim, C: Dim, L: Dim> UnsizedMatrixArray<T, R, C, L> {
    fn new_with(rows: R, columns: C, length: L, f: impl FnMut() -> T) -> Self {
        let mut data = Vec::new();
        let l = rows.value() * columns.value() * length.value();
        data.reserve_exact(l);
        data.resize_with(l, f);

        let data: &mut [T] = Box::<[T]>::leak(data.into_boxed_slice());

        assert_eq!(data.len(), rows.value() * columns.value() * length.value());

        UnsizedMatrixArray {
            data: NonNull::new(data.as_mut_ptr()).unwrap(),
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
        unsafe {
            &mut *self
                .data
                .as_ptr()
                .add(r + c * self.rows.value() + i * self.columns.value() * self.rows.value())
        }
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

pub(crate) fn assign_matrix<T: Scalar, N: Dim, M: Dim>(
    m1: &mut nalgebra::Matrix<T, N, M, impl StorageMut<T, N, M>>,
    m2: nalgebra::Matrix<T, N, M, impl Storage<T, N, M>>,
) {
    assert_eq!(m1.shape_generic(), m2.shape_generic());

    for (e1, e2) in m1.iter_mut().zip(m2.into_iter()) {
        *e1 = e2.clone()
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{Const, Dyn};

    use super::OMatrixArray;

    #[test]
    fn test_unsized_matrix_array() {
        let matrix_array = OMatrixArray::new_with(Const::<4> {}, Const::<4> {}, Dyn(600), || 1.);

        assert_eq!(matrix_array.index(10)[(2, 2)], 1.);
    }
}
