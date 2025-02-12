use nalgebra::{
    Complex, ComplexField, Const, DefaultAllocator, Dim, Dyn, Matrix, RealField, Scalar, Storage,
    StorageMut, Vector, Vector2, Vector3, allocator::Allocator,
};
use num_traits::One;
use std::fmt::{Debug, LowerExp};

/// Ax = cBx
pub(crate) fn qz<T: RealField + Copy + LowerExp, D: Dim>(
    a: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    b: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    eigenvalues: &mut Matrix<
        Complex<T>,
        D,
        Const<1>,
        impl StorageMut<Complex<T>, D, Const<1>> + Debug,
    >,
    eigenvalue_scale: &mut Matrix<T, D, Const<1>, impl StorageMut<T, D, Const<1>> + Debug>,
    eigenvectors: &mut Matrix<Complex<T>, D, D, impl StorageMut<Complex<T>, D, D> + Debug>,
) where
    DefaultAllocator: Allocator<D, D> + Allocator<D>,
{
    let mut z = Matrix::identity_generic(a.shape_generic().0, a.shape_generic().1);

    general_qr(a, b);
    general_hessenberg(a, b, &mut z);

    qz_iteration(a, b, &mut z);
    solve_real_schur(a, b, &mut z, eigenvalues, eigenvalue_scale, eigenvectors);
}

fn general_qr<T: RealField + Copy, D: Dim>(
    a: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    b: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
) {
    let s = a.shape().1;

    for c_idx in 0..(s - 1) {
        let reflector = LeftReflector::new(b, c_idx, c_idx, Dyn(s - c_idx));

        reflector.apply(a, c_idx, 0, Dyn(s));
        reflector.apply(b, c_idx, c_idx, Dyn(s - c_idx));
    }
}

fn general_hessenberg<T: RealField + Copy + LowerExp, D: Dim>(
    a: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    b: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    z: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
) {
    let s = a.shape().1;

    for c_idx in 0..(s - 2) {
        for r_idx in ((c_idx + 2)..s).rev() {
            // Householder transformation Q
            let reflector = LeftReflector::new(a, r_idx - 1, c_idx, Const::<2>);

            reflector.apply(a, r_idx - 1, c_idx, Dyn(s - c_idx));
            reflector.apply(b, r_idx - 1, c_idx, Dyn(s - c_idx));

            // Householder transformation Z
            let reflector = RightReflector::new(b, r_idx, r_idx - 1, Const::<2>);

            reflector.apply(a, 0, r_idx - 1, Dyn(s));
            reflector.apply(b, 0, r_idx - 1, Dyn(s));
            reflector.apply(z, 0, r_idx - 1, Dyn(s));
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct QZState {
    lower: usize,
    upper: usize,
}

fn deflate<T: RealField + Copy, D: Dim>(
    a: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    b: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    state: QZState,
) -> Option<QZState> {
    // Infinite eigenvalues
    let mut tol = T::zero(); // The matrix value above
    for idx in state.lower..(state.upper - 1) {
        if b[(idx, idx)].abs() < T::default_epsilon() * (tol + b[(idx, idx + 1)].abs()) {
            todo!("Infinite eigenvalue encountered")
        }

        tol = b[(idx, idx + 1)].abs();
    }

    // Actual deflation
    for idx in state.lower..(state.upper - 1) {
        let mut scale = a[(idx, idx)].abs() + a[(idx + 1, idx + 1)].abs();

        if scale.is_zero() {
            if state.upper > idx + 2 {
                scale += a[(idx + 2, idx + 1)].abs()
            }

            if state.lower < idx {
                scale += a[(idx, idx - 1)].abs()
            }
        }

        if a[(idx + 1, idx)].abs() < T::default_epsilon() * scale {
            let idx = idx + 1;
            a[(idx, idx - 1)] = T::zero();

            if idx + 2 >= state.upper {
                // We are deflating at most a 2x2 matrix, which is dealt with in post-processing by
                // the quadratic formula
                if idx <= 2 {
                    // and whatever remains is also at most a 2x2 matrix.
                    // This means we have achieved the Shur form
                    return None;
                } else {
                    // Everything above idx must be done, since state.upper starts at the size of
                    // the matrix. Hence, we can move down.
                    return Some(QZState {
                        lower: 0,
                        upper: idx,
                    });
                }
            } else {
                // Focus on the subset of the matrix that has decoupled (the lower right part)
                return Some(QZState {
                    lower: idx,
                    upper: state.upper,
                });
            }
        }
    }

    // No deflation, continue iterating
    Some(state)
}

#[derive(Debug)]
struct LeftReflector<T, R: Dim, S> {
    tau: T,
    v: Matrix<T, R, Const<1>, S>,
}

impl<T: Scalar + ComplexField + Copy, S: StorageMut<T, D, Const<1>>, D: Dim>
    LeftReflector<T, D, S>
{
    fn apply<R: Dim, C: Dim>(
        &self,
        mat: &mut Matrix<T, R, C, impl StorageMut<T, R, C>>,
        offset_r: usize,
        offset_c: usize,
        length: impl Dim,
    ) {
        let reflector_size = self.v.shape_generic().0;
        // eliminate bounds check in the loops
        let mut mat = mat.generic_view_mut((offset_r, offset_c), (reflector_size, length));

        for i in 0..length.value() {
            let mut s = T::zero();
            let row_view = mat.generic_view((0, i), (reflector_size, Const::<1>));
            for j in 0..reflector_size.value() {
                s += self.v[j].conjugate() * row_view[j];
            }
            let s = s * self.tau.conjugate();

            for j in 0..reflector_size.value() {
                mat[(j, i)] -= self.v[j] * s;
            }
        }
    }

    fn new_from_vec(x: &Matrix<T, D, Const<1>, impl Storage<T, D, Const<1>>>) -> Self
    where
        DefaultAllocator: Allocator<D, Const<1>, Buffer<T> = S>,
    {
        let alpha = x[0];
        let beta = ComplexField::from_real(-x.norm().copysign(alpha.real()));
        let mut value = x.clone_owned();

        if beta == T::zero() {
            Self {
                tau: T::zero(),
                v: value,
            }
        } else {
            value *= T::one() / (alpha - beta);
            value[0] = T::one();

            Self {
                tau: (beta - alpha) / beta,
                v: value,
            }
        }
    }

    fn new<R: Dim, C: Dim>(
        mat: &Matrix<T, R, C, impl Storage<T, R, C>>,
        offset_r: usize,
        offset_c: usize,
        length: D,
    ) -> Self
    where
        DefaultAllocator: Allocator<D, Const<1>, Buffer<T> = S>,
    {
        let x = mat.generic_view((offset_r, offset_c), (length, Const::<1>));

        Self::new_from_vec(&x)
    }
}

#[derive(Debug)]
struct RightReflector<T, R: Dim, S> {
    tau: T,
    v: Matrix<T, Const<1>, R, S>,
}

impl<T: Scalar + ComplexField + Copy, S: StorageMut<T, Const<1>, D>, D: Dim>
    RightReflector<T, D, S>
{
    fn apply<R: Dim, C: Dim>(
        &self,
        mat: &mut Matrix<T, R, C, impl StorageMut<T, R, C>>,
        offset_r: usize,
        offset_c: usize,
        length: impl Dim,
    ) {
        let reflector_size = self.v.shape_generic().1;
        // eliminate bounds check in the loops
        let mut mat = mat.generic_view_mut((offset_r, offset_c), (length, reflector_size));

        for i in 0..length.value() {
            let mut s = T::zero();
            let column_view = mat.generic_view((i, 0), (Const::<1>, reflector_size));
            for j in 0..reflector_size.value() {
                s += self.v[j].conjugate() * column_view[j];
            }
            let s = s * self.tau.conjugate();

            for j in 0..reflector_size.value() {
                mat[(i, j)] -= self.v[j] * s;
            }
        }
    }

    fn new<R: Dim, C: Dim>(
        mat: &Matrix<T, R, C, impl Storage<T, R, C>>,
        offset_r: usize,
        offset_c: usize,
        length: D,
    ) -> Self
    where
        DefaultAllocator: Allocator<Const<1>, D, Buffer<T> = S>,
    {
        let x = mat.generic_view((offset_r, offset_c), (Const::<1>, length));
        let n = x.len();
        let alpha = x[n - 1];
        let beta = ComplexField::from_real(-x.norm().copysign(alpha.real()));
        let mut value = x.clone_owned();

        if beta == T::zero() {
            Self {
                tau: T::zero(),
                v: value,
            }
        } else {
            value *= T::one() / (alpha - beta);
            value[n - 1] = T::one();

            Self {
                tau: (beta - alpha) / beta,
                v: value,
            }
        }
    }
}

fn implicit_qz_shift<T: RealField + Copy + LowerExp, D: Dim>(
    a: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    b: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    z: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    state: QZState,
) {
    let s = a.shape().1;
    // We should only end up here for blocks of at least 3x3
    assert!(state.lower + 2 < state.upper);

    // Construct the shift polynomial
    let i = state.upper - 2;

    let p_a = -b[(i, i)] * b[(i + 1, i + 1)];
    let p_b = a[(i + 1, i + 1)] * b[(i, i)] + b[(i + 1, i + 1)] * a[(i, i)]
        - b[(i, i + 1)] * a[(i + 1, i)];
    let p_c = a[(i, i + 1)] * a[(i + 1, i)] - a[(i, i)] * a[(i + 1, i + 1)];

    // Construct A/B e1 and (A/B)^2 e1
    let i = state.lower;

    let a_u = a.generic_view((i, i), (Const::<3>, Const::<2>));
    let mut b_u = b
        .generic_view((i, i), (Const::<2>, Const::<2>))
        .clone_owned();

    // Inversion of 2x2 upper triangular matrix
    b_u[(1, 0)] = T::zero(); // A hint towards the compiler
    b_u[(0, 0)] = T::one() / b_u[(0, 0)];
    b_u[(1, 1)] = T::one() / b_u[(1, 1)];
    b_u[(0, 1)] = -b_u[(0, 1)] * b_u[(0, 0)] * b_u[(1, 1)];

    let e1 = Vector2::new(T::one(), T::zero());

    let ab = a_u * (b_u * e1);
    let abab = a_u * (b_u * ab.generic_view((0, 0), (Const::<2>, Const::<1>)));

    // First column of the shift polynomial. As part of a QR decomposition of the shift
    // polynomial, this needs to be eliminated, e.g. with a Householder transformation
    let x = Vector3::new(T::one(), T::zero(), T::zero()) * p_c + ab * p_b + abab * p_a;
    let i = state.lower;
    // let x = Vector2::new(a[(i, i)] / b[(i, i)], a[(i + 1, i)] / b[(i, i)]);

    let x = LeftReflector::new_from_vec(&x);

    x.apply(a, i, i, Dyn(s - i));
    x.apply(b, i, i, Dyn(s - i));

    // Bulge chasing
    for i in state.lower..(state.upper - 3) {
        let reflector = RightReflector::new(b, i + 2, i, Const::<3>);
        reflector.apply(a, 0, i, Dyn(i + 4));
        reflector.apply(b, 0, i, Dyn(i + 3));
        reflector.apply(z, 0, i, Dyn(s));

        b[(i + 2, i)] = T::zero();
        b[(i + 2, i + 1)] = T::zero();

        let reflector = RightReflector::new(b, i + 1, i, Const::<2>);
        reflector.apply(a, 0, i, Dyn(i + 4));
        reflector.apply(b, 0, i, Dyn(i + 2));
        reflector.apply(z, 0, i, Dyn(s));

        b[(i + 1, i)] = T::zero();

        let reflector = LeftReflector::new(a, i + 1, i, Const::<3>);
        reflector.apply(a, i + 1, i, Dyn(s - i));
        reflector.apply(b, i + 1, i, Dyn(s - i));

        a[(i + 2, i)] = T::zero();
        a[(i + 3, i)] = T::zero();
    }

    let i = state.upper - 3;

    // The bulge is now in the lower right corner of the matrix
    let reflector = RightReflector::new(b, i + 2, i, Const::<3>);
    reflector.apply(a, 0, i, Dyn(i + 3));
    reflector.apply(b, 0, i, Dyn(i + 3));
    reflector.apply(z, 0, i, Dyn(s));

    b[(i + 2, i)] = T::zero();
    b[(i + 2, i + 1)] = T::zero();

    let reflector = RightReflector::new(b, i + 1, i, Const::<2>);
    reflector.apply(a, 0, i, Dyn(i + 3));
    reflector.apply(b, 0, i, Dyn(i + 2));
    reflector.apply(z, 0, i, Dyn(s));

    b[(i + 1, i)] = T::zero();

    let reflector = LeftReflector::new(a, i + 1, i, Const::<2>);
    reflector.apply(a, i + 1, i, Dyn(s - i));
    reflector.apply(b, i + 1, i, Dyn(s - i));

    a[(i + 2, i)] = T::zero();

    let reflector = RightReflector::new(b, i + 2, i + 1, Const::<2>);
    reflector.apply(a, 0, i + 1, Dyn(i + 3));
    reflector.apply(b, 0, i + 1, Dyn(i + 3));
    reflector.apply(z, 0, i + 1, Dyn(s));

    b[(i + 2, i + 1)] = T::zero();
}

fn qz_iteration<T: RealField + Copy + LowerExp, D: Dim>(
    a: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    b: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    z: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
) {
    let n = a.shape().0;
    let mut state = QZState { lower: 0, upper: n };

    #[cfg_attr(not(test), expect(unused_variables))]
    for i in 0..(30 * n) {
        loop {
            let Some(new_state) = deflate(a, b, state) else {
                return;
            };

            if state == new_state {
                break;
            }

            state = new_state;
        }

        #[cfg(test)]
        eprintln!("Iter {i}, {:?}", state);

        implicit_qz_shift(a, b, z, state);
        #[cfg(test)]
        eprint!("{:.8e}", MatrixFormatter { mat: a });
        #[cfg(test)]
        eprint!("{:.8e}", MatrixFormatter { mat: b });

        for col_idx in 0..a.shape().1 {
            for row_idx in (col_idx + 2)..a.shape().0 {
                a[(row_idx, col_idx)] = T::zero();
            }
        }

        for col_idx in 0..b.shape().1 {
            for row_idx in (col_idx + 1)..b.shape().0 {
                b[(row_idx, col_idx)] = T::zero();
            }
        }
    }

    todo!();
}

fn solve_real_schur<T: RealField + Copy + LowerExp, D: Dim>(
    a: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    b: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    z: &mut Matrix<T, D, D, impl StorageMut<T, D, D> + Debug>,
    eigenvalues: &mut Matrix<
        Complex<T>,
        D,
        Const<1>,
        impl StorageMut<Complex<T>, D, Const<1>> + Debug,
    >,
    eigenvalue_scale: &mut Matrix<T, D, Const<1>, impl StorageMut<T, D, Const<1>> + Debug>,
    eigenvectors: &mut Matrix<Complex<T>, D, D, impl StorageMut<Complex<T>, D, D> + Debug>,
) where
    DefaultAllocator: Allocator<D, Const<1>> + Allocator<Const<2>, Const<1>> + Allocator<D, D>,
{
    let n = a.shape_generic().0;
    // First, we need to make sure that 2x2 blocks are complex eigenvalue pairs, and decompose the
    // others

    let mut i = 0;

    while i < n.value() {
        if i + 1 == n.value() || a[(i + 1, i)].is_zero() {
            eigenvalues[i] = Complex::<T>::from_real(a[(i, i)]);
            eigenvalue_scale[i] = b[(i, i)];

            i += 1;
        } else if i + 2 == n.value() || a[(i + 2, i + 1)].is_zero() {
            // Setup quadratic equation (based on Moler & Stewart)
            let b_11 = T::one() / b[(i, i)];
            let b_22 = T::one() / b[(i + 1, i + 1)];

            let shift1 = a[(i, i)] * b_11;
            let shift2 = a[(i + 1, i + 1)] * b_22;

            let (p, q, shift);

            if shift1.abs() <= shift2.abs() {
                let shifted_a12 = a[(i, i + 1)] - shift1 * b[(i, i + 1)];
                let shifted_a22 = a[(i + 1, i + 1)] - shift1 * b[(i + 1, i + 1)];
                // a21 does not get shifted, as b21 is zero

                // usual quadratic formula, but with a11 = 0
                p = T::from_subset(&0.5)
                    * (shifted_a22 * b_22 - b[(i, i + 1)] * a[(i + 1, i)] * b_11 * b_22);
                q = shifted_a12 * a[(i + 1, i)] * b_11 * b_22;
                shift = shift1;
            } else {
                let shifted_a12 = a[(i, i + 1)] - shift2 * b[(i, i + 1)];
                let shifted_a11 = a[(i, i)] - shift2 * b[(i, i)];

                p = T::from_subset(&0.5)
                    * (shifted_a11 * b_11 - b[(i, i + 1)] * a[(i + 1, i)] * b_11 * b_22);
                q = shifted_a12 * a[(i + 1, i)] * b_11 * b_22;
                shift = shift2;
            }

            let r = p * p + q;

            if r.is_sign_negative() {
                // Complex pair
                let lambda = Complex {
                    re: shift + p,
                    im: r.abs().sqrt(),
                };

                // We don't want to modify the matrices, since otherwise they would become complex
                let mut a = a
                    .generic_view((i, i), (Const::<2>, Const::<2>))
                    .map(Complex::from_real);
                let mut b = b
                    .generic_view((i, i), (Const::<2>, Const::<2>))
                    .map(Complex::from_real);

                let mut e = a - b * lambda;

                let reflector = if e.row(0).norm() > e.row(1).norm() {
                    RightReflector::new(&e, 0, 0, Const::<2>)
                } else {
                    RightReflector::new(&e, 1, 0, Const::<2>)
                };

                reflector.apply(&mut a, 0, 0, Const::<2>);
                reflector.apply(&mut b, 0, 0, Const::<2>);
                reflector.apply(&mut e, 0, 0, Const::<2>);

                let reflector = if a.norm() > b.norm() * lambda.abs() {
                    LeftReflector::new(&a, 0, 0, Const::<2>)
                } else {
                    LeftReflector::new(&b, 0, 0, Const::<2>)
                };

                reflector.apply(&mut a, 0, 0, Const::<2>);
                reflector.apply(&mut b, 0, 0, Const::<2>);

                eigenvalues[i] = a[(0, 0)] / b[(0, 0)].signum();
                eigenvalue_scale[i] = b[(0, 0)].modulus();

                eigenvalues[i + 1] = a[(1, 1)] / b[(1, 1)].signum();
                eigenvalue_scale[i + 1] = b[(1, 1)].modulus();
            } else {
                // Real pair
                let lambda = shift + p + r.sqrt().copysign(p);
                let e = a.generic_view((i, i), (Const::<2>, Const::<2>))
                    - b.generic_view((i, i), (Const::<2>, Const::<2>)) * lambda;

                let reflector = if e.row(0).norm() > e.row(1).norm() {
                    RightReflector::new(&e, 0, 0, Const::<2>)
                } else {
                    RightReflector::new(&e, 1, 0, Const::<2>)
                };

                reflector.apply(a, 0, i, Dyn(i + 2));
                reflector.apply(b, 0, i, Dyn(i + 2));
                reflector.apply(z, 0, i, Dyn(n.value()));

                let reflector = if a.generic_view((i, i), (Const::<2>, Const::<2>)).norm()
                    > b.generic_view((i, i), (Const::<2>, Const::<2>)).norm() * lambda.abs()
                {
                    LeftReflector::new(a, i, i, Const::<2>)
                } else {
                    LeftReflector::new(b, i, i, Const::<2>)
                };

                reflector.apply(a, i, i, Dyn(n.value() - i));
                reflector.apply(b, i, i, Dyn(n.value() - i));

                a[(i + 1, i)] = T::zero();
                b[(i + 1, i)] = T::zero();

                eigenvalues[i] = Complex::from_real(a[(i, i)]);
                eigenvalue_scale[i] = b[(i, i)];

                eigenvalues[i + 1] = Complex::from_real(a[(i + 1, i + 1)]);
                eigenvalue_scale[i + 1] = b[(i + 1, i + 1)];
            }

            i += 2;
        } else {
            unreachable!("Non-Schur form")
        }
    }

    // Compute the eigenvectors

    let mut i_iter = (0..n.value()).rev();

    while let Some(i) = i_iter.next() {
        if eigenvalues[i].imaginary().is_zero() {
            let lambda = eigenvalues[i].real() / eigenvalue_scale[i];
            let mut current_eigenvector = Vector::<T, D, _>::zeros_generic(n, Const {});

            current_eigenvector[i] = T::one();

            let mut j_iter = (0..i).rev();

            while let Some(j) = j_iter.next() {
                let v = current_eigenvector.generic_view((j + 1, 0), (Dyn(i - j), Const::<1>));
                let idx_offset = (j, j + 1);
                let length = (Const::<1>, Dyn(i - j));

                if eigenvalues[j].imaginary().is_zero() {
                    let rhs = a.generic_view(idx_offset, length) * v
                        - b.generic_view(idx_offset, length) * v * lambda;
                    let scale = a[(j, j)] - lambda * b[(j, j)];

                    current_eigenvector[j] = *(-rhs / scale).as_scalar();
                } else {
                    let idx_offset2 = (j - 1, j + 1);

                    let rhs1 = *(a.generic_view(idx_offset, length) * v
                        - b.generic_view(idx_offset, length) * v * lambda)
                        .as_scalar();
                    let rhs2 = *(a.generic_view(idx_offset2, length) * v
                        - b.generic_view(idx_offset2, length) * v * lambda)
                        .as_scalar();

                    let a = a
                        .generic_view((j - 1, j - 1), (Const::<2>, Const::<2>))
                        .clone_owned();
                    let b = b
                        .generic_view((j - 1, j - 1), (Const::<2>, Const::<2>))
                        .clone_owned();

                    let e = a - b * lambda;

                    // Gaussian elimination
                    if e[(0, 0)].abs() > e[(1, 0)].abs() {
                        let s = e[(1, 0)] / e[(0, 0)];

                        current_eigenvector[j] = -(rhs2 - s * rhs1) / (e[(1, 1)] - s * e[(0, 1)]);
                        current_eigenvector[j - 1] =
                            -(rhs1 + e[(0, 1)] * current_eigenvector[j]) / e[(0, 0)];
                    } else {
                        let s = e[(0, 0)] / e[(1, 0)];

                        current_eigenvector[j] = -(rhs1 - s * rhs2) / (e[(0, 1)] - s * e[(1, 1)]);
                        current_eigenvector[j - 1] =
                            -(rhs2 + e[(1, 1)] * current_eigenvector[j]) / e[(1, 0)];
                    }

                    let _ = j_iter.next();
                }
            }

            eigenvectors
                .column_mut(i)
                .copy_from(&(&*z * current_eigenvector).map(Complex::from_real));
        } else {
            let lambda = eigenvalues[i] / eigenvalue_scale[i];
            let mut current_eigenvector = Vector::<Complex<T>, D, _>::zeros_generic(n, Const {});

            let e = {
                let a = a
                    .generic_view((i, i - 1), (Const::<1>, Const::<2>))
                    .map(Complex::from_real);
                let b = b
                    .generic_view((i, i - 1), (Const::<1>, Const::<2>))
                    .map(Complex::from_real);
                a - b * lambda
            };

            current_eigenvector[i] = Complex::<T>::one();
            current_eigenvector[i - 1] = -e[(0, 1)] / e[(0, 0)];

            let mut j_iter = (0..(i - 1)).rev();

            while let Some(j) = j_iter.next() {
                let v = current_eigenvector.generic_view((j + 1, 0), (Dyn(i - j), Const::<1>));
                let idx_offset = (j, j + 1);
                let length = (Const::<1>, Dyn(i - j));

                if eigenvalues[j].imaginary().is_zero() {
                    let rhs = a.generic_view(idx_offset, length).map(Complex::from_real) * v
                        - b.generic_view(idx_offset, length).map(Complex::from_real) * v * lambda;
                    let scale =
                        Complex::from_real(a[(j, j)]) - lambda * Complex::from_real(b[(j, j)]);

                    current_eigenvector[j] = *(-rhs / scale).as_scalar();
                } else {
                    let idx_offset2 = (j - 1, j + 1);

                    let rhs1 = *(a.generic_view(idx_offset, length).map(Complex::from_real) * v
                        - b.generic_view(idx_offset, length).map(Complex::from_real) * v * lambda)
                        .as_scalar();
                    let rhs2 = *(a.generic_view(idx_offset2, length).map(Complex::from_real) * v
                        - b.generic_view(idx_offset2, length).map(Complex::from_real) * v * lambda)
                        .as_scalar();

                    let a = a
                        .generic_view((j - 1, j - 1), (Const::<2>, Const::<2>))
                        .map(Complex::from_real);

                    let b = b
                        .generic_view((j - 1, j - 1), (Const::<2>, Const::<2>))
                        .map(Complex::from_real);

                    let e = a - b * lambda;

                    // Gaussian elimination
                    if e[(0, 0)].abs() > e[(1, 0)].abs() {
                        let s = e[(1, 0)] / e[(0, 0)];

                        current_eigenvector[j] = -(rhs2 - s * rhs1) / (e[(1, 1)] - s * e[(0, 1)]);
                        current_eigenvector[j - 1] =
                            -(rhs1 + e[(0, 1)] * current_eigenvector[j]) / e[(0, 0)];
                    } else {
                        let s = e[(0, 0)] / e[(1, 0)];

                        current_eigenvector[j] = -(rhs1 - s * rhs2) / (e[(0, 1)] - s * e[(1, 1)]);
                        current_eigenvector[j - 1] =
                            -(rhs2 + e[(1, 1)] * current_eigenvector[j]) / e[(1, 0)];
                    }

                    let _ = j_iter.next();
                }
            }

            let current_eigenvector = &(z.map(Complex::from_real) * current_eigenvector);

            eigenvectors.column_mut(i).copy_from(current_eigenvector);

            eigenvectors
                .column_mut(i - 1)
                .copy_from(&current_eigenvector.map(Complex::conjugate));

            let _ = i_iter.next();
        }
    }
}

#[cfg(test)]
struct MatrixFormatter<'a, T, R, C, S> {
    mat: &'a Matrix<T, R, C, S>,
}

#[cfg(test)]
impl<T, R: Dim, C: Dim, S> LowerExp for MatrixFormatter<'_, T, R, C, S>
where
    T: nalgebra::Scalar + LowerExp,
    S: nalgebra::RawStorage<T, R, C>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn val_width<T: nalgebra::Scalar + LowerExp>(
            val: &T,
            f: &mut std::fmt::Formatter<'_>,
        ) -> usize {
            match f.precision() {
                Some(precision) => format!("{:.1$e}", val, precision).chars().count(),
                None => format!("{:.e}", val).chars().count(),
            }
        }

        let (nrows, ncols) = self.mat.shape();

        if nrows == 0 || ncols == 0 {
            return write!(f, "[ ]");
        }

        let mut max_length = 0;

        for i in 0..nrows {
            for j in 0..ncols {
                max_length = usize::max(max_length, val_width(&self.mat[(i, j)], f));
            }
        }

        let max_length_with_space = max_length + 1;

        writeln!(
            f,
            "  ┌ {:>width$} ┐",
            "",
            width = max_length_with_space * ncols - 1
        )?;

        for i in 0..nrows {
            write!(f, "  │")?;
            for j in 0..ncols {
                let number_length = val_width(&self.mat[(i, j)], f) + 1;
                let pad = max_length_with_space - number_length;
                write!(f, " {:>thepad$}", "", thepad = pad)?;
                match f.precision() {
                    Some(precision) => write!(f, "{:.1$e}", (*self.mat)[(i, j)], precision)?,
                    None => write!(f, "{:.e}", (*self.mat)[(i, j)])?,
                }
            }
            writeln!(f, " │")?;
        }

        writeln!(
            f,
            "  └ {:>width$} ┘",
            "",
            width = max_length_with_space * ncols - 1
        )
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{ComplexField, Matrix3, Vector3};
    use num_complex::Complex64;

    use crate::linalg::qz::{MatrixFormatter, solve_real_schur};

    use super::{general_hessenberg, general_qr, qz_iteration};

    #[test]
    fn test_qz_steps() {
        let eigenmatrix = Matrix3::from_row_slice(&[5., 0., 0., 0., 0., -10., 0., 10., 0.]);
        let ortho = Matrix3::from_row_slice(&[
            -0.0075541465053381,
            -0.4275523991053181,
            0.9039590039873834,
            0.9987028830428752,
            -0.048745986966488,
            -0.0147098659583176,
            -0.0503536123076048,
            -0.902675342952243,
            -0.4273660479654716,
        ]);
        let a = ortho.transpose() * eigenmatrix * ortho;
        let invertible = Matrix3::from_row_slice(&[
            0.28184663, 0.30548729, 0.47989487, 0.44022205, 0.48263787, 0.39080369, 0.28497564,
            0.20632106, 0.38766485,
        ]);
        let mut b = invertible;

        let mut a: Matrix3<f64> = b * a;

        let mut z = Matrix3::identity();

        general_qr(&mut a, &mut b);

        for elem in a.iter() {
            assert!(elem.is_finite());
        }

        for elem in b.iter() {
            assert!(elem.is_finite());
        }

        for col_idx in 0..b.shape().1 {
            for row_idx in (col_idx + 1)..b.shape().0 {
                assert!(
                    f64::abs(b[(row_idx, col_idx)]) <= 20. * f64::EPSILON,
                    "{row_idx}, {col_idx} is not zero, but {:?}",
                    b[(row_idx, col_idx)]
                );

                b[(row_idx, col_idx)] = 0.;
            }
        }

        general_hessenberg(&mut a, &mut b, &mut z);

        for elem in a.iter() {
            assert!(elem.is_finite());
        }

        for elem in b.iter() {
            assert!(elem.is_finite());
        }

        for elem in z.iter() {
            assert!(elem.is_finite());
        }

        for col_idx in 0..a.shape().1 {
            for row_idx in (col_idx + 2)..a.shape().0 {
                assert!(
                    f64::abs(a[(row_idx, col_idx)]) <= 20. * f64::EPSILON,
                    "{row_idx}, {col_idx} is not zero, but {:?}",
                    a[(row_idx, col_idx)]
                );
                a[(row_idx, col_idx)] = 0.;
            }
        }

        for col_idx in 0..b.shape().1 {
            for row_idx in (col_idx + 1)..b.shape().0 {
                assert!(
                    f64::abs(b[(row_idx, col_idx)]) <= 20. * f64::EPSILON,
                    "{row_idx}, {col_idx} is not zero, but {:?}",
                    b[(row_idx, col_idx)]
                );
                b[(row_idx, col_idx)] = 0.;
            }
        }

        qz_iteration(&mut a, &mut b, &mut z);

        let mut eigenvalues = Vector3::<Complex64>::default();
        let mut eigenvalue_scale = Vector3::<f64>::default();
        let mut eigenvectors = Matrix3::<Complex64>::default();

        solve_real_schur(
            &mut a,
            &mut b,
            &mut z,
            &mut eigenvalues,
            &mut eigenvalue_scale,
            &mut eigenvectors,
        );

        let eigenvalues = eigenvalues.component_div(&eigenvalue_scale.map(Complex64::from_real));

        eprintln!("{:.10e}", MatrixFormatter { mat: &eigenvalues });

        eprint!(
            "{:.3e}",
            MatrixFormatter {
                mat: &(ortho.map(Complex64::from_real) * eigenvectors)
            }
        );

        assert_eq!(Complex64 { im: 10.000000000000009, re: 3.932342611736938e-15 }, eigenvalues[0]);
        assert_eq!(Complex64 { im: -10.000000000000007, re: 4.570981397001066e-15 }, eigenvalues[1]);
        assert_eq!(Complex64::from_real(4.999999999999996), eigenvalues[2]);
    }
}
