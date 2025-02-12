//! Provides post-processing routines to obtain displacement functions, mode-ids, ...

use std::f64::consts::PI;

use lapack::dggev3;
use nalgebra::{ComplexField, DMatrix, Matrix2, Vector2};
use num::{complex::Complex64, Zero};

use crate::model::{DimensionlessCoefficients, StellarModel};

/// Result from the post processing of a solution to the 1D oscillation equations
pub struct Rotating1DPostprocessing {
    /// The locations of the grid points, scaled by the radius of the star
    pub x: Box<[f64]>,
    /// y1 solution vector
    pub y1: Box<[f64]>,
    /// y2 solution vector
    pub y2: Box<[f64]>,
    /// y3 solution vector
    pub y3: Box<[f64]>,
    /// y4 solution vector
    pub y4: Box<[f64]>,
    /// Radial displacement vector
    pub xi_r: Box<[f64]>,
    /// Horizontal displacement vector
    pub xi_h: Box<[f64]>,
    /// Gravitational potential perturbation
    pub psi: Box<[f64]>,
    /// Derivative of the gravitional potential
    pub dpsi: Box<[f64]>,
    /// Density perturbation
    pub rho: Box<[f64]>,
    /// Pressure perturbation
    pub p: Box<[f64]>,
    /// ∇ξ
    pub chi: Box<[f64]>,
    /// Clockwise winding number
    pub clockwise_winding: f64,
    /// Counter-clockwise winding number
    pub counter_clockwise_winding: f64,
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let diff = a - b;

    if diff > PI {
        diff - 2. * PI
    } else if diff < -PI {
        diff + 2. * PI
    } else {
        diff
    }
}

impl Rotating1DPostprocessing {
    /// Post-process the results
    pub fn new(
        freq: f64,
        eigenvector: &[f64],
        ell: u64,
        m: i64,
        model: &StellarModel,
    ) -> Rotating1DPostprocessing {
        assert!(eigenvector.len().is_multiple_of(4));
        assert_eq!(model.r_coord[0], 0.);

        let mut y1 = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut y2 = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut y3 = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut y4 = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut xi_r = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut xi_h = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut p_prime = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut psi_prime = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut dpsi_prime = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut rho_prime = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut chi = vec![0.; model.r_coord.len()].into_boxed_slice();

        let time_scale = model.freq_scale();

        let lambda = (ell * (ell + 1)) as f64;
        let m = m as f64;
        let ell_i32: i32 = ell
            .try_into()
            .expect("ell is never going to be so big to cause problems here");

        let mut norm = 0.;

        let trapezoid = {
            let mut trapezoid = vec![0.; model.r_coord.len()];

            trapezoid[0] = 0.5 * (model.r_coord[1] - model.r_coord[0]) / model.radius;
            trapezoid[model.r_coord.len() - 1] = 0.5
                * (model.r_coord[model.r_coord.len() - 1] - model.r_coord[model.r_coord.len() - 2])
                / model.radius;

            for i in 1..(model.r_coord.len() - 1) {
                trapezoid[i] = 0.5 * (model.r_coord[i + 1] - model.r_coord[i - 1]) / model.radius;
            }

            trapezoid
        };

        for i in 1..y1.len() {
            y1[i] = eigenvector[(i - 1) * 4];
            y2[i] = eigenvector[(i - 1) * 4 + 1];
            y3[i] = eigenvector[(i - 1) * 4 + 2];
            y4[i] = eigenvector[(i - 1) * 4 + 3];

            let DimensionlessCoefficients {
                v_gamma,
                a_star,
                u: _,
                c1,
            } = model.dimensionless_coefficients(i);
            let dphi = model.grav * model.m_coord[i] / model.r_coord[i].powi(2);

            xi_r[i] = y1[i] * model.r_coord[i].powi(ell_i32 - 1) / model.radius.powi(ell_i32 - 2);
            p_prime[i] = y2[i] * model.rho[i] * dphi * model.r_coord[i].powi(ell_i32 - 1)
                / model.radius.powi(ell_i32 - 2);
            psi_prime[i] =
                y3[i] * dphi * model.r_coord[i].powi(ell_i32 - 1) / model.radius.powi(ell_i32 - 2);
            dpsi_prime[i] =
                y4[i] * dphi * model.r_coord[i].powi(ell_i32 - 2) / model.radius.powi(ell_i32 - 2);

            let rsigma = freq * time_scale - m * model.rot[i];
            let omega_rsq;
            let rel_rot;

            if ell != 0 {
                let rot = m * model.rot[i];
                omega_rsq = (lambda * (freq - rot) + 2. * rot) * (freq - rot);
                rel_rot = 2. * rot / (lambda * (freq - rot) + 2. * rot);

                let f = 2. * rot * time_scale / (ell * (ell + 1)) as f64;
                xi_h[i] = 1. / (rsigma * model.rho[i] * (rsigma + f))
                    * ((p_prime[i] + model.rho[i] * psi_prime[i]) / model.r_coord[i]
                        - f * rsigma * model.rho[i] * xi_r[i]);
            } else {
                omega_rsq = 1.;
                rel_rot = 0.;
                xi_h[i] = 0.;
            }

            let xdy1 = (v_gamma - 1. - ell as f64 - lambda * rel_rot) * y1[i]
                + (-v_gamma + lambda.powi(2) / (omega_rsq * c1)) * y2[i]
                + lambda.powi(2) / (omega_rsq * c1) * y3[i];
            chi[i] = (ell + 1) as f64 * model.r_coord[i].powi(ell_i32 - 2)
                / model.radius.powi(ell_i32 - 2)
                * (y1[i] + xdy1)
                - lambda / model.r_coord[i] * xi_h[i];
            rho_prime[i] = model.rho[i]
                * (p_prime[i] / (model.gamma1[i] * model.p[i])
                    + a_star * xi_r[i] / model.r_coord[i]);

            norm += model.rho[i]
                * model.r_coord[i].powi(2)
                * (xi_r[i] * xi_r[i] + lambda * xi_h[i] * xi_h[i])
                * trapezoid[i];
        }

        // Handle central point
        y1[0] = eigenvector[0];
        y2[0] = eigenvector[1];
        y3[0] = eigenvector[2];
        y4[0] = eigenvector[3];

        if ell != 1 {
            xi_r[0] = 0.;
            dpsi_prime[0] = 0.;
            xi_h[0] = 0.;
        } else {
            let ddphi0 = 4. / 3. * std::f64::consts::PI * model.rho[0] * model.grav;
            xi_r[0] = y1[0] * model.radius;
            dpsi_prime[0] = y4[0] * ddphi0 * model.radius;

            let rsigma = freq * time_scale - model.rot[0];
            let f = if ell != 0 {
                2. * m * model.rot[0] / (ell * (ell + 1)) as f64
            } else {
                0.
            };

            xi_h[0] = 1. / (rsigma * (rsigma + f))
                * ((y2[0] + y3[0]) * ddphi0 * model.radius - f * xi_r[0]);
        }

        if ell == 0 {
            p_prime[0] =
                y2[0] * model.rho[0].powi(2) * model.radius.powi(2) * model.grav * 4. / 3. * PI;
            psi_prime[0] = y3[0] * model.rho[0] * model.radius.powi(2) * model.grav * 4. / 3. * PI;
            rho_prime[0] = model.rho[0] * p_prime[0] / (model.gamma1[0] * model.p[0]);
            chi[0] = -rho_prime[0] / model.rho[0];
        } else {
            p_prime[0] = 0.;
            psi_prime[0] = 0.;
            rho_prime[0] = 0.;
            chi[0] = 0.;
        }

        let norm = 1. / norm.sqrt();

        for i in 0..y1.len() {
            y1[i] *= norm;
            y2[i] *= norm;
            y3[i] *= norm;
            y4[i] *= norm;
            xi_r[i] *= norm;
            xi_h[i] *= norm;
            psi_prime[i] *= norm;
            dpsi_prime[i] *= norm;
            rho_prime[i] *= norm;
            p_prime[i] *= norm;
            chi[i] *= norm;
        }

        let (cww, ccww) = xi_r
            .iter()
            .zip(xi_h.iter())
            .skip(1)
            .map(|(&xi_r, &xi_h)| f64::atan2(xi_r, xi_h))
            .filter(|arg0: &f64| f64::is_finite(*arg0))
            .map_windows(|&[alpha1, alpha2]| angle_diff(alpha2, alpha1))
            .fold((0., 0.), |(cww, ccww), val| {
                if ell == 0 {
                    (cww + val.abs(), ccww)
                } else if val >= 0. {
                    (cww, ccww + val)
                } else {
                    (cww - val, ccww)
                }
            });

        Rotating1DPostprocessing {
            x: model.r_coord.clone(),
            y1,
            y2,
            y3,
            y4,
            xi_r,
            xi_h,
            clockwise_winding: cww / PI,
            counter_clockwise_winding: ccww / PI,
            psi: psi_prime,
            dpsi: dpsi_prime,
            rho: rho_prime,
            p: p_prime,
            chi,
        }
    }
}

fn beta_k(k: u64, m: i64) -> f64 {
    let k = k as f64;
    let m = m as f64;

    ((k * k - m * m) / (4. * k * k - 1.)).sqrt()
}

fn q_kl2(k: u64, l: u64, m: i64) -> f64 {
    (3. / 2.)
        * if k == l {
            beta_k(k + 1, m).powi(2) + beta_k(k, m).powi(2) - 1. / 3.
        } else if k == l + 2 {
            beta_k(k, m) * beta_k(l + 1, m)
        } else if k + 2 == l {
            beta_k(k + 1, m) * beta_k(l, m)
        } else {
            0.
        }
}

fn q_kl2_h(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl2(k, l, m) * ((lambda_k + lambda_l) / 2. - 3.)
}

fn q_kl2_hd(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl2(k, l, m) * ((lambda_l - lambda_k) / 2. + 3.)
}

fn inner_prod_r(k: u64, l: u64) -> f64 {
    if k == l {
        1.
    } else {
        0.
    }
}

fn inner_prod_h(k: u64, l: u64) -> f64 {
    if k == l {
        (l * (l + 1)) as f64
    } else {
        0.
    }
}

/// Input for the deformation solver
pub struct ModeToPerturb<'a> {
    /// Spherical degree of the mode
    pub ell: u64,
    /// Frequency of the mode in units of the dynamical frequency of the model
    pub freq: f64,
    /// Result from the post processing of the mode
    pub post_processing: &'a Rotating1DPostprocessing,
}

/// Output from the deformation solver
pub struct ModeCoupling {
    /// Frequencies in units of the dynamical frequency of the model
    pub freqs: Box<[Complex64]>,
    /// Coupling between the modes. The eigenvectors are the columns of the matrix, with the
    /// ordering of the coefficients determined by the ordering of the modes in the input
    pub coupling: DMatrix<Complex64>,
    /// Mode norm coupling matrix
    pub d: DMatrix<f64>,
    /// Coriolis coupling matrix
    pub r: DMatrix<f64>,
    /// Structure coupling matrix and original solution
    pub l: DMatrix<f64>,
    /// Azimuthal order selected for this perturbation
    pub m: i64,
}

/// Using the structure deformation, perturb the frequencies and eigenfunctions of the modes.
///
/// The rotation used to compute the modes should match the rotation frequency in the perturbed
/// metric. Similarly, the azimuthal order should also match the way the modes are computed
pub fn perturb_deformed(
    model: &StellarModel,
    modes: &[ModeToPerturb],
    m: i64,
    PerturbedMetric {
        beta,
        dbeta,
        ddbeta,
        rot,
    }: &PerturbedMetric,
) -> ModeCoupling {
    let trapezoid = {
        let mut trapezoid = vec![0.; model.r_coord.len()];

        trapezoid[0] = 0.5 * (model.r_coord[1] - model.r_coord[0]) / model.radius;
        trapezoid[model.r_coord.len() - 1] = 0.5
            * (model.r_coord[model.r_coord.len() - 1] - model.r_coord[model.r_coord.len() - 2])
            / model.radius;

        for i in 1..(model.r_coord.len() - 1) {
            trapezoid[i] = 0.5 * (model.r_coord[i + 1] - model.r_coord[i - 1]) / model.radius;
        }

        trapezoid
    };

    let epsilon = beta.to_owned();
    let mut adepsilon = vec![0.; beta.len()];
    let mut threeepsilonadepsilon = vec![0.; beta.len()];
    let mut dthreeepsilonadepsilon = vec![0.; beta.len()];

    for i in 0..beta.len() {
        adepsilon[i] = model.r_coord[i] * dbeta[i];
        threeepsilonadepsilon[i] = 3. * epsilon[i] + adepsilon[i];
        dthreeepsilonadepsilon[i] = 4. * dbeta[i] + model.r_coord[i] * ddbeta[i];
    }

    let rot = m as f64 * rot;
    // rho * (omega + mOmega) * xi_l * xi_r
    let mut d_squared = DMatrix::from_element(modes.len(), modes.len(), 0.);
    let mut d_linear = DMatrix::from_element(modes.len(), modes.len(), 0.);
    let mut d_zero = DMatrix::from_element(modes.len(), modes.len(), 0.);

    for left_mode in 0..modes.len() {
        for right_mode in 0..modes.len() {
            let l = &modes[left_mode];
            let r = &modes[right_mode];

            for rc in 1..l.post_processing.x.len() {
                let val = trapezoid[rc]
                    * model.r_coord[rc].powi(2)
                    * model.rho[rc]
                    * ((inner_prod_r(l.ell, r.ell)
                        + 2. * q_kl2(l.ell, r.ell, m) * (epsilon[rc] + adepsilon[rc]))
                        * l.post_processing.xi_r[rc]
                        * r.post_processing.xi_r[rc]
                        + l.post_processing.xi_r[rc]
                            * r.post_processing.xi_h[rc]
                            * q_kl2_hd(l.ell, r.ell, m)
                            * epsilon[rc]
                        + l.post_processing.xi_h[rc]
                            * r.post_processing.xi_r[rc]
                            * q_kl2_hd(r.ell, l.ell, m)
                            * epsilon[rc]
                        + l.post_processing.xi_h[rc]
                            * r.post_processing.xi_h[rc]
                            * (inner_prod_h(r.ell, l.ell)
                                + 2. * q_kl2_h(r.ell, l.ell, m) * epsilon[rc]));
                assert!(
                    val.is_finite(),
                    "Not finite number at {}, {}",
                    left_mode,
                    right_mode
                );
                d_squared[(left_mode, right_mode)] += val;
                d_linear[(left_mode, right_mode)] += rot * val;
                d_zero[(left_mode, right_mode)] += rot * rot * val;
            }
        }
    }

    // 2i * rho * (omega + mOmega) * Omega * xi_l * ez x xi_r
    let mut r_linear = DMatrix::from_element(modes.len(), modes.len(), 0.);
    let mut r_zero = DMatrix::from_element(modes.len(), modes.len(), 0.);

    for left_mode in 0..modes.len() {
        for right_mode in 0..modes.len() {
            let l = &modes[left_mode];
            let r = &modes[right_mode];

            for rc in 1..l.post_processing.x.len() {
                let val = trapezoid[rc]
                    * 2.
                    * model.rho[rc]
                    * model.r_coord[rc].powi(2)
                    * (l.post_processing.xi_r[rc]
                        * l.post_processing.xi_h[rc]
                        * q_kl2(l.ell, r.ell, m)
                        * 2.
                        * (epsilon[rc] + adepsilon[rc])
                        + l.post_processing.xi_h[rc]
                            * r.post_processing.xi_r[rc]
                            * q_kl2(l.ell, r.ell, m)
                            * 2.
                            * (epsilon[rc] + adepsilon[rc])
                        + l.post_processing.xi_h[rc]
                            * r.post_processing.xi_h[rc]
                            * (if r.ell == l.ell { 1. } else { 0. } + 6. * q_kl2(l.ell, r.ell, m))
                            * 2.
                            * epsilon[rc]);
                assert!(
                    val.is_finite(),
                    "Not finite number at {}, {}",
                    left_mode,
                    right_mode
                );
                r_linear[(left_mode, right_mode)] += rot * val;
                r_zero[(left_mode, right_mode)] += rot * rot * val;
            }
        }
    }

    let mut l_zero = DMatrix::from_element(modes.len(), modes.len(), 0.);

    for left_mode in 0..modes.len() {
        for right_mode in 0..modes.len() {
            let l = &modes[left_mode];
            let r = &modes[right_mode];

            for rc in 1..l.post_processing.x.len() {
                l_zero[(left_mode, right_mode)] += trapezoid[rc]
                    * model.r_coord[rc].powi(2)
                    * (r.freq.powi(2)
                        * model.rho[rc]
                        * (inner_prod_r(r.ell, l.ell)
                            * l.post_processing.xi_r[rc]
                            * r.post_processing.xi_r[rc]
                            + inner_prod_h(r.ell, l.ell)
                                * l.post_processing.xi_h[rc]
                                * r.post_processing.xi_h[rc])
                        // gravity perturbation
                        + q_kl2(l.ell, r.ell, m)
                            * l.post_processing.psi[rc]
                            * r.post_processing.rho[rc]
                            * threeepsilonadepsilon[rc]
                        + q_kl2(l.ell, r.ell, m)
                            * model.rho[rc]
                            * l.post_processing.psi[rc]
                            * r.post_processing.xi_r[rc]
                            * dthreeepsilonadepsilon[rc]
                        + q_kl2_hd(l.ell, r.ell, m)
                            * model.rho[rc]
                            * l.post_processing.psi[rc]
                            * r.post_processing.xi_h[rc]
                            * threeepsilonadepsilon[rc] / model.r_coord[rc]
                        // density perturbation
                        - model.grav * model.m_coord[rc] / model.r_coord[rc].powi(2)
                            * l.post_processing.xi_r[rc]
                            * r.post_processing.xi_r[rc]
                            * model.rho[rc]
                            * q_kl2(l.ell, r.ell, m)
                            * dthreeepsilonadepsilon[rc]
                        - model.grav * model.m_coord[rc] / model.r_coord[rc].powi(2)
                            * l.post_processing.xi_r[rc]
                            * r.post_processing.xi_h[rc]
                            * model.rho[rc]
                            * q_kl2_hd(l.ell, r.ell, m)
                            * threeepsilonadepsilon[rc] / model.r_coord[rc]
                        // pressure perturbation
                        + model.gamma1[rc]
                            * model.p[rc]
                            * l.post_processing.chi[rc]
                            * (q_kl2(l.ell, r.ell, m)
                                * r.post_processing.xi_r[rc]
                                * dthreeepsilonadepsilon[rc]
                                + q_kl2_hd(l.ell, r.ell, m)
                                    * r.post_processing.xi_h[rc]
                                    * threeepsilonadepsilon[rc] / model.r_coord[rc]));
                assert!(
                    l_zero[(left_mode, right_mode)].is_finite(),
                    "Not finite number at {}, {}, {}: {}",
                    left_mode,
                    right_mode,
                    rc,
                    l_zero[(left_mode, right_mode)]
                );
            }
        }
    }

    // Av = omega Bv
    //
    //    0     1     | 1     0
    //  Zero  Linear  | 0  Squared

    let mut a = DMatrix::from_element(modes.len() * 2, modes.len() * 2, 0.);
    let mut b = DMatrix::from_element(modes.len() * 2, modes.len() * 2, 0.);

    a.view_range_mut(modes.len().., 0..modes.len())
        .copy_from(&(&d_zero - &r_zero + &l_zero));
    a.view_range_mut(0..modes.len(), modes.len()..)
        .fill_diagonal(1.);
    a.view_range_mut(modes.len().., modes.len()..)
        .copy_from(&(&d_linear - &r_linear));

    b.view_range_mut(0..modes.len(), 0..modes.len())
        .fill_diagonal(1.);
    b.view_range_mut(modes.len().., modes.len()..)
        .copy_from(&d_squared);

    let mut eigenval_real = vec![0.; modes.len() * 2];
    let mut eigenval_imag = vec![0.; modes.len() * 2];
    let mut eigenval_scale = vec![0.; modes.len() * 2];
    let mut eigenvectors = DMatrix::from_element(modes.len() * 2, modes.len() * 2, 0.);

    let n = (modes.len() * 2) as i32;
    let mut info = 0;
    let mut workspace = vec![0.; 1];

    unsafe {
        dggev3(
            b'N',
            b'V',
            n,
            a.as_mut_slice(),
            n,
            b.as_mut_slice(),
            n,
            &mut eigenval_real,
            &mut eigenval_imag,
            &mut eigenval_scale,
            [].as_mut_slice(),
            1,
            eigenvectors.as_mut_slice(),
            n,
            &mut workspace,
            -1,
            &mut info,
        )
    }

    assert_eq!(info, 0);

    let lwork = workspace[0] as i32;
    let mut workspace = vec![0.; lwork as usize];

    unsafe {
        dggev3(
            b'N',
            b'V',
            n,
            a.as_mut_slice(),
            n,
            b.as_mut_slice(),
            n,
            &mut eigenval_real,
            &mut eigenval_imag,
            &mut eigenval_scale,
            [].as_mut_slice(),
            1,
            eigenvectors.as_mut_slice(),
            n,
            &mut workspace,
            lwork,
            &mut info,
        )
    }

    assert_eq!(info, 0);

    let mut eigenvalues = vec![Complex64::zero(); eigenval_real.len()];

    for i in 0..eigenvalues.len() {
        eigenvalues[i] = Complex64::new(
            eigenval_real[i] / eigenval_scale[i],
            eigenval_imag[i] / eigenval_scale[i],
        );
    }

    let mut skip_next = false;

    let mut eigenvectors = eigenvectors.map(Complex64::from_real);

    for i in 0..eigenvalues.len() {
        if skip_next {
            skip_next = false;
            continue;
        }
        if eigenvalues[i].imaginary() != 0. {
            let real_part = &eigenvectors.column(i).clone_owned();
            let imag_part = &eigenvectors.column(i + 1).clone_owned();

            eigenvectors.set_column(i, &(real_part + imag_part * Complex64::i()));
            eigenvectors.set_column(i + 1, &(real_part - imag_part * Complex64::i()));
            skip_next = true;
        }
    }

    ModeCoupling {
        freqs: eigenvalues.into(),
        coupling: eigenvectors,
        d: d_squared,
        r: r_linear,
        l: l_zero,
        m,
    }
}

/// Contains the results of deforming the stellar structure with rotation
pub struct PerturbedMetric {
    /// P2 perturbation
    pub beta: Box<[f64]>,
    /// Derivative of beta
    pub dbeta: Box<[f64]>,
    /// Second derivative of beta
    pub ddbeta: Box<[f64]>,
    /// Rotation frequency
    pub rot: f64,
}

/// Deform the stellar structure of a model for a give rotation frequency
pub fn perturb_structure(model: &StellarModel, rot: f64) -> PerturbedMetric {
    let mut y = vec![Vector2::new(0., 0.); model.r_coord.len()];

    y[1] = Vector2::new(1., 2.);

    for i in 2..model.r_coord.len() {
        let delta = model.r_coord[i] - model.r_coord[i - 1];
        let x_12 = 0.5 * (model.r_coord[i] + model.r_coord[i - 1]);

        let DimensionlessCoefficients {
            v_gamma,
            a_star,
            u: _,
            c1: _,
        } = model.dimensionless_coefficients(i);

        let k = 4. * PI * model.radius.powi(2) / model.m_coord[i]
            * model.rho[i]
            * model.r_coord[i]
            * (-a_star + v_gamma);

        let DimensionlessCoefficients {
            v_gamma,
            a_star,
            u: _,
            c1: _,
        } = model.dimensionless_coefficients(i - 1);

        let k_prev = 4. * PI * model.radius.powi(2) / model.m_coord[i - 1]
            * model.rho[i - 1]
            * model.r_coord[i - 1]
            * (-a_star + v_gamma);

        let a = 0.5 * delta / x_12
            * Matrix2::new(
                -1.,
                1.,
                6. + 0.5 * (model.r_coord[i] / model.radius).powi(2) * (k + k_prev),
                -2.,
            );
        let diag = Matrix2::from_diagonal_element(1.);

        let step = nalgebra::Matrix2::try_inverse(diag - a).unwrap() * (diag + a);

        y[i] = step * y[i - 1];
    }

    let upper = y.last().unwrap();

    let a2 = -5. / 6. / (3. * upper.x + upper.y);

    let mut beta = vec![0.; model.r_coord.len()];
    let mut dbeta = vec![0.; model.r_coord.len()];
    let mut ddbeta = vec![0.; model.r_coord.len()];

    for i in 1..beta.len() {
        let dmda = 4. * PI * model.r_coord[i].powi(2) * model.rho[i] / model.m_coord[i];

        let DimensionlessCoefficients {
            v_gamma,
            a_star,
            u: _,
            c1: _,
        } = model.dimensionless_coefficients(i);

        let k = 4. * PI * model.radius.powi(2) / model.m_coord[i]
            * model.rho[i]
            * model.r_coord[i]
            * (-a_star + v_gamma);
        let ddpsi = ((6. + (model.r_coord[i] / model.radius).powi(2) * k) * y[i].x - 2. * y[i].y)
            / model.r_coord[i]
            / model.radius;

        beta[i] = 2. * model.r_coord[i] / model.radius * model.mass / model.m_coord[i]
            * a2
            * y[i].x
            * (model.r_coord[i] / model.radius)
            * rot.powi(2)
            / (model.grav * model.mass / model.radius.powi(3));
        dbeta[i] = beta[i] / model.r_coord[i] - beta[i] * dmda
            + beta[i] / (y[i].x * model.r_coord[i]) * y[i].y;
        ddbeta[i] = -2. * beta[i] / model.r_coord[i] * dmda
            + 2. * beta[i] / (y[i].x * model.r_coord[i].powi(2)) * y[i].y
            - 2. * beta[i] / (y[i].x * model.r_coord[i]) * dmda * y[i].y
            + 2. * beta[i] * dmda.powi(2)
            + beta[i] / (y[i].x * model.r_coord[i]) * ddpsi
            - beta[i] / model.m_coord[i] * 8. * PI * model.r_coord[i] * model.rho[i]
            - beta[i] * dmda / model.r_coord[i] * (-a_star + v_gamma);
    }

    PerturbedMetric {
        beta: beta.into_boxed_slice(),
        dbeta: dbeta.into_boxed_slice(),
        ddbeta: ddbeta.into_boxed_slice(),
        rot,
    }
}
