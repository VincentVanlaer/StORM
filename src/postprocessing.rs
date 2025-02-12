//! Provides post-processing routines to obtain displacement functions, mode-ids, ...

use std::f64::consts::PI;

use itertools::Itertools;
use nalgebra::{ComplexField, Const, DMatrix, DVector, Dyn, Matrix2, Vector2};
use num_complex::Complex64;
use num_traits::Zero;

use crate::{
    linalg::qz,
    model::{DimensionlessCoefficients, StellarModel},
};

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
    /// Radial displacement component
    pub xi_r: Box<[f64]>,
    /// Horizontal displacement component
    pub xi_h: Box<[f64]>,
    /// Toroidal displacement component for l - 1, phase shifted by (-i)
    pub xi_tn: Box<[f64]>,
    /// Toroidal displacement component for l + 1, phase shifted by (-i)
    pub xi_tp: Box<[f64]>,
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
    /// Clockwise crossing number (g-mode crossings)
    pub cross_clockwise: u64,
    /// Counter-clockwise crossing number (p-mode crossings)
    pub cross_counter_clockwise: u64,
    /// Radial order
    ///
    /// This is computed from the clockwise and counter-clockwise crossing number. The exact
    /// formula depends on the degree of the mode
    pub radial_order: i64,
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
        let mut xi_tn = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut xi_tp = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut p_prime = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut psi_prime = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut dpsi_prime = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut rho_prime = vec![0.; model.r_coord.len()].into_boxed_slice();
        let mut chi = vec![0.; model.r_coord.len()].into_boxed_slice();

        let freq_scale = model.freq_scale();

        let lambda = (ell * (ell + 1)) as f64;
        let lambda_n1 = (ell * (ell.saturating_sub(1))) as f64;
        let lambda_p1 = ((ell + 2) * (ell + 1)) as f64;
        let q_hd_n = q_kl1_hd(ell, ell.saturating_sub(1), m);
        let q_hd_p = q_kl1_hd(ell, ell + 1, m);
        let q_h_n = q_kl1_h(ell, ell.saturating_sub(1), m);
        let q_h_p = q_kl1_h(ell, ell + 1, m);
        let mf = m as f64;
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

            let rsigma = (freq - mf * model.rot[i]) * freq_scale;
            let omega_rsq;
            let rel_rot;

            if ell != 0 {
                let rot = mf * model.rot[i];
                omega_rsq = (lambda * (freq - rot) + 2. * rot) * (freq - rot);
                rel_rot = 2. * rot / (lambda * (freq - rot) + 2. * rot);

                let f = 2. * rot * freq_scale / (ell * (ell + 1)) as f64;
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

            if m.unsigned_abs() == ell || ell == 1 {
                xi_tn[i] = 0.;
            } else {
                xi_tn[i] = 2. * model.rot[i]
                    / (lambda_n1 * (freq - mf * model.rot[i]) + 2. * mf * model.rot[i])
                    * (-q_hd_n * xi_r[i] + q_h_n * xi_h[i]);
            }

            if ell == 0 {
                xi_tp[i] = 0.;
            } else {
                xi_tp[i] = 2. * model.rot[i]
                    / (lambda_p1 * (freq - mf * model.rot[i]) + 2. * mf * model.rot[i])
                    * (-q_hd_p * xi_r[i] + q_h_p * xi_h[i]);
            }

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

            let rsigma = (freq - mf * model.rot[0]) * freq_scale;
            let f = if ell != 0 {
                2. * mf * model.rot[0] * freq_scale / (ell * (ell + 1)) as f64
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

        if m.unsigned_abs() == ell || ell == 1 {
            xi_tn[0] = 0.;
        } else {
            xi_tn[0] = 2. * model.rot[0]
                / (lambda_n1 * (freq - mf * model.rot[0]) + 2. * mf * model.rot[0])
                * (-q_hd_n * xi_r[0] + q_h_n * xi_h[0]);
        }

        if ell == 0 {
            xi_tp[0] = 0.;
        } else {
            xi_tp[0] = 2. * model.rot[0]
                / (lambda_p1 * (freq - mf * model.rot[0]) + 2. * mf * model.rot[0])
                * (-q_hd_p * xi_r[0] + q_h_p * xi_h[0]);
        }

        let norm = 1. / norm.sqrt();

        for i in 0..y1.len() {
            y1[i] *= norm;
            y2[i] *= norm;
            y3[i] *= norm;
            y4[i] *= norm;
            xi_r[i] *= norm;
            xi_h[i] *= norm;
            xi_tn[i] *= norm;
            xi_tp[i] *= norm;
            psi_prime[i] *= norm;
            dpsi_prime[i] *= norm;
            rho_prime[i] *= norm;
            p_prime[i] *= norm;
            chi[i] *= norm;
        }

        let (cross_clockwise, cross_counter_clockwise, radial_order) = match ell {
            0 => {
                let (cw, ccw) = count_windings(&y1, &y2);

                (cw, ccw, ccw as i64)
            }
            1 => {
                let mut y1_alt = vec![0.; model.r_coord.len()].into_boxed_slice();
                let mut y2_alt = vec![0.; model.r_coord.len()].into_boxed_slice();

                for i in 0..y2_alt.len() {
                    y1_alt[i] = (1. - model.dimensionless_coefficients(i).u / 3.) * y1[i]
                        + (y3[i] - y4[i]) / 3.;
                    y2_alt[i] = y2[i] - y1[i];
                }

                let (cw, ccw) = count_windings(&y1_alt[2..], &y2_alt[2..]);

                if cw > ccw {
                    (cw, ccw, ccw as i64 - cw as i64)
                } else {
                    (cw, ccw, ccw as i64 - cw as i64 + 1)
                }
            }
            _ => {
                let mut y2_alt = vec![0.; model.r_coord.len()].into_boxed_slice();

                for i in 0..y2_alt.len() {
                    y2_alt[i] = y2[i] + y3[i];
                }

                let (cw, ccw) = count_windings(&y1, &y2_alt);

                (cw, ccw, ccw as i64 - cw as i64)
            }
        };

        Rotating1DPostprocessing {
            x: model.r_coord.clone(),
            y1,
            y2,
            y3,
            y4,
            xi_r,
            xi_h,
            psi: psi_prime,
            dpsi: dpsi_prime,
            rho: rho_prime,
            p: p_prime,
            chi,
            xi_tn,
            xi_tp,
            cross_clockwise,
            cross_counter_clockwise,
            radial_order,
        }
    }
}

fn count_windings(y1: &[f64], y2: &[f64]) -> (u64, u64) {
    let mut clockwise = 0;
    let mut counter_clockwise = 0;

    #[cfg(test)]
    eprintln!("---");
    y1.iter()
        .zip(y2.iter())
        .tuple_windows()
        .enumerate()
        .for_each(|(_i, ((&y1_1, &y2_1), (&y1_2, &y2_2)))| {
            if y1_1 <= 0. && y1_2 > 0. {
                // left to right
                let yt = y2_1 - y1_1 * (y2_2 - y2_1) / (y1_2 - y1_1);
                if yt > 0. {
                    #[cfg(test)]
                    eprintln!("↷ {_i}, {y1_1:.5e}, {y1_2:.5e}, {yt:.5e}");
                    // Above
                    clockwise += 1
                } else {
                    #[cfg(test)]
                    eprintln!("↺ {_i}, {y1_1:.5e}, {y1_2:.5e}, {yt:.5e}");
                    // Below (or exact zero, this is ignored)
                    counter_clockwise += 1;
                }
            } else if y1_1 >= 0. && y1_2 < 0. {
                // right to left
                let yt = y2_1 - y1_1 * (y2_2 - y2_1) / (y1_2 - y1_1);
                if yt > 0. {
                    // Above
                    #[cfg(test)]
                    eprintln!("↶ {_i}, {y1_1:.5e}, {y1_2:.5e}, {yt:.5e}");
                    counter_clockwise += 1
                } else {
                    #[cfg(test)]
                    eprintln!("↻ {_i}, {y1_1:.5e}, {y1_2:.5e}, {yt:.5e}");
                    // Below  (or exact zero, this is ignored)
                    clockwise += 1
                }
            }
        });

    #[cfg(test)]
    eprintln!("--- cw: {clockwise} ccw: {counter_clockwise} ---");
    (clockwise, counter_clockwise)
}

fn beta_k(k: u64, m: i64) -> f64 {
    let k = k as f64;
    let m = m as f64;

    ((k * k - m * m) / (4. * k * k - 1.)).sqrt()
}

fn q_kl1(k: u64, l: u64, m: i64) -> f64 {
    (-1.).powi((m % 2) as i32)
        * if k == l + 1 {
            beta_k(k, m)
        } else if l == k + 1 {
            beta_k(l, m)
        } else {
            0.
        }
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

fn q_kl1_h(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl1(k, l, m) * ((lambda_k + lambda_l) / 2. - 1.)
}

fn q_kl2_h(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl2(k, l, m) * ((lambda_k + lambda_l) / 2. - 3.)
}

fn q_kl1_hd(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl1(k, l, m) * ((lambda_l - lambda_k) / 2. + 1.)
}

fn q_kl2_hd(k: u64, l: u64, m: i64) -> f64 {
    let lambda_k = (k * (k + 1)) as f64;
    let lambda_l = (l * (l + 1)) as f64;

    q_kl2(k, l, m) * ((lambda_l - lambda_k) / 2. + 3.)
}

fn inner_prod_r(k: u64, l: u64) -> f64 {
    if k == l { 1. } else { 0. }
}

fn inner_prod_h(k: u64, l: u64) -> f64 {
    if k == l { (l * (l + 1)) as f64 } else { 0. }
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
        rot: _,
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

    let mf = m as f64;
    // rho * (omega - mOmega) * xi_l * xi_r
    let mut d_squared = DMatrix::from_element(modes.len(), modes.len(), 0.);
    let mut d_linear = DMatrix::from_element(modes.len(), modes.len(), 0.);
    let mut d_zero = DMatrix::from_element(modes.len(), modes.len(), 0.);

    // 2i * rho * (omega - mOmega) * Omega * xi_l * ez x xi_r
    let mut r_linear = DMatrix::from_element(modes.len(), modes.len(), 0.);
    let mut r_zero = DMatrix::from_element(modes.len(), modes.len(), 0.);

    let mut l_zero = DMatrix::from_element(modes.len(), modes.len(), 0.);

    for left_mode in 0..modes.len() {
        for right_mode in 0..modes.len() {
            let l = &modes[left_mode];
            let r = &modes[right_mode];
            let q_kl2 = q_kl2(l.ell, r.ell, m);
            let q_kl2_hrd = q_kl2_hd(l.ell, r.ell, m);
            let q_kl2_hld = q_kl2_hd(r.ell, l.ell, m);
            let q_kl2_h = q_kl2_h(l.ell, r.ell, m);
            let q_rt_p;
            let q_rt_n;
            let q_ht_p;
            let q_ht_n;
            if l.ell == r.ell {
                q_rt_p = q_kl1_hd(l.ell, r.ell + 1, m);
                q_rt_n = q_kl1_hd(l.ell, r.ell - 1, m);
                q_ht_p = -q_kl1_h(l.ell, r.ell + 1, m);
                q_ht_n = -q_kl1_h(l.ell, r.ell - 1, m);
            } else {
                q_rt_p = 0.;
                q_rt_n = 0.;
                q_ht_p = 0.;
                q_ht_n = 0.;
            }
            let inner_prod_r = inner_prod_r(l.ell, r.ell);
            let inner_prod_h = inner_prod_h(l.ell, r.ell);

            for rc in 1..l.post_processing.x.len() {
                let rr = l.post_processing.xi_r[rc] * r.post_processing.xi_r[rc];
                let rh = l.post_processing.xi_r[rc] * r.post_processing.xi_h[rc];
                let hr = l.post_processing.xi_h[rc] * r.post_processing.xi_r[rc];
                let hh = l.post_processing.xi_h[rc] * r.post_processing.xi_h[rc];
                let rtp = l.post_processing.xi_r[rc] * r.post_processing.xi_tp[rc];
                let rtn = l.post_processing.xi_r[rc] * r.post_processing.xi_tn[rc];
                let htp = l.post_processing.xi_h[rc] * r.post_processing.xi_tp[rc];
                let htn = l.post_processing.xi_h[rc] * r.post_processing.xi_tn[rc];
                let vol = trapezoid[rc] * model.r_coord[rc].powi(2) * model.rho[rc];

                let val = vol
                    * (rr * inner_prod_r
                        + hh * inner_prod_h
                        + rr * 2. * q_kl2 * (epsilon[rc] + adepsilon[rc])
                        + (rh * q_kl2_hrd + hr * q_kl2_hld + hh * 2. * q_kl2_h) * epsilon[rc]);

                assert!(
                    val.is_finite(),
                    "Not finite number at {}, {}",
                    left_mode,
                    right_mode
                );
                d_squared[(left_mode, right_mode)] += val;
                d_linear[(left_mode, right_mode)] += 2. * mf * model.rot[rc] * val;
                d_zero[(left_mode, right_mode)] += mf * mf * model.rot[rc] * model.rot[rc] * val;

                let val = vol
                    * 2.
                    * (inner_prod_r * (hr + rh + hh)
                        + (rh + hr) * q_kl2 * (2. * epsilon[rc] + adepsilon[rc])
                        + hh * (inner_prod_r + 6. * q_kl2) * 2. * epsilon[rc]);

                let val_t = vol
                    * 2.
                    * model.rot[rc]
                    * (q_rt_p * rtp + q_rt_n * rtn + q_ht_n * htn + q_ht_p * htp);

                assert!(
                    val.is_finite(),
                    "Not finite number at {}, {}",
                    left_mode,
                    right_mode
                );
                r_linear[(left_mode, right_mode)] += mf * model.rot[rc] * val + val_t;
                r_zero[(left_mode, right_mode)] +=
                    mf * model.rot[rc] * (mf * model.rot[rc] * val + val_t);

                l_zero[(left_mode, right_mode)] += trapezoid[rc]
                    * model.r_coord[rc].powi(2)
                    * ((r.freq - mf * model.rot[rc]).powi(2)
                        * model.rho[rc]
                        * (inner_prod_r * rr + inner_prod_h * hh)
                        + 2. * mf
                            * model.rot[rc]
                            * (r.freq - mf * model.rot[rc])
                            * model.rho[rc]
                            * inner_prod_r
                            * (rh + hr + hh)
                        + q_kl2
                            * l.post_processing.psi[rc]
                            * r.post_processing.rho[rc]
                            * threeepsilonadepsilon[rc]
                        + q_kl2
                            * model.rho[rc]
                            * l.post_processing.psi[rc]
                            * r.post_processing.xi_r[rc]
                            * dthreeepsilonadepsilon[rc]
                        + q_kl2_hrd
                            * model.rho[rc]
                            * l.post_processing.psi[rc]
                            * r.post_processing.xi_h[rc]
                            * threeepsilonadepsilon[rc]
                            / model.r_coord[rc]
                        - model.grav * model.m_coord[rc] / model.r_coord[rc].powi(2)
                            * rr
                            * model.rho[rc]
                            * q_kl2
                            * dthreeepsilonadepsilon[rc]
                        - model.grav * model.m_coord[rc] / model.r_coord[rc].powi(2)
                            * rh
                            * model.rho[rc]
                            * q_kl2_hrd
                            * threeepsilonadepsilon[rc]
                            / model.r_coord[rc]
                        + model.gamma1[rc]
                            * model.p[rc]
                            * l.post_processing.chi[rc]
                            * (q_kl2 * r.post_processing.xi_r[rc] * dthreeepsilonadepsilon[rc]
                                + q_kl2_hrd
                                    * r.post_processing.xi_h[rc]
                                    * threeepsilonadepsilon[rc]
                                    / model.r_coord[rc]));
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
        .copy_from(&(-&d_zero + &r_zero + &l_zero));
    a.view_range_mut(0..modes.len(), modes.len()..)
        .fill_diagonal(1.);
    a.view_range_mut(modes.len().., modes.len()..)
        .copy_from(&(&d_linear - &r_linear));

    b.view_range_mut(0..modes.len(), 0..modes.len())
        .fill_diagonal(1.);
    b.view_range_mut(modes.len().., modes.len()..)
        .copy_from(&d_squared);

    let mut eigenval = DVector::from_element(modes.len() * 2, Complex64::zero());
    let mut eigenval_scale = DVector::from_element(modes.len() * 2, 0.);
    let mut eigenvectors =
        DMatrix::from_element(modes.len() * 2, modes.len() * 2, Complex64::zero());

    qz::qz(
        &mut a,
        &mut b,
        &mut eigenval,
        &mut eigenval_scale,
        &mut eigenvectors,
    );

    let eigenvalues = eigenval.component_div(&eigenval_scale.map(Complex64::from_real));

    let (eigenvalues, eigenvectors): (Vec<Complex64>, Vec<DVector<Complex64>>) = eigenvalues
        .iter()
        .zip(eigenvectors.column_iter())
        .filter(|&(val, _)| val.real() >= 0.)
        .map(|(val, x)| {
            (
                val,
                x.generic_view((modes.len(), 0), (Dyn(modes.len()), Const::<1>))
                    .clone_owned(),
            )
        })
        .collect();

    ModeCoupling {
        freqs: eigenvalues.into(),
        coupling: DMatrix::from_columns(&eigenvectors),
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
            * rot.powi(2);
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

#[cfg(test)]
mod tests {
    use std::{
        num::NonZeroU64,
        path::{Path, PathBuf},
    };

    use itertools::Itertools;
    use ndarray::linspace;

    use crate::{
        bracket::Precision,
        dynamic_interface::{DifferenceSchemes, MultipleShooting},
        model::StellarModel,
        system::adiabatic::{GridScale, Rotating1D},
    };

    use super::Rotating1DPostprocessing;

    fn compute_mode_id(
        ell: u64,
        m: i64,
        lower: f64,
        upper: f64,
        steps: usize,
        model: impl AsRef<Path>,
    ) -> Vec<(u64, u64, i64)> {
        let model = {
            let main_dir: PathBuf = std::env::var("CARGO_MANIFEST_DIR").unwrap().into();
            let model_file = main_dir.join(model);

            StellarModel::from_gsm(model_file).unwrap()
        };

        let system = Rotating1D::from_model(&model, ell, m);
        let determinant =
            MultipleShooting::new(&system, DifferenceSchemes::Colloc2, &GridScale { scale: 0 });
        let points = linspace(lower, upper, steps);

        determinant
            .scan_and_optimize(
                points,
                Precision::ULP(const { NonZeroU64::new(1).unwrap() }),
            )
            .map(|bracket| {
                let post = Rotating1DPostprocessing::new(
                    bracket.root,
                    &determinant.eigenvector(bracket.root),
                    ell,
                    m,
                    &model,
                );

                (
                    post.cross_clockwise,
                    post.cross_counter_clockwise,
                    post.radial_order,
                )
            })
            .collect()
    }

    #[test]
    fn test_mode_id_radial() {
        assert_eq!(
            compute_mode_id(0, 0, 3.0, 25.0, 25, "test-data/test-model-zams.GSM"),
            (1..=19).map(|i| (0, i, i as i64)).collect_vec()
        );
    }

    #[test]
    fn test_mode_id_radial_other() {
        assert_eq!(
            compute_mode_id(0, 0, 2.0, 25.0, 25, "test-data/joel-test-model.GSM"),
            (1..=17).map(|i| (0, i, i as i64)).collect_vec()
        );
    }

    #[test]
    fn test_mode_id_dipole() {
        assert_eq!(
            compute_mode_id(1, 0, 1., 25., 80, "test-data/test-model-zams.GSM"),
            vec![
                (2, 0, -2),
                (1, 0, -1),
                (0, 0, 1),
                (0, 1, 2),
                (0, 2, 3),
                (0, 3, 4),
                (0, 4, 5),
                (0, 5, 6),
                (0, 6, 7),
                (0, 7, 8),
                (0, 8, 9),
                (0, 9, 10),
                (0, 10, 11),
                (0, 11, 12),
                (0, 12, 13),
                (0, 13, 14),
                (0, 14, 15),
                (0, 15, 16),
                (0, 16, 17),
                (0, 17, 18),
                (0, 18, 19)
            ]
        )
    }
    #[test]
    fn test_mode_id_quadrupole() {
        assert_eq!(
            compute_mode_id(2, 0, 2.0, 25.0, 80, "test-data/test-model-tams.GSM"),
            vec![
                (3, 0, -3),
                (2, 0, -2),
                (1, 0, -1),
                (1, 1, 0),
                (1, 2, 1),
                (1, 3, 2),
                (1, 4, 3),
                (0, 4, 4),
                (0, 5, 5),
                (0, 6, 6),
                (0, 7, 7),
                (0, 8, 8),
                (0, 9, 9),
                (0, 10, 10),
                (0, 11, 11),
                (0, 12, 12),
                (0, 13, 13),
                (0, 14, 14),
                (0, 15, 15),
                (0, 16, 16),
                (0, 17, 17),
                (0, 18, 18)
            ]
        );
    }
}
