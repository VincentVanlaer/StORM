//! Provides post-processing routines to obtain displacement functions, mode-ids, ...

use std::f64::consts::PI;

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
            let f;
            if ell != 0 {
                f = 2. * m * model.rot[0] / (ell * (ell + 1)) as f64;
            } else {
                f = 0.;
            }

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
