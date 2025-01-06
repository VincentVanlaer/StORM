//! Provides post-processing routines to obtain displacement functions, mode-ids, ...

use std::f64::consts::PI;

use crate::model::StellarModel;

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

        let mut y1 = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut y2 = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut y2_other = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut y3 = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut y4 = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut xi_r = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut xi_h = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut p_prime = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut phi_prime = vec![0.; eigenvector.len() / 4].into_boxed_slice();
        let mut dphi_prime = vec![0.; eigenvector.len() / 4].into_boxed_slice();

        let freq_scale = (model.grav * model.mass / model.radius.powi(3)).sqrt();

        let m = m as f64;
        let ell_i32: i32 = ell
            .try_into()
            .expect("ell is never going to be so big to cause problems here");

        for i in 1..y1.len() {
            y1[i] = eigenvector[i * 4];
            y2[i] = eigenvector[i * 4 + 1];
            y3[i] = eigenvector[i * 4 + 2];
            y4[i] = eigenvector[i * 4 + 3];

            let dphi = model.grav * model.m_coord[i] / model.r_coord[i].powi(2);

            xi_r[i] = y1[i] * model.r_coord[i].powi(ell_i32 - 1) / model.radius.powi(ell_i32 - 2);
            p_prime[i] = y2[i] * model.rho[i] * dphi * model.r_coord[i].powi(ell_i32 - 1)
                / model.radius.powi(ell_i32 - 2);
            phi_prime[i] =
                y3[i] * dphi * model.r_coord[i].powi(ell_i32 - 1) / model.radius.powi(ell_i32 - 2);
            dphi_prime[i] =
                y4[i] * dphi * model.r_coord[i].powi(ell_i32 - 2) / model.radius.powi(ell_i32 - 2);

            y2_other[i] = (p_prime[i] / model.rho[i] + phi_prime[i]) / (model.r_coord[i] * dphi);

            let rsigma = freq * freq_scale - m * model.rot[i];
            if ell != 0 {
                let f = 2. * m * model.rot[i] / (ell * (ell + 1)) as f64;

                xi_h[i] = 1. / (rsigma * model.rho[i] * (rsigma + f))
                    * ((p_prime[i] + model.rho[i] * phi_prime[i]) / model.r_coord[i]
                        - f * model.rho[i] * xi_r[i]);
            } else {
                xi_h[i] = 0.;
            }
        }

        // Handle central point
        y1[0] = eigenvector[0];
        y2[0] = eigenvector[1];
        y3[0] = eigenvector[2];
        y4[0] = eigenvector[3];
        y2_other[0] = 0.;

        if ell != 1 {
            xi_r[0] = 0.;
            dphi_prime[0] = 0.;
            xi_h[0] = 0.;
        } else {
            let ddphi0 = 4. / 3. * std::f64::consts::PI * model.rho[0] * model.grav;
            xi_r[0] = y1[0] * model.radius;
            dphi_prime[0] = y4[0] * ddphi0 * model.radius;

            let rsigma = freq * freq_scale - model.rot[0];
            let f = 2. * m * model.rot[0] / (ell * (ell + 1)) as f64;

            xi_h[0] = 1. / (rsigma * (rsigma + f))
                * ((y2[0] + y3[0]) * ddphi0 * model.radius - f * xi_r[0]);
        }

        p_prime[0] = 0.;
        phi_prime[0] = 0.;

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
            x: model.r_coord.as_slice().unwrap().to_owned().into(),
            y1,
            y2,
            y3,
            y4,
            xi_r,
            xi_h,
            clockwise_winding: cww / PI,
            counter_clockwise_winding: ccww / PI,
        }
    }
}
