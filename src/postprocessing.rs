//! Provides post-processing routines to obtain displacement functions, mode-ids, ...

use std::f64::consts::PI;

use itertools::Itertools;

use crate::{
    gaunt::{q_kl1_h, q_kl1_hd},
    model::DiscreteModelSegment,
};

/// Result from the post processing of a solution to the 1D oscillation equations
#[derive(Debug, Clone)]
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
    /// Locations of the radial nodes
    pub nodes: Box<[f64]>,
}

impl Rotating1DPostprocessing {
    /// Post-process the results
    pub fn new(
        freq: f64,
        eigenvector: &[f64],
        ell: u64,
        m: i64,
        model: &[DiscreteModelSegment],
    ) -> Rotating1DPostprocessing {
        assert!(eigenvector.len().is_multiple_of(4));
        let total_len = model.iter().map(|s| s.dimensionless.r_coord.len()).sum();
        assert_eq!(total_len * 4, eigenvector.len());

        let mut r_coord = vec![0.; total_len].into_boxed_slice();
        let mut y1 = vec![0.; total_len].into_boxed_slice();
        let mut y2 = vec![0.; total_len].into_boxed_slice();
        let mut y3 = vec![0.; total_len].into_boxed_slice();
        let mut y4 = vec![0.; total_len].into_boxed_slice();
        let mut xi_r = vec![0.; total_len].into_boxed_slice();
        let mut xi_h = vec![0.; total_len].into_boxed_slice();
        let mut xi_tn = vec![0.; total_len].into_boxed_slice();
        let mut xi_tp = vec![0.; total_len].into_boxed_slice();
        let mut p_prime = vec![0.; total_len].into_boxed_slice();
        let mut psi_prime = vec![0.; total_len].into_boxed_slice();
        let mut dpsi_prime = vec![0.; total_len].into_boxed_slice();
        let mut rho_prime = vec![0.; total_len].into_boxed_slice();
        let mut chi = vec![0.; total_len].into_boxed_slice();

        // Required for determining the winding numbers of dipole modes
        let mut u = vec![0.; total_len].into_boxed_slice();

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
        let mut idx = 0;

        for s in model {
            let s = &s.dimensionless;

            let trapezoid = {
                let mut trapezoid = vec![0.; s.r_coord.len()];

                trapezoid[0] = 0.5 * (s.r_coord[1] - s.r_coord[0]);
                trapezoid[s.r_coord.len() - 1] =
                    0.5 * (s.r_coord[s.r_coord.len() - 1] - s.r_coord[s.r_coord.len() - 2]);

                for i in 1..(s.r_coord.len() - 1) {
                    trapezoid[i] = 0.5 * (s.r_coord[i + 1] - s.r_coord[i - 1]);
                }

                trapezoid
            };

            for p_idx in 0..s.r_coord.len() {
                r_coord[idx] = s.r_coord[p_idx];
                u[idx] = s.u[p_idx];
                y1[idx] = eigenvector[idx * 4];
                y2[idx] = eigenvector[idx * 4 + 1];
                y3[idx] = eigenvector[idx * 4 + 2];
                y4[idx] = eigenvector[idx * 4 + 3];

                let dphi = s.m_coord[p_idx] / s.r_coord[p_idx].powi(2);

                xi_r[idx] = y1[idx] * s.r_coord[p_idx].powi(ell_i32 - 1);
                p_prime[idx] = y2[idx] * s.rho[p_idx] * dphi * s.r_coord[p_idx].powi(ell_i32 - 1);
                psi_prime[idx] = y3[idx] * dphi * s.r_coord[p_idx].powi(ell_i32 - 1);
                dpsi_prime[idx] = y4[idx] * dphi * s.r_coord[p_idx].powi(ell_i32 - 2);

                let rsigma = freq - mf * s.rot[p_idx];
                let omega_rsq;
                let rel_rot;

                if ell != 0 {
                    let rot = mf * s.rot[p_idx];
                    omega_rsq = (lambda * (freq - rot) + 2. * rot) * (freq - rot);
                    rel_rot = 2. * rot / (lambda * (freq - rot) + 2. * rot);

                    let f = 2. * rot / (ell * (ell + 1)) as f64;
                    xi_h[idx] = 1. / (rsigma * s.rho[p_idx] * (rsigma + f))
                        * ((p_prime[idx] + s.rho[p_idx] * psi_prime[idx]) / s.r_coord[p_idx]
                            - f * rsigma * s.rho[p_idx] * xi_r[idx]);
                } else {
                    omega_rsq = 1.;
                    rel_rot = 0.;
                    xi_h[idx] = 0.;
                }

                let xdy1 = (s.v[p_idx] / s.gamma1[p_idx] - 1. - ell as f64 - lambda * rel_rot)
                    * y1[idx]
                    + (-s.v[p_idx] / s.gamma1[p_idx] + lambda.powi(2) / (omega_rsq * s.c1[p_idx]))
                        * y2[idx]
                    + lambda.powi(2) / (omega_rsq * s.c1[p_idx]) * y3[idx];
                chi[idx] = s.r_coord[p_idx].powi(ell_i32 - 2)
                    * ((ell as f64 + 1.) * y1[idx] + xdy1)
                    - lambda / s.r_coord[p_idx] * xi_h[idx];
                rho_prime[idx] = -s.rho[p_idx] * chi[idx]
                    + xi_r[idx] * s.rho[p_idx] / s.r_coord[p_idx]
                        * (s.v[p_idx] / s.gamma1[p_idx] + s.a_star[p_idx]);

                if m.unsigned_abs() == ell || ell == 1 {
                    xi_tn[idx] = 0.;
                } else {
                    xi_tn[idx] = 2. * s.rot[p_idx]
                        / (lambda_n1 * (freq - mf * s.rot[p_idx]) + 2. * mf * s.rot[p_idx])
                        * (-q_hd_n * xi_r[idx] + q_h_n * xi_h[idx]);
                }

                xi_tp[idx] = 2. * s.rot[p_idx]
                    / (lambda_p1 * (freq - mf * s.rot[p_idx]) + 2. * mf * s.rot[p_idx])
                    * (-q_hd_p * xi_r[idx] + q_h_p * xi_h[idx]);

                if s.r_coord[p_idx] != 0. {
                    norm += s.rho[p_idx]
                        * s.r_coord[p_idx].powi(2)
                        * (xi_r[idx] * xi_r[idx] + lambda * xi_h[idx] * xi_h[idx])
                        * trapezoid[p_idx];
                }

                idx += 1;
            }
        }

        // Handle central point
        let s = &model[0].dimensionless;

        if ell != 1 {
            xi_r[0] = 0.;
            dpsi_prime[0] = 0.;
            xi_h[0] = 0.;
        } else {
            let ddphi0 = 4. / 3. * std::f64::consts::PI * s.rho[0];
            xi_r[0] = y1[0];
            dpsi_prime[0] = y4[0] * ddphi0;

            let rsigma = freq - mf * s.rot[0];
            let f = if ell != 0 {
                2. * mf * s.rot[0] / (ell * (ell + 1)) as f64
            } else {
                0.
            };

            xi_h[0] = 1. / (rsigma * (rsigma + f)) * ((y2[0] + y3[0]) * ddphi0 - f * xi_r[0]);
        }

        if ell == 0 {
            p_prime[0] = y2[0] * s.rho[0].powi(2) * 4. / 3. * PI;
            psi_prime[0] = y3[0] * s.rho[0] * 4. / 3. * PI;
            rho_prime[0] = s.rho[0] * p_prime[0] / (s.gamma1[0] * s.p[0]);
            chi[0] = -rho_prime[0] / s.rho[0];
        } else {
            p_prime[0] = 0.;
            psi_prime[0] = 0.;
            rho_prime[0] = 0.;
            chi[0] = 0.;
        }

        if m.unsigned_abs() == ell || ell == 1 {
            xi_tn[0] = 0.;
        } else {
            xi_tn[0] = 2. * s.rot[0] / (lambda_n1 * (freq - mf * s.rot[0]) + 2. * mf * s.rot[0])
                * (-q_hd_n * xi_r[0] + q_h_n * xi_h[0]);
        }

        if ell == 0 {
            xi_tp[0] = 0.;
        } else {
            xi_tp[0] = 2. * s.rot[0] / (lambda_p1 * (freq - mf * s.rot[0]) + 2. * mf * s.rot[0])
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

            #[cfg(test)]
            {
                assert!(y1[i].is_finite());
                assert!(y2[i].is_finite());
                assert!(y3[i].is_finite());
                assert!(y4[i].is_finite());
                assert!(xi_r[i].is_finite());
                assert!(xi_h[i].is_finite());
                assert!(xi_tn[i].is_finite());
                assert!(xi_tp[i].is_finite());
                assert!(psi_prime[i].is_finite());
                assert!(dpsi_prime[i].is_finite());
                assert!(rho_prime[i].is_finite());
                assert!(p_prime[i].is_finite());
                assert!(chi[i].is_finite());
            }
        }

        let (cross_clockwise, cross_counter_clockwise, radial_order, nodes) = match ell {
            0 => {
                let (cw, ccw, nodes) = count_windings(&y1, &y2, &r_coord);

                (cw, ccw, ccw as i64, nodes)
            }
            1 => {
                let mut y1_alt = vec![0.; y1.len()].into_boxed_slice();
                let mut y2_alt = vec![0.; y1.len()].into_boxed_slice();

                for i in 0..y2_alt.len() {
                    y1_alt[i] = (1. - u[i] / 3.) * y1[i] + (y3[i] - y4[i]) / 3.;
                    y2_alt[i] = y2[i] - y1[i];
                }

                let (cw, ccw, nodes) = count_windings(&y1_alt[2..], &y2_alt[2..], &r_coord[2..]);

                if cw > ccw {
                    (cw, ccw, ccw as i64 - cw as i64, nodes)
                } else {
                    (cw, ccw, ccw as i64 - cw as i64 + 1, nodes)
                }
            }
            _ => {
                let mut y2_alt = vec![0.; y1.len()].into_boxed_slice();

                for i in 0..y2_alt.len() {
                    y2_alt[i] = y2[i] + y3[i];
                }

                let (cw, ccw, nodes) = count_windings(&y1, &y2_alt, &r_coord);

                (cw, ccw, ccw as i64 - cw as i64, nodes)
            }
        };

        Rotating1DPostprocessing {
            x: r_coord,
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
            nodes,
        }
    }
}

fn count_windings(y1: &[f64], y2: &[f64], x: &[f64]) -> (u64, u64, Box<[f64]>) {
    let mut clockwise = 0;
    let mut counter_clockwise = 0;
    let mut nodes = Vec::new();

    #[cfg(test)]
    eprintln!("---");
    y1.iter()
        .zip(y2.iter())
        .zip(x.iter())
        .tuple_windows()
        .enumerate()
        .for_each(|(_i, (((&y1_1, &y2_1), &x_1), ((&y1_2, &y2_2), &x_2)))| {
            if y1_1 <= 0. && y1_2 > 0. {
                // left to right
                let yt = y2_1 - y1_1 * (y2_2 - y2_1) / (y1_2 - y1_1);
                let xt = x_2 - y1_1 * (x_2 - x_1) / (y1_2 - y1_1);
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
                nodes.push(xt);
            } else if y1_1 >= 0. && y1_2 < 0. {
                // right to left
                let yt = y2_1 - y1_1 * (y2_2 - y2_1) / (y1_2 - y1_1);
                let xt = x_2 - y1_1 * (x_2 - x_1) / (y1_2 - y1_1);
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
                nodes.push(xt);
            }
        });

    #[cfg(test)]
    eprintln!("--- cw: {clockwise} ccw: {counter_clockwise} ---");
    (clockwise, counter_clockwise, nodes.into())
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
        dynamic_interface::{DifferenceSchemes, ErasedSolver},
        model::{DiscreteModel, interpolate::LinearInterpolator, loader::ModelFormat},
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

            DiscreteModel::load(&model_file, ModelFormat::GSM).unwrap()
        };

        let determinant = ErasedSolver::new(
            &LinearInterpolator::new(&model),
            ell,
            m,
            DifferenceSchemes::Colloc2,
            &model
                .segments
                .iter()
                .map(|x| &*x.dimensionless.r_coord)
                .collect_vec(),
        );
        let points = linspace(lower, upper, steps);

        determinant
            .scan_and_optimize(
                points,
                Precision::ULP(const { NonZeroU64::new(1).unwrap() }),
            )
            .into_iter()
            .map(|bracket| {
                let post = Rotating1DPostprocessing::new(
                    bracket.root,
                    &determinant.eigenvector(bracket.root),
                    ell,
                    m,
                    &model.segments,
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
