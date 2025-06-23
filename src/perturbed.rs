//! Provides routines for implementing higher order effects of rotation, such as the toroidal
//! components and non-spherical deformation

use std::f64::consts::PI;

use nalgebra::{ComplexField, Const, DMatrix, DVector, Dyn, Matrix2, Vector2};
use num_complex::Complex64;
use num_traits::Zero;

use crate::{
    gaunt::{q_kl1_h, q_kl1_hd, q_kl2, q_kl2_h, q_kl2_hd},
    linalg::qz,
    model::DiscreteModel,
    postprocessing::Rotating1DPostprocessing,
};

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
    DiscreteModel {
        dimensionless: model,
        ..
    }: &DiscreteModel,
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

        trapezoid[0] = 0.5 * (model.r_coord[1] - model.r_coord[0]);
        trapezoid[model.r_coord.len() - 1] =
            0.5 * (model.r_coord[model.r_coord.len() - 1] - model.r_coord[model.r_coord.len() - 2]);

        for i in 1..(model.r_coord.len() - 1) {
            trapezoid[i] = 0.5 * (model.r_coord[i + 1] - model.r_coord[i - 1]);
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
                        - model.m_coord[rc] / model.r_coord[rc].powi(2)
                            * rr
                            * model.rho[rc]
                            * q_kl2
                            * dthreeepsilonadepsilon[rc]
                        - model.m_coord[rc] / model.r_coord[rc].powi(2)
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
pub fn perturb_structure(
    DiscreteModel {
        dimensionless: model,
        ..
    }: &DiscreteModel,
    rot: f64,
) -> PerturbedMetric {
    let mut y = vec![Vector2::new(0., 0.); model.r_coord.len()];

    y[1] = Vector2::new(1., 2.);

    for i in 2..model.r_coord.len() {
        let delta = model.r_coord[i] - model.r_coord[i - 1];
        let x_12 = 0.5 * (model.r_coord[i] + model.r_coord[i - 1]);

        let k = 4. * PI / model.m_coord[i]
            * model.rho[i]
            * model.r_coord[i]
            * (-model.a_star[i] + model.v[i] / model.gamma1[i]);

        let k_prev = 4. * PI / model.m_coord[i - 1]
            * model.rho[i - 1]
            * model.r_coord[i - 1]
            * (-model.a_star[i - 1] + model.v[i - 1] / model.gamma1[i - 1]);

        let a = 0.5 * delta / x_12
            * Matrix2::new(
                -1.,
                1.,
                6. + 0.5 * model.r_coord[i].powi(2) * (k + k_prev),
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

        let k = 4. * PI / model.m_coord[i]
            * model.rho[i]
            * model.r_coord[i]
            * (-model.a_star[i] + model.v[i] / model.gamma1[i]);
        let ddpsi = ((6. + model.r_coord[i].powi(2) * k) * y[i].x - 2. * y[i].y) / model.r_coord[i];

        beta[i] =
            2. * model.r_coord[i] / model.m_coord[i] * a2 * y[i].x * model.r_coord[i] * rot.powi(2);
        dbeta[i] = beta[i] / model.r_coord[i] - beta[i] * dmda
            + beta[i] / (y[i].x * model.r_coord[i]) * y[i].y;
        ddbeta[i] = -2. * beta[i] / model.r_coord[i] * dmda
            + 2. * beta[i] / (y[i].x * model.r_coord[i].powi(2)) * y[i].y
            - 2. * beta[i] / (y[i].x * model.r_coord[i]) * dmda * y[i].y
            + 2. * beta[i] * dmda.powi(2)
            + beta[i] / (y[i].x * model.r_coord[i]) * ddpsi
            - beta[i] / model.m_coord[i] * 8. * PI * model.r_coord[i] * model.rho[i]
            - beta[i] * dmda / model.r_coord[i] * (-model.a_star[i] + model.v[i] / model.gamma1[i]);
    }

    PerturbedMetric {
        beta: beta.into_boxed_slice(),
        dbeta: dbeta.into_boxed_slice(),
        ddbeta: ddbeta.into_boxed_slice(),
        rot,
    }
}
