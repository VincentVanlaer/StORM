//! Provides routines for implementing higher order effects of rotation, such as the toroidal
//! components and non-spherical deformation

use std::f64::consts::PI;

use nalgebra::{ComplexField, Const, DMatrix, DVector, Dyn, Matrix2, Matrix3, Vector2};
use num_complex::Complex64;
use num_traits::Zero;

use crate::{
    gaunt::{q_kl1_h, q_kl1_hd, q_kl2, q_kl2_h, q_kl2_hd},
    linalg::qz,
    model::{DiscreteModel, PerturbedMetric},
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
        alpha,
        dalpha,
        ddalpha,
        beta,
        dbeta,
        ddbeta,
        rot: _,
        mass_delta: _,
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

    let epsilona = alpha.to_owned();
    let mut adepsilona = vec![0.; alpha.len()];
    let mut threeepsilonadepsilona = vec![0.; alpha.len()];
    let mut adthreeepsilonadepsilona = vec![0.; alpha.len()];

    for i in 0..alpha.len() {
        adepsilona[i] = dalpha[i];
        threeepsilonadepsilona[i] = 3. * epsilona[i] + adepsilona[i];
        adthreeepsilonadepsilona[i] = 3. * dalpha[i] + ddalpha[i];
    }

    let epsilonb = beta.to_owned();
    let mut adepsilonb = vec![0.; beta.len()];
    let mut threeepsilonadepsilonb = vec![0.; beta.len()];
    let mut adthreeepsilonadepsilonb = vec![0.; beta.len()];

    for i in 0..beta.len() {
        adepsilonb[i] = dbeta[i];
        threeepsilonadepsilonb[i] = 3. * epsilonb[i] + adepsilonb[i];
        adthreeepsilonadepsilonb[i] = 3. * dbeta[i] + ddbeta[i];
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
                q_rt_n = q_kl1_hd(l.ell, r.ell.saturating_sub(1), m);
                q_ht_p = -q_kl1_h(l.ell, r.ell + 1, m);
                q_ht_n = -q_kl1_h(l.ell, r.ell.saturating_sub(1), m);
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
                    * (rr * inner_prod_r * (1. + 2. * (epsilona[rc] + adepsilona[rc]))
                        + hh * inner_prod_h * (1. + 2. * epsilona[rc])
                        + rr * 2. * q_kl2 * (epsilonb[rc] + adepsilonb[rc])
                        + (rh * q_kl2_hrd + hr * q_kl2_hld + hh * 2. * q_kl2_h) * epsilonb[rc]);

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
                    * (inner_prod_r
                        * ((hr + rh) * (1. + 2. * epsilona[rc] + adepsilona[rc])
                            + hh * (1. + 2. * epsilona[rc]))
                        + (rh + hr) * q_kl2 * (2. * epsilonb[rc] + adepsilonb[rc])
                        + hh * (inner_prod_r + 6. * q_kl2) * 2. * epsilonb[rc]);

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
                assert!(
                    val_t.is_finite(),
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
                        + l.post_processing.psi[rc]
                            * r.post_processing.rho[rc]
                            * threeepsilonadepsilona[rc]
                        - model.rho[rc]
                            * l.post_processing.psi[rc]
                            * r.post_processing.xi_r[rc]
                            * adthreeepsilonadepsilona[rc]
                            / model.r_coord[rc]
                        + l.post_processing.psi[rc]
                            * q_kl2
                            * r.post_processing.rho[rc]
                            * threeepsilonadepsilonb[rc]
                        - model.rho[rc]
                            * l.post_processing.psi[rc]
                            * q_kl2
                            * r.post_processing.xi_r[rc]
                            * adthreeepsilonadepsilonb[rc]
                            / model.r_coord[rc]
                        - l.post_processing.p[rc]
                            * r.post_processing.xi_r[rc]
                            * adthreeepsilonadepsilona[rc]
                            / model.r_coord[rc]
                        - l.post_processing.p[rc]
                            * q_kl2
                            * r.post_processing.xi_r[rc]
                            * adthreeepsilonadepsilonb[rc]
                            / model.r_coord[rc]
                        - l.post_processing.p[rc]
                            * q_kl2_hrd
                            * r.post_processing.xi_h[rc]
                            * threeepsilonadepsilonb[rc]
                            / model.r_coord[rc]
                        + q_kl2_hrd
                            * model.rho[rc]
                            * l.post_processing.psi[rc]
                            * r.post_processing.xi_h[rc]
                            * threeepsilonadepsilonb[rc]
                            / model.r_coord[rc]);
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

/// Deform the stellar structure of a model for a give rotation frequency
pub fn perturb_structure(
    DiscreteModel {
        dimensionless: model,
        ..
    }: &DiscreteModel,
    rot: f64,
) -> PerturbedMetric {
    let mut y0 = vec![Vector2::new(0., 0.); model.r_coord.len()];
    let mut y2 = vec![Vector2::new(0., 0.); model.r_coord.len()];
    let diag2 = Matrix2::from_diagonal_element(1.);
    let diag3 = Matrix3::from_diagonal_element(1.);

    y2[1] = Vector2::new(1., 2.);

    for i in 1..model.r_coord.len() {
        let delta = model.r_coord[i] - model.r_coord[i - 1];
        let x_12 = 0.5 * (model.r_coord[i] + model.r_coord[i - 1]);

        let k = 4. * PI / model.m_coord[i]
            * model.rho[i]
            * model.r_coord[i]
            * (-model.a_star[i] - model.v[i] / model.gamma1[i]);

        let k_prev = if i == 1 {
            0.
        } else {
            4. * PI / model.m_coord[i - 1]
                * model.rho[i - 1]
                * model.r_coord[i - 1]
                * (-model.a_star[i - 1] - model.v[i - 1] / model.gamma1[i - 1])
        };

        let a0 = 0.5 * delta / x_12
            * Matrix3::new(
                -1.,
                1.,
                0.,
                0.5 * x_12.powi(2) * (k + k_prev),
                -2.,
                x_12,
                0.,
                0.,
                0.,
            );

        let step = nalgebra::Matrix3::try_inverse(diag3 - a0).unwrap() * (diag3 + a0);

        y0[i] = step.generic_view((0, 0), (Const::<2>, Const::<2>)) * y0[i - 1]
            + step.generic_view((0, 2), (Const::<2>, Const::<1>));

        if i != 1 {
            let a2 = 0.5 * delta / x_12
                * Matrix2::new(-1., 1., 6. + 0.5 * x_12.powi(2) * (k + k_prev), -2.);
            let step = nalgebra::Matrix2::try_inverse(diag2 - a2).unwrap() * (diag2 + a2);

            y2[i] = step * y2[i - 1];
        }
    }

    let upper = y2.last().unwrap();

    let a2 = -5. / 6. / (3. * upper.x + upper.y);

    let mut alpha = vec![0.; model.r_coord.len()];
    let mut dalpha = vec![0.; model.r_coord.len()];
    let mut ddalpha = vec![0.; model.r_coord.len()];

    let mut beta = vec![0.; model.r_coord.len()];
    let mut dbeta = vec![0.; model.r_coord.len()];
    let mut ddbeta = vec![0.; model.r_coord.len()];
    let mut mass_delta = 0.;

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

    for i in 1..beta.len() {
        let dlnrhodlna = -model.a_star[i] - model.v[i] / model.gamma1[i];
        let k = 4. * PI / model.m_coord[i] * model.rho[i] * model.r_coord[i] * dlnrhodlna;

        let psi = y0[i].x * model.r_coord[i];
        let dpsi = y0[i].y;
        let ddpsi = (model.r_coord[i].powi(2) * k * y0[i].x - 2. * y0[i].y) / model.r_coord[i] + 1.;

        alpha[i] = 2. * model.r_coord[i] / model.m_coord[i] * psi * rot.powi(2);
        dalpha[i] = alpha[i] * (1. - model.u[i] + model.r_coord[i] * dpsi / psi);
        ddalpha[i] = dalpha[i] * (1. - model.u[i] + model.r_coord[i] * dpsi / psi)
            + alpha[i]
                * (-3. * model.u[i] - model.u[i] * dlnrhodlna
                    + model.u[i].powi(2)
                    + model.r_coord[i] * dpsi / psi
                    - (model.r_coord[i] * dpsi / psi).powi(2)
                    + model.r_coord[i].powi(2) * ddpsi / psi);

        let psi = y2[i].x * model.r_coord[i] * a2;
        let dpsi = y2[i].y * a2;
        let ddpsi =
            ((6. + model.r_coord[i].powi(2) * k) * y2[i].x - 2. * y2[i].y) / model.r_coord[i] * a2;

        beta[i] = 2. * model.r_coord[i] / model.m_coord[i] * psi * rot.powi(2);
        dbeta[i] = beta[i] * (1. - model.u[i] + model.r_coord[i] * dpsi / psi);
        ddbeta[i] = dbeta[i] * (1. - model.u[i] + model.r_coord[i] * dpsi / psi)
            + beta[i]
                * (-3. * model.u[i] - model.u[i] * dlnrhodlna
                    + model.u[i].powi(2)
                    + model.r_coord[i] * dpsi / psi
                    - (model.r_coord[i] * dpsi / psi).powi(2)
                    + model.r_coord[i].powi(2) * ddpsi / psi);
        mass_delta +=
            trapezoid[i] * model.rho[i] * model.r_coord[i].powi(2) * (3. * alpha[i] + dalpha[i]);
    }

    mass_delta *= 4. * PI;

    // TODO: check central point (not that it will have a significant influence)

    PerturbedMetric {
        alpha: alpha.into(),
        dalpha: dalpha.into(),
        ddalpha: ddalpha.into(),
        beta: beta.into(),
        dbeta: dbeta.into(),
        ddbeta: ddbeta.into(),
        rot,
        mass_delta,
    }
}

#[cfg(test)]
mod tests {
    use itertools::{Itertools, izip};
    use nalgebra::ComplexField;

    use crate::{
        bracket::Precision,
        dynamic_interface::{DifferenceSchemes, ErasedSolver},
        model::{PerturbedMetric, interpolate::LinearInterpolator, polytrope::construct_polytrope},
        perturbed::perturb_structure,
        postprocessing::Rotating1DPostprocessing,
        system::adiabatic::Rotating1D,
    };

    use super::{ModeToPerturb, perturb_deformed};

    fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
        (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
    }
    const ROT: f64 = 0.01;

    #[ignore = "used for output"]
    #[test]
    /// This test is a comparison with the results of Saio (1981). Because StORM does have the same
    /// form of the expressions and does coupling between modes, this will not exactly reproduce
    /// the results. However, they should be more or less the same.
    fn perturbed_asymptotics() {
        let mut poly3 = construct_polytrope(3., 5. / 3., 0.001);

        // Low rotation rate to supress higher order effects
        poly3.dimensionless.rot.fill(ROT);

        let deformed = perturb_structure(&poly3, ROT);

        let runner = |ell: u64, scan: &[f64]| {
            let solver = ErasedSolver::new(
                &LinearInterpolator::new(&poly3),
                Rotating1D::new(ell, 0),
                DifferenceSchemes::Colloc6,
                &poly3.dimensionless.r_coord,
            );

            let solutions = solver
                .scan_and_optimize(scan.iter().cloned(), Precision::Relative(1e-10))
                .collect_vec();

            let post_processing = solutions
                .iter()
                .map(|x| {
                    Rotating1DPostprocessing::new(
                        x.root,
                        &solver.eigenvector(x.root),
                        ell,
                        0,
                        &poly3,
                    )
                })
                .collect_vec();

            let radial_orders = post_processing.iter().map(|x| x.radial_order).collect_vec();

            // First check the toroidal contributions
            let result_tor = perturb_deformed(
                &poly3,
                &solutions
                    .iter()
                    .zip(post_processing.iter())
                    .map(|(sol, post)| ModeToPerturb {
                        ell,
                        freq: sol.root,
                        post_processing: post,
                    })
                    .collect_vec(),
                0,
                &PerturbedMetric {
                    alpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    dalpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    ddalpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    beta: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    dbeta: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    ddbeta: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    rot: 0.,
                    mass_delta: 0.,
                },
            );

            let mut perturbed_freqs_tor = result_tor
                .freqs
                .iter()
                .map(|complex| complex.real())
                .collect_vec();
            perturbed_freqs_tor.sort_by(f64::total_cmp);

            let x1 = perturbed_freqs_tor
                .iter()
                .zip(solutions.iter())
                .map(|(freq, sol)| (freq - sol.root) / ROT.powi(2) * sol.root)
                .collect_vec();

            // Now check the deformation contributions
            let post_processing_no_tor = post_processing
                .iter()
                .map(|x| {
                    let mut x = x.clone();
                    x.xi_tp = vec![0.; x.x.len()].into();
                    x.xi_tn = vec![0.; x.x.len()].into();
                    x
                })
                .collect_vec();

            let result_def_s = perturb_deformed(
                &poly3,
                &solutions
                    .iter()
                    .zip(post_processing_no_tor.iter())
                    .map(|(sol, post)| ModeToPerturb {
                        ell,
                        freq: sol.root,
                        post_processing: post,
                    })
                    .collect_vec(),
                0,
                &PerturbedMetric {
                    alpha: deformed.alpha.clone(),
                    dalpha: deformed.dalpha.clone(),
                    ddalpha: deformed.ddalpha.clone(),
                    beta: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    dbeta: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    ddbeta: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    rot: ROT,
                    mass_delta: 0.,
                },
            );

            let mut perturbed_freqs_def_s = result_def_s
                .freqs
                .iter()
                .map(|complex| complex.real())
                .collect_vec();
            perturbed_freqs_def_s.sort_by(f64::total_cmp);

            let z = perturbed_freqs_def_s
                .iter()
                .zip(solutions.iter())
                .map(|(freq, sol)| (freq - sol.root) / ROT.powi(2) * sol.root)
                .collect_vec();

            let result_def_n = perturb_deformed(
                &poly3,
                &solutions
                    .iter()
                    .zip(post_processing_no_tor.iter())
                    .map(|(sol, post)| ModeToPerturb {
                        ell,
                        freq: sol.root,
                        post_processing: post,
                    })
                    .collect_vec(),
                0,
                &PerturbedMetric {
                    alpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    dalpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    ddalpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                    beta: deformed.beta.clone(),
                    dbeta: deformed.dbeta.clone(),
                    ddbeta: deformed.ddbeta.clone(),
                    rot: ROT,
                    mass_delta: 0.,
                },
            );

            let mut perturbed_freqs_def_n = result_def_n
                .freqs
                .iter()
                .map(|complex| complex.real())
                .collect_vec();
            perturbed_freqs_def_n.sort_by(f64::total_cmp);

            let x2 = perturbed_freqs_def_n
                .iter()
                .zip(solutions.iter())
                .map(|(freq, sol)| (freq - sol.root) / ROT.powi(2) * sol.root)
                .collect_vec();

            let c1: Option<Vec<_>>;
            let y2: Option<Vec<_>>;

            if ell != 0 {
                let solver_m1 = ErasedSolver::new(
                    &LinearInterpolator::new(&poly3),
                    Rotating1D::new(ell, 1),
                    DifferenceSchemes::Colloc6,
                    &poly3.dimensionless.r_coord,
                );

                let solutions_m1 = solver_m1
                    .scan_and_optimize(scan.iter().cloned(), Precision::Relative(1e-10))
                    .collect_vec();

                let post_processing_m1 = solutions_m1
                    .iter()
                    .map(|x| {
                        Rotating1DPostprocessing::new(
                            x.root,
                            &solver.eigenvector(x.root),
                            ell,
                            1,
                            &poly3,
                        )
                    })
                    .inspect(|x| {
                        dbg!(x.radial_order);
                    })
                    .collect_vec();

                c1 = Some(
                    solutions_m1
                        .iter()
                        .zip(solutions.iter())
                        .map(|(freq, sol)| 1. - (freq.root - sol.root) / ROT)
                        .collect_vec(),
                );

                let post_processing_no_tor_m1 = post_processing_m1
                    .iter()
                    .map(|x| {
                        let mut x = x.clone();
                        x.xi_tp = vec![0.; x.x.len()].into();
                        x.xi_tn = vec![0.; x.x.len()].into();
                        x
                    })
                    .collect_vec();

                let result_def_n_m1 = perturb_deformed(
                    &poly3,
                    &solutions_m1
                        .iter()
                        .zip(post_processing_no_tor_m1.iter())
                        .map(|(sol, post)| ModeToPerturb {
                            ell,
                            freq: sol.root,
                            post_processing: post,
                        })
                        .collect_vec(),
                    1,
                    &PerturbedMetric {
                        alpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                        dalpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                        ddalpha: vec![0.; poly3.dimensionless.r_coord.len()].into(),
                        beta: deformed.beta.clone(),
                        dbeta: deformed.dbeta.clone(),
                        ddbeta: deformed.ddbeta.clone(),
                        rot: ROT,
                        mass_delta: 0.,
                    },
                );

                let mut perturbed_freqs_def_n_m1 = result_def_n_m1
                    .freqs
                    .iter()
                    .map(|complex| complex.real())
                    .collect_vec();
                perturbed_freqs_def_n_m1.sort_by(f64::total_cmp);

                y2 = Some(
                    perturbed_freqs_def_n_m1
                        .iter()
                        .zip(solutions_m1.iter())
                        .map(|(freq, sol)| (freq - sol.root) / ROT.powi(2) * sol.root)
                        .collect_vec(),
                );
            } else {
                y2 = None;
                c1 = None;
            }

            (
                radial_orders,
                solutions.iter().map(|x| x.root).collect_vec(),
                x1,
                x2,
                z,
                c1,
                y2,
            )
        };

        let (l0_radial_order, l0_sol, l0_x1, l0_x2, l0_z, _l0_c1, _l0_y2) =
            runner(0, &[3., 4., 5., 6.]);
        let (l1_radial_order, l1_sol, l1_x1, l1_x2, l1_z, l1_c1, l1_y2) =
            runner(1, &[0.8, 0.9, 1., 1.5, 2., 3., 4., 5., 6., 8., 9., 10.]);
        let (l2_radial_order, l2_sol, l2_x1, l2_x2, l2_z, l2_c1, l2_y2) =
            runner(2, &linspace(1.3, 11., 100).collect_vec());
        let (l3_radial_order, l3_sol, l3_x1, l3_x2, l3_z, l3_c1, l3_y2) =
            runner(3, &linspace(1.5, 11., 100).collect_vec());

        println!(
            "{:^4} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}",
            "", "ω²", "C1", "X1", "Z", "X2", "Y2"
        );
        println!("{:─<58}", "");
        println!(
            "{:^4} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}",
            "", "", "", "ℓ = 0", "", "", ""
        );
        println!("{:─<58}", "");

        for (n, f, x1, _, z) in izip!(l0_radial_order, l0_sol, l0_x1, l0_x2, l0_z) {
            let n = if n == 1 {
                "F".to_string()
            } else {
                format!("{}H", n - 1)
            };
            println!(
                "{:<4} {:8.3} {:^8} {:8.3} {:8.3} {:^8} {:^8}",
                n,
                f * f,
                "",
                x1,
                z,
                "",
                ""
            );
        }

        println!("{:─<58}", "");
        println!(
            "{:^4} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}",
            "", "", "", "ℓ = 1", "", "", ""
        );
        println!("{:─<58}", "");

        for (n, f, x1, x2, z, c1, y2) in izip!(
            l1_radial_order,
            l1_sol,
            l1_x1,
            l1_x2,
            l1_z,
            l1_c1.unwrap(),
            l1_y2.unwrap()
        )
        .rev()
        {
            let n = if n == 0 {
                "f".to_string()
            } else if n > 0 {
                format!("p{}", n)
            } else {
                format!("g{}", -n)
            };
            println!(
                "{:<4} {:8.3} {:8.3} {:8.3} {:8.3} {:8.3} {:8.3}",
                n,
                f * f,
                c1,
                x1,
                z,
                x2,
                y2 - x2
            );
        }

        println!("{:─<58}", "");
        println!(
            "{:^4} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}",
            "", "", "", "ℓ = 2", "", "", ""
        );
        println!("{:─<58}", "");

        for (n, f, x1, x2, z, c1, y2) in izip!(
            l2_radial_order,
            l2_sol,
            l2_x1,
            l2_x2,
            l2_z,
            l2_c1.unwrap(),
            l2_y2.unwrap()
        )
        .rev()
        {
            let n = if n == 0 {
                "f".to_string()
            } else if n > 0 {
                format!("p{}", n)
            } else {
                format!("g{}", -n)
            };
            println!(
                "{:<4} {:8.3} {:8.3} {:8.3} {:8.3} {:8.3} {:8.3}",
                n,
                f * f,
                c1,
                x1,
                z,
                x2,
                y2 - x2
            );
        }

        println!("{:─<58}", "");
        println!(
            "{:^4} {:^8} {:^8} {:^8} {:^8} {:^8} {:^8}",
            "", "", "", "ℓ = 3", "", "", ""
        );
        println!("{:─<58}", "");

        for (n, f, x1, x2, z, c1, y2) in izip!(
            l3_radial_order,
            l3_sol,
            l3_x1,
            l3_x2,
            l3_z,
            l3_c1.unwrap(),
            l3_y2.unwrap()
        )
        .rev()
        {
            let n = if n == 0 {
                "f".to_string()
            } else if n > 0 {
                format!("p{}", n)
            } else {
                format!("g{}", -n)
            };
            println!(
                "{:<4} {:8.3} {:8.3} {:8.3} {:8.3} {:8.3} {:8.3}",
                n,
                f * f,
                c1,
                x1,
                z,
                x2,
                y2 - x2
            );
        }

        assert!(false);
    }
}
