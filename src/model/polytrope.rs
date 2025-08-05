use std::f64::consts::PI;

use itertools::Itertools;

use super::{ContinuousModel, DimensionedProperties, DimensionlessProperties, DiscreteModel};

fn eval_lane_emden(xi: f64, phi: f64, theta: f64, n: f64) -> (f64, f64) {
    if xi == 0. {
        (theta.powf(n) * xi.powi(2), 0.)
    } else {
        (theta.powf(n) * xi.powi(2), -phi / (xi * xi))
    }
}

fn rk4(xi: f64, phi: f64, theta: f64, n: f64, step: f64) -> (f64, f64) {
    let hstep = step / 2.;

    let k1 = eval_lane_emden(xi, phi, theta, n);
    let k2 = eval_lane_emden(xi + hstep, phi + hstep * k1.0, theta + hstep * k1.1, n);
    let k3 = eval_lane_emden(xi + hstep, phi + hstep * k2.0, theta + hstep * k2.1, n);
    let k4 = eval_lane_emden(xi + step, phi + step * k3.0, theta + step * k3.1, n);

    (
        phi + step / 6. * (k1.0 + 2. * k2.0 + 2. * k3.0 + k4.0),
        theta + step / 6. * (k1.1 + 2. * k2.1 + 2. * k3.1 + k4.1),
    )
}

fn solve_lane_emden(step: f64, n: f64) -> (Box<[f64]>, Box<[f64]>, Box<[f64]>) {
    let mut thetas = Vec::new();
    let mut phis = Vec::new();
    let mut xis = Vec::new();

    let mut theta = 1.;
    let mut phi = 0.;
    let mut nsteps = 0;

    thetas.push(theta);
    phis.push(phi);
    xis.push(0.);

    while theta > 0. {
        (phi, theta) = rk4((nsteps as f64) * step, phi, theta, n, step);
        nsteps += 1;

        thetas.push(theta);
        phis.push(phi);
        xis.push((nsteps as f64) * step);
    }

    (xis.into(), thetas.into(), phis.into())
}

/// Create a polytrope model from the polytropic index, first adiabatic exponent and integration
/// step size.
///
/// The resulting model will be dimensionless
pub fn construct_polytrope(n: f64, gamma1: f64, step: f64) -> DiscreteModel {
    assert!(n < 5.);
    assert!(n >= 0.);

    let (xi, theta, phi) = solve_lane_emden(step, n);

    let mut u = xi
        .iter()
        .zip(theta.iter())
        .zip(phi.iter())
        .map(|((&xi, &theta), &phi)| xi.powi(3) * theta.powf(n) / phi)
        .collect_vec();

    u[0] = 3.;

    let mut v = xi
        .iter()
        .zip(theta.iter())
        .zip(phi.iter())
        .map(|((&xi, &theta), &phi)| (n + 1.) * phi / (xi * theta))
        .collect_vec();

    v[0] = 0.;

    let max_phi: f64 = phi.iter().fold(0., |a, &b| a.max(b));
    let max_xi: f64 = xi.iter().fold(0., |a, &b| a.max(b));

    let mut c1 = xi
        .iter()
        .zip(phi.iter())
        .map(|(&xi, &phi)| xi.powi(3) / max_xi.powi(3) * max_phi / phi)
        .collect_vec();

    c1[0] = max_phi / max_xi.powi(3) * 3.;

    let mut a_star = xi
        .iter()
        .zip(theta.iter())
        .zip(phi.iter())
        .map(|((&xi, &theta), &phi)| ((n + 1.) / gamma1 - n) * phi / (xi * theta))
        .collect_vec();

    a_star[0] = 0.;

    let x = xi.iter().map(|xi| xi / max_xi).collect_vec();

    let mut rho = theta.iter().map(|&theta| theta.powf(n)).collect_vec();
    rho[0] = 1.;

    let dm = rho
        .iter()
        .zip(xi.iter())
        .tuple_windows()
        .map(|((r1, x1), (r2, x2))| (r2 + r1) * 0.5 * (x2 - x1) / max_xi)
        .collect_vec();

    let m_coord = [0.]
        .into_iter()
        .chain(dm.iter().scan(0., |sum, x| {
            *sum += x;
            Some(*sum)
        }))
        .collect_vec();
    let total_m = m_coord.last().unwrap();
    let k = 4. * PI * rho[0].powf(1. / n - 1.) / (n + 1.) / xi.last().unwrap().powi(2);

    let rho = rho.iter().map(|r| r / total_m).collect_vec();
    let m_coord = m_coord.iter().map(|r| r / total_m).collect_vec();
    let p = rho.iter().map(|r| r.powf(1. + 1. / n) * k).collect_vec();

    DiscreteModel {
        dimensionless: DimensionlessProperties {
            r_coord: x.into(),
            m_coord: m_coord.into(),
            rho: rho.into(),
            p: p.into(),
            v: v.into(),
            u: u.into(),
            gamma1: vec![gamma1; xi.len()].into(),
            a_star: a_star.into(),
            c1: c1.into(),
            rot: vec![0.; xi.len()].into(),
        },
        scale: None,
        metric: None,
    }
}

/// Index-zero (constant density sphere) analytical polytrope
pub struct Polytrope0 {
    /// First adiabatic exponent
    pub gamma1: f64,
}

impl Polytrope0 {
    const MAX_X: f64 =
        2.449489742783178098197284074705891391965947480656670128432692567250960377457315;

    /// Compute the exact solutions for this polytrope model. See [Pekeris 1938](https://ui.adsabs.harvard.edu/abs/1938ApJ....88..189P)
    pub fn exact(&self, ell: u64, radial_order: i64) -> f64 {
        let n = ell as f64;
        let k = 2. * radial_order as f64;

        let d = -2. + self.gamma1 / 4. * (k * (k + 5. + 2. * n) + 6. + 4. * n);
        let beta = d + (d * d + n * (n + 1.)).sqrt();

        beta.sqrt()
    }
}

impl ContinuousModel for Polytrope0 {
    fn inner(&self) -> f64 {
        0.
    }

    fn outer(&self) -> f64 {
        1.
    }

    fn eval(&self, grid: &[f64]) -> DiscreteModel {
        let xi = grid.iter().map(|g| g * Self::MAX_X).collect_vec();
        let theta = xi.iter().map(|x| 1. - x.powi(2) / 6.).collect_vec();
        let phi = xi.iter().map(|x| x.powi(3) / 3.).collect_vec();

        let mut v = xi
            .iter()
            .zip(theta.iter())
            .zip(phi.iter())
            .map(|((&xi, &theta), &phi)| phi / (xi * theta))
            .collect_vec();

        v[0] = 0.;

        let mut a_star = xi
            .iter()
            .zip(theta.iter())
            .zip(phi.iter())
            .map(|((&xi, &theta), &phi)| -phi / (xi * theta * self.gamma1))
            .collect_vec();

        a_star[0] = 0.;

        let phi_max = Self::MAX_X.powi(3) / 3.;

        let rho = vec![3. / (4. * PI); xi.len()];
        let m_coord = phi.iter().map(|r| r / phi_max).collect_vec();
        let p = vec![f64::INFINITY; xi.len()];

        DiscreteModel {
            dimensionless: DimensionlessProperties {
                r_coord: grid.to_owned().into_boxed_slice(),
                m_coord: m_coord.into(),
                rho: rho.into(),
                p: p.into(),
                v: v.into(),
                u: vec![3.; xi.len()].into(),
                gamma1: vec![self.gamma1; xi.len()].into(),
                a_star: a_star.into(),
                c1: vec![1.; xi.len()].into(),
                rot: vec![0.; xi.len()].into(),
            },
            scale: None,
            metric: None,
        }
    }

    fn dimensions(&self) -> Option<DimensionedProperties> {
        None
    }
}
