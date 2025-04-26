use itertools::Itertools;

use super::{DimensionlessProperties, Model};

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

pub struct Polytrope {
    v_gamma: Box<[f64]>,
    u: Box<[f64]>,
    c1: Box<[f64]>,
    a_star: Box<[f64]>,
    x: Box<[f64]>,
}

impl Polytrope {
    pub fn new(n: f64, gamma1: f64, step: f64) -> Self {
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

        let mut v_gamma = xi
            .iter()
            .zip(theta.iter())
            .zip(phi.iter())
            .map(|((&xi, &theta), &phi)| (n + 1.) * phi / (xi * theta * gamma1))
            .collect_vec();

        v_gamma[0] = 0.;

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

        Self {
            v_gamma: v_gamma.into(),
            u: u.into(),
            c1: c1.into(),
            a_star: a_star.into(),
            x: x.into(),
        }
    }
}

impl Model for Polytrope {
    type ModelPoint = DimensionlessProperties;

    fn len(&self) -> usize {
        self.x.len()
    }

    fn pos(&self, idx: usize) -> f64 {
        self.x[idx]
    }

    fn eval(&self, idx: usize) -> Self::ModelPoint {
        DimensionlessProperties {
            v_gamma: self.v_gamma[idx],
            a_star: self.a_star[idx],
            u: self.u[idx],
            c1: self.c1[idx],
            rot: 0.,
        }
    }
}
