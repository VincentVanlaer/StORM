use super::{Boundary, Moments};
use crate::linalg::Matrix;
use crate::model::{StellarModel, GRAV};
use crate::stepper::StepMoments;
use color_eyre::Result;
use std::f64::consts::PI;

pub(crate) struct NonRotating1D {
    components: Vec<ModelPoint>,
    ell: f64,
    m: f64,
    u_upper: f64,
}

#[derive(Clone, Copy, Debug)]
struct ModelPoint {
    a: Matrix<f64, 4, 4>,
    c1: f64,
    rot: f64,
    x: f64,
}

impl NonRotating1D {
    pub(crate) fn from_model(value: &StellarModel, ell: u64, m: i64) -> Result<Self> {
        let ell = ell as f64;
        let mut components: Vec<_> = vec![
            ModelPoint {
                a: [0.0f64; 16].into(),
                c1: 0.0,
                x: 0.0,
                rot: 0.0,
            };
            value.r_coord.len()
        ];

        let r_cubed = value.r_coord.mapv(|a| a.powi(3));
        let mut v_gamma =
            GRAV * &value.m_coord * &value.rho / (&value.p * &value.r_coord * &value.gamma1);
        let mut a_star = &r_cubed / (GRAV * &value.m_coord) * &value.nsqrd;
        let mut u = 4.0 * PI * &value.rho * &r_cubed / &value.m_coord;
        let mut c1 = &r_cubed / value.radius.powi(3) * value.mass / &value.m_coord;
        let x = &value.r_coord / value.radius;

        v_gamma[0] = 0.0;
        a_star[0] = 0.0;
        u[0] = 3.0;
        c1[0] = value.mass / value.radius.powi(3) * 3.0 / (4.0 * PI * value.rho[0]);

        for (i, component) in components.iter_mut().enumerate() {
            component.a[0][0] = v_gamma[i] - 1.0 - ell;
            component.a[1][0] = -v_gamma[i];
            component.a[2][0] = 0.0;
            component.a[3][0] = 0.0;

            component.a[0][1] = -a_star[i];
            component.a[1][1] = a_star[i] - u[i] + 3. - ell;
            component.a[2][1] = 0.;
            component.a[3][1] = -1.;

            component.a[0][2] = 0.0;
            component.a[1][2] = 0.0;
            component.a[2][2] = 3. - u[i] - ell;
            component.a[3][2] = 1.;

            component.a[0][3] = u[i] * a_star[i];
            component.a[1][3] = u[i] * v_gamma[i];
            component.a[2][3] = ell * (ell + 1.);
            component.a[3][3] = -u[i] + 2. - ell;

            component.rot = value.rot[i];
            component.c1 = c1[i];
            component.x = x[i];
        }

        Ok(NonRotating1D {
            components,
            ell,
            m: m as f64,
            u_upper: *u.last().unwrap(),
        })
    }
}

struct ModelPointsIterator<'model> {
    model: &'model NonRotating1D,
    pos: usize,
    subpos: usize,
    total_subpos: usize,
    frequency: f64,
}

impl ModelPointsIterator<'_> {
    fn new(scale: u32, model: &NonRotating1D, frequency: f64) -> ModelPointsIterator {
        if scale == 0 {
            ModelPointsIterator {
                model,
                pos: 1,
                subpos: 0,
                total_subpos: 1,
                frequency,
            }
        } else {
            ModelPointsIterator {
                model,
                pos: 1,
                subpos: 0,
                total_subpos: 2usize.pow(scale),
                frequency,
            }
        }
    }
}

impl Iterator for ModelPointsIterator<'_> {
    type Item = (f64, Matrix<f64, 4, 4>, Matrix<f64, 4, 4>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.model.components.len() == 0 || (self.pos + 1) == self.model.components.len() {
            return None;
        }
        let l = self.model.ell;
        let lambda = l * (l + 1.);
        let m = self.model.m;
        let omega = self.frequency;

        let add_frequency = |point: ModelPoint| {
            let mut a = point.a.clone();
            let omega_rsq =
                (lambda * (omega - m * point.rot) + 2. * m * point.rot) * (omega - m * point.rot);
            let rel_rot =
                2. * m * point.rot / (lambda * (omega - m * point.rot) + 2. * m * point.rot);

            a[0][0] += -lambda * rel_rot;
            a[1][0] += lambda.powi(2) / (omega_rsq * point.c1);
            a[2][0] += lambda.powi(2) / (omega_rsq * point.c1);

            a[0][1] += point.c1
                * (omega - m * point.rot).powi(2)
                * (1. - (2. * m * point.rot).powi(2) / omega_rsq);

            a[1][1] += lambda * rel_rot;
            a[2][1] += lambda * rel_rot;

            a
        };

        let lower = self.model.components[self.pos];
        let lower_a = add_frequency(lower);

        let upper = self.model.components[self.pos + 1];
        let upper_a = add_frequency(upper);

        let intercept = lower_a
            + (upper_a - lower_a) * ((self.subpos as f64 + 0.5) / (self.total_subpos as f64));
        let slope = (upper_a - lower_a) * (1.0 / (self.total_subpos as f64));

        let delta = (upper.x - lower.x) / (self.total_subpos as f64);
        let sublower = lower.x + delta * (self.subpos as f64);
        let subupper = lower.x + delta * (self.subpos as f64 + 1.);

        self.subpos += 1;

        if self.subpos == self.total_subpos {
            self.pos += 1;
            self.subpos = 0;
        }

        Some((subupper.ln() - sublower.ln(), slope, intercept))
    }
}

impl ExactSizeIterator for ModelPointsIterator<'_> {
    fn len(&self) -> usize {
        (self.model.components.len() - 2) * self.total_subpos
    }
}

impl Moments<f64, ModelGrid, 4, 1> for NonRotating1D {
    fn evaluate_moments(
        &self,
        grid: &ModelGrid,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = crate::stepper::StepMoments<f64, 4, 1>> {
        ModelPointsIterator::new(grid.scale, &self, frequency).map(move |(delta, _s, i)| {
            StepMoments {
                moments: [i],
                delta,
            }
        })
    }
}
impl Moments<f64, ModelGrid, 4, 2> for NonRotating1D {
    fn evaluate_moments(
        &self,
        grid: &ModelGrid,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = crate::stepper::StepMoments<f64, 4, 2>> {
        ModelPointsIterator::new(grid.scale, &self, frequency).map(move |(delta, s, i)| {
            StepMoments {
                moments: [i, s],
                delta,
            }
        })
    }
}
impl Moments<f64, ModelGrid, 4, 3> for NonRotating1D {
    fn evaluate_moments(
        &self,
        grid: &ModelGrid,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = crate::stepper::StepMoments<f64, 4, 3>> {
        ModelPointsIterator::new(grid.scale, &self, frequency).map(move |(delta, s, i)| {
            StepMoments {
                moments: [i, s, [0.0; 16].into()],
                delta,
            }
        })
    }
}
impl Moments<f64, ModelGrid, 4, 4> for NonRotating1D {
    fn evaluate_moments(
        &self,
        grid: &ModelGrid,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = crate::stepper::StepMoments<f64, 4, 4>> {
        ModelPointsIterator::new(grid.scale, &self, frequency).map(move |(delta, s, i)| {
            StepMoments {
                moments: [i, s, [0.0; 16].into(), [0.0; 16].into()],
                delta,
            }
        })
    }
}

impl Boundary<f64, 4, 2, 2> for NonRotating1D {
    fn inner_boundary(&self, frequency: f64) -> Matrix<f64, 2, 4> {
        [
            [self.components[0].c1 * frequency * frequency, 0.],
            [-self.ell, 0.],
            [-self.ell, self.ell],
            [0., -1.],
        ]
        .into()
    }

    fn outer_boundary(&self, _frequency: f64) -> Matrix<f64, 2, 4> {
        [
            [1., self.u_upper],
            [-1., 0.],
            [0., self.ell + 1.0],
            [0., 1.],
        ]
        .into()
    }
}

pub(crate) struct ModelGrid {
    pub scale: u32,
}
