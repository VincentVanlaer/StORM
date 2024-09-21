use super::{Boundary, Moments};
use crate::linalg::Matrix;
use crate::model::{StellarModel, GRAV};
use crate::stepper::StepMoments;
use color_eyre::Result;
use std::f64::consts::PI;

pub struct Rotating1D {
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

impl Rotating1D {
    pub fn from_model(value: &StellarModel, ell: u64, m: i64) -> Result<Self> {
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
            component.a[0][1] = -v_gamma[i];
            component.a[0][2] = 0.0;
            component.a[0][3] = 0.0;

            component.a[1][0] = -a_star[i];
            component.a[1][1] = a_star[i] - u[i] + 3. - ell;
            component.a[1][2] = 0.;
            component.a[1][3] = -1.;

            component.a[2][0] = 0.0;
            component.a[2][1] = 0.0;
            component.a[2][2] = 3. - u[i] - ell;
            component.a[2][3] = 1.;

            component.a[3][0] = u[i] * a_star[i];
            component.a[3][1] = u[i] * v_gamma[i];
            component.a[3][2] = ell * (ell + 1.);
            component.a[3][3] = -u[i] + 2. - ell;

            component.rot = value.rot[i];
            component.c1 = c1[i];
            component.x = x[i];
        }

        Ok(Rotating1D {
            components,
            ell,
            m: m as f64,
            u_upper: *u.last().unwrap(),
        })
    }
}

struct ModelPointsIterator<'model> {
    model: &'model Rotating1D,
    pos: usize,
    subpos: usize,
    total_subpos: usize,
    frequency: f64,
}

impl ModelPointsIterator<'_> {
    fn new(scale: u32, model: &Rotating1D, frequency: f64) -> ModelPointsIterator {
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

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.model.components.is_empty() || (self.pos + 1) == self.model.components.len() {
            return None;
        }
        let l = self.model.ell;
        let lambda = l * (l + 1.);
        let m = self.model.m;
        let omega = self.frequency;

        let add_frequency = |point: ModelPoint| {
            let mut a = point.a;
            let omega_rsq = if l == 0. && point.rot == 0. {
                1.0 // To prevent issues lower down with 0 / 0
            } else {
                (lambda * (omega - m * point.rot) + 2. * m * point.rot) * (omega - m * point.rot)
            };

            if l != 0. {
                let rel_rot =
                    2. * m * point.rot / (lambda * (omega - m * point.rot) + 2. * m * point.rot);

                a[0][0] += -lambda * rel_rot;
                a[0][1] += lambda.powi(2) / (omega_rsq * point.c1);
                a[0][2] += lambda.powi(2) / (omega_rsq * point.c1);

                a[1][1] += lambda * rel_rot;
                a[1][2] += lambda * rel_rot;
            }

            a[1][0] += point.c1
                * (omega - m * point.rot).powi(2)
                * (1. - (2. * m * point.rot).powi(2) / omega_rsq);

            a
        };

        let lower = self.model.components[self.pos];
        let lower_a = add_frequency(lower) * (1.0 / lower.x);

        let upper = self.model.components[self.pos + 1];
        let upper_a = add_frequency(upper) * (1.0 / upper.x);

        let intercept = lower_a
            + (upper_a - lower_a) * ((self.subpos as f64 + 0.5) / (self.total_subpos as f64));
        let slope = (upper_a - lower_a) * (1.0 / (self.total_subpos as f64));
        // This version of the code is not really faster, even though there
        // is quite the reduction in assembly
        //
        // let intercept = lower_a
        //     + (upper_a - lower_a)
        //         * ((self.subpos as f64 + 0.5) / (self.total_subpos as f64));
        // let slope = (upper_a - lower_a)
        //     * (1.0 / (self.total_subpos as f64));

        let delta = (upper.x - lower.x) / (self.total_subpos as f64);
        let sublower = lower.x + delta * (self.subpos as f64);
        let subupper = lower.x + delta * (self.subpos as f64 + 1.);

        self.subpos += 1;

        if self.subpos == self.total_subpos {
            self.pos += 1;
            self.subpos = 0;
        }

        Some((subupper - sublower, slope, intercept))
    }
}

impl ExactSizeIterator for ModelPointsIterator<'_> {
    fn len(&self) -> usize {
        (self.model.components.len() - 2) * self.total_subpos
    }
}

struct IterWrapper<'a, G> {
    iter: ModelPointsIterator<'a>,
    wrapped: G,
}

impl<
        const ORDER: usize,
        G: Fn((f64, Matrix<f64, 4, 4>, Matrix<f64, 4, 4>)) -> StepMoments<f64, 4, ORDER>,
    > Iterator for IterWrapper<'_, G>
{
    type Item = crate::stepper::StepMoments<f64, 4, ORDER>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            None => None,
            Some(r) => Some((self.wrapped)(r)),
        }
    }
}

impl<
        const ORDER: usize,
        G: Fn((f64, Matrix<f64, 4, 4>, Matrix<f64, 4, 4>)) -> StepMoments<f64, 4, ORDER>,
    > ExactSizeIterator for IterWrapper<'_, G>
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl Moments<f64, ModelGrid, 4, 1> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &ModelGrid,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = crate::stepper::StepMoments<f64, 4, 1>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(delta, _s, i)| StepMoments {
                delta,
                moments: [i],
            },
        }
    }
}
impl Moments<f64, ModelGrid, 4, 2> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &ModelGrid,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = crate::stepper::StepMoments<f64, 4, 2>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(delta, s, i)| StepMoments {
                delta,
                moments: [i, s],
            },
        }
    }
}
impl Moments<f64, ModelGrid, 4, 3> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &ModelGrid,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = crate::stepper::StepMoments<f64, 4, 3>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(delta, s, i)| StepMoments {
                delta,
                moments: [i, s, [[0.0; 4]; 4].into()],
            },
        }
    }
}
impl Moments<f64, ModelGrid, 4, 4> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &ModelGrid,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = crate::stepper::StepMoments<f64, 4, 4>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(delta, s, i)| StepMoments {
                delta,
                moments: [i, s, [0.0; 16].into(), [0.0; 16].into()],
            },
        }
    }
}

impl Boundary<f64, 4, 2> for Rotating1D {
    fn inner_boundary(&self, omega: f64) -> Matrix<f64, 4, 2> {
        let l = self.ell;
        let lambda = l * (l + 1.);
        let m = self.m;
        let rot = self.components[0].rot;
        let omega_rsq;
        let rel_rot;
        if l == 0. {
            omega_rsq = 1.;
            rel_rot = 1.;
        } else {
            omega_rsq = (lambda * (omega - m * rot) + 2. * m * rot) * (omega - m * rot);
            rel_rot = 2. * m * rot / (lambda * (omega - m * rot) + 2. * m * rot);
        }

        [
            [
                self.components[0].c1
                    * (omega - m * rot).powi(2)
                    * (1. - (2. * m * rot).powi(2) / omega_rsq),
                lambda * rel_rot - self.ell,
                lambda * rel_rot - self.ell,
                0.,
            ],
            [0., 0., self.ell, 1.],
        ]
        .into()
    }

    fn outer_boundary(&self, _frequency: f64) -> Matrix<f64, 4, 2> {
        [[1., -1., 0., 0.], [self.u_upper, 0., self.ell + 1., 1.]].into()
    }
}

pub struct ModelGrid {
    pub scale: u32,
}
