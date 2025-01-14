use nalgebra::Const;
use nalgebra::OMatrix;

use super::{Boundary, GridLength, Moments};
use crate::linalg::OwnedArray;
use crate::model::StellarModel;
use crate::stepper::StepMoments;
use std::f64::consts::PI;

/// Spherically symmetric stellar oscillation equations with a full rotation term
///
/// Note that this system of equations only contains a single spherical harmonic, and will
/// therefore not be completely valid in a rotating star.
pub struct Rotating1D {
    components: Vec<ModelPoint>,
    ell: f64,
    m: f64,
    u_upper: f64,
}

type FourMatrix = OMatrix<f64, Const<4>, Const<4>>;

#[derive(Clone, Copy, Debug)]
struct ModelPoint {
    a: FourMatrix,
    c1: f64,
    rot: f64,
    x: f64,
}

impl Rotating1D {
    /// Construct this system of equations from a stellar model and selection of the relevant
    /// spherical harmonic.
    pub fn from_model(value: &StellarModel, ell: u64, m: i64) -> Rotating1D {
        let ell = ell as f64;
        let mut components: Vec<_> = vec![
            ModelPoint {
                a: [[0.0f64; 4]; 4].into(),
                c1: 0.0,
                x: 0.0,
                rot: 0.0,
            };
            value.r_coord.len()
        ];

        let r_cubed = value.r_coord.mapv(|a| a.powi(3));
        let mut v_gamma =
            value.grav * &value.m_coord * &value.rho / (&value.p * &value.r_coord * &value.gamma1);
        let mut a_star = &r_cubed / (value.grav * &value.m_coord) * &value.nsqrd;
        let mut u = 4.0 * PI * &value.rho * &r_cubed / &value.m_coord;
        let mut c1 = &r_cubed / value.radius.powi(3) * value.mass / &value.m_coord;
        let x = &value.r_coord / value.radius;

        v_gamma[0] = 0.0;
        a_star[0] = 0.0;
        u[0] = 3.0;
        c1[0] = value.mass / value.radius.powi(3) * 3.0 / (4.0 * PI * value.rho[0]);

        for (i, component) in components.iter_mut().enumerate() {
            *component.a.index_mut((0, 0)) = v_gamma[i] - 1.0 - ell;
            *component.a.index_mut((0, 1)) = -v_gamma[i];
            *component.a.index_mut((0, 2)) = 0.0;
            *component.a.index_mut((0, 3)) = 0.0;

            *component.a.index_mut((1, 0)) = -a_star[i];
            *component.a.index_mut((1, 1)) = a_star[i] - u[i] + 3. - ell;
            *component.a.index_mut((1, 2)) = 0.;
            *component.a.index_mut((1, 3)) = -1.;

            *component.a.index_mut((2, 0)) = 0.0;
            *component.a.index_mut((2, 1)) = 0.0;
            *component.a.index_mut((2, 2)) = 3. - u[i] - ell;
            *component.a.index_mut((2, 3)) = 1.;

            *component.a.index_mut((3, 0)) = u[i] * a_star[i];
            *component.a.index_mut((3, 1)) = u[i] * v_gamma[i];
            *component.a.index_mut((3, 2)) = ell * (ell + 1.);
            *component.a.index_mut((3, 3)) = -u[i] + 2. - ell;

            component.rot = value.rot[i];
            component.c1 = c1[i];
            component.x = x[i];
        }

        Rotating1D {
            components,
            ell,
            m: m as f64,
            u_upper: *u.last().unwrap(),
        }
    }
}

struct ModelPointsIterator<'model> {
    model: &'model Rotating1D,
    pos: usize,
    skip: usize,
    subpos: usize,
    total_subpos: usize,
    frequency: f64,
}

impl ModelPointsIterator<'_> {
    fn new(scale: i32, model: &Rotating1D, frequency: f64) -> ModelPointsIterator {
        if scale >= 0 {
            ModelPointsIterator {
                model,
                pos: 1,
                subpos: 0,
                skip: 1,
                total_subpos: 2_usize.pow(scale.unsigned_abs()),
                frequency,
            }
        } else {
            ModelPointsIterator {
                model,
                pos: 1,
                subpos: 0,
                skip: 2_usize.pow(scale.unsigned_abs()),
                total_subpos: 1,
                frequency,
            }
        }
    }
}

impl Iterator for ModelPointsIterator<'_> {
    type Item = (FourMatrix, FourMatrix);

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.model.components.is_empty() || (self.pos + self.skip) >= self.model.components.len()
        {
            return None;
        }
        let l = self.model.ell;
        let lambda = l * (l + 1.);
        let m = self.model.m;
        let omega = self.frequency;

        let add_frequency = |point: ModelPoint| {
            let mut a = point.a;
            let omega_rsq = if l == 0. {
                1.0 // To prevent issues lower down with 0 / 0
            } else {
                (lambda * (omega - m * point.rot) + 2. * m * point.rot) * (omega - m * point.rot)
            };

            if l != 0. {
                let rel_rot =
                    2. * m * point.rot / (lambda * (omega - m * point.rot) + 2. * m * point.rot);

                *a.index_mut((0, 0)) += -lambda * rel_rot;
                *a.index_mut((0, 1)) += lambda.powi(2) / (omega_rsq * point.c1);
                *a.index_mut((0, 2)) += lambda.powi(2) / (omega_rsq * point.c1);

                *a.index_mut((1, 1)) += lambda * rel_rot;
                *a.index_mut((1, 2)) += lambda * rel_rot;
            }

            *a.index_mut((1, 0)) += point.c1
                * (omega - m * point.rot).powi(2)
                * (1. - (2. * m * point.rot).powi(2) / omega_rsq);

            a
        };

        // Bounds check due to potential overflow
        let lower = self.model.components[self.pos];
        let upper = self.model.components[self.pos + self.skip];

        let delta = (upper.x - lower.x) / (self.total_subpos as f64);
        let sublower = lower.x + delta * (self.subpos as f64);
        let subupper = lower.x + delta * (self.subpos as f64 + 1.);
        // let sublower = sublower.ln();
        // let subupper = subupper.ln();

        let delta = subupper - sublower;

        let lower_a = add_frequency(lower) * (delta / lower.x);
        // let lower_a = add_frequency(lower);

        let upper_a = add_frequency(upper) * (delta / upper.x);
        // let upper_a = add_frequency(upper);

        let intercept = lower_a
            + (upper_a - lower_a) * ((self.subpos as f64 + 0.5) / (self.total_subpos as f64));
        let slope = (upper_a - lower_a) * (1. / (self.total_subpos as f64));

        self.subpos += 1;

        if self.subpos == self.total_subpos {
            self.pos += self.skip;
            self.subpos = 0;
        }

        Some((slope, intercept))
    }
}

impl ExactSizeIterator for ModelPointsIterator<'_> {
    fn len(&self) -> usize {
        if self.skip == 1 {
            (self.model.components.len() - 2) * self.total_subpos
        } else {
            (self.model.components.len() - 2) / self.skip
        }
    }
}

struct IterWrapper<'a, G> {
    iter: ModelPointsIterator<'a>,
    wrapped: G,
}

type FourStepMoments<const ORDER: usize> =
    StepMoments<f64, Const<4>, Const<ORDER>, OwnedArray<f64, Const<4>, Const<4>, Const<ORDER>>>;

impl<const ORDER: usize, G: Fn((FourMatrix, FourMatrix)) -> FourStepMoments<ORDER>> Iterator
    for IterWrapper<'_, G>
{
    type Item = FourStepMoments<ORDER>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        match self.iter.next() {
            None => None,
            Some(r) => Some((self.wrapped)(r)),
        }
    }
}

impl<const ORDER: usize, G: Fn((FourMatrix, FourMatrix)) -> FourStepMoments<ORDER>>
    ExactSizeIterator for IterWrapper<'_, G>
{
    fn len(&self) -> usize {
        self.iter.len()
    }
}

impl GridLength<GridScale> for Rotating1D {
    fn len(&self, grid: &GridScale) -> usize {
        if grid.scale >= 0 {
            (self.components.len() - 2) * 2usize.pow(grid.scale.unsigned_abs())
        } else {
            (self.components.len() - 2) / 2usize.pow(grid.scale.unsigned_abs())
        }
    }
}

impl Moments<f64, GridScale, Const<4>, Const<1>> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &GridScale,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = FourStepMoments<1>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(_s, i)| StepMoments {
                moments: [i].into(),
            },
        }
    }

    fn shape(&self) -> Const<4> {
        Const
    }
}
impl Moments<f64, GridScale, Const<4>, Const<2>> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &GridScale,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = FourStepMoments<2>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(s, i)| StepMoments {
                moments: [i, s].into(),
            },
        }
    }

    fn shape(&self) -> Const<4> {
        Const
    }
}
impl Moments<f64, GridScale, Const<4>, Const<3>> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &GridScale,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = FourStepMoments<3>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(s, i)| StepMoments {
                moments: [i, s, [[0.0; 4]; 4].into()].into(),
            },
        }
    }

    fn shape(&self) -> Const<4> {
        Const
    }
}
impl Moments<f64, GridScale, Const<4>, Const<4>> for Rotating1D {
    fn evaluate_moments(
        &self,
        grid: &GridScale,
        frequency: f64,
    ) -> impl ExactSizeIterator<Item = FourStepMoments<4>> {
        IterWrapper {
            iter: ModelPointsIterator::new(grid.scale, self, frequency),
            wrapped: |(s, i)| StepMoments {
                moments: [i, s, [[0.0; 4]; 4].into(), [[0.0; 4]; 4].into()].into(),
            },
        }
    }

    fn shape(&self) -> Const<4> {
        Const
    }
}

impl Boundary<f64, Const<4>, Const<2>> for Rotating1D {
    fn inner_boundary(&self, omega: f64) -> OMatrix<f64, Const<2>, Const<4>> {
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
                0.,
            ],
            [lambda * rel_rot - self.ell, 0.],
            [lambda * rel_rot - self.ell, self.ell],
            [0., -1.],
        ]
        .into()
    }

    fn outer_boundary(&self, _frequency: f64) -> OMatrix<f64, Const<2>, Const<4>> {
        [[1., self.u_upper], [-1., 0.], [0., self.ell + 1.], [0., 1.]].into()
    }
}

/// Grid scaling for [Rotating1D]
///
/// This is a rough method of scaling the grid, as every interval will be divided a certain number
/// od times, without taking into account where refinement is actually necessary.
pub struct GridScale {
    /// How much times the grid is divided into two sub intervals
    ///
    /// If `scale` equals one, the grid is divide in two, if `scale` equals two, the grid is
    /// divided into four, ...
    pub scale: i32,
}
