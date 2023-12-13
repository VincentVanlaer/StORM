use std::f64::consts::PI;

use crate::{linalg::Matrix, model::StellarModel};
use color_eyre::Result;
use ndarray::Array1;
use ndarray_interp::interp1d::{CubicSpline, Interp1DBuilder};
use num::Float;

pub(crate) trait Interpolator<T: Float, const N: usize>
where
    [(); N * N]: Sized,
{
    fn evaluate(&self, location: T, frequency: T) -> Matrix<T, N, N>;
}

pub(crate) trait System<T: Float, const N: usize, const N_INNER: usize, const N_OUTER: usize>:
    Interpolator<T, N>
where
    [(); N * N]: Sized,
    [(); N_INNER * N]: Sized,
    [(); N_OUTER * N]: Sized,
{
    fn inner_boundary(&self, frequency: f64) -> Matrix<T, N_INNER, N>;
    fn outer_boundary(&self, _frequency: f64) -> Matrix<T, N_OUTER, N>;
}

pub(crate) struct NonRotating1D {
    grid_power: u64,
    components: Vec<Matrix<f64, 4, 4>>,
    c1: Vec<f64>,
    ell: f64,
    u_upper: f64,
}

const MAX_GRID_POWER: u64 = 1_000_000;
const GRAV: f64 = 6.67430e-8;

impl NonRotating1D {
    pub(crate) fn from_model(value: &StellarModel, ell: u64) -> Result<Self> {
        let ell = ell as f64;
        let smallest_difference = value
            .r_coord
            .windows(2)
            .into_iter()
            .map(|x| x[1] - x[0])
            .min_by(|a, b| a.total_cmp(b))
            .expect("Model has more than one element");
        let grid_power = ((1.0 / smallest_difference).ceil() as u64)
            .next_power_of_two()
            .max(MAX_GRID_POWER);

        let mut components: Vec<Matrix<f64, 4, 4>> =
            vec![
                [0.0f64; 16].into();
                <u64 as TryInto<usize>>::try_into(grid_power).unwrap() + 1_usize
            ];

        let r_cubed = value.r_coord.mapv(|a| a.powi(3));
        let mut v = GRAV * &value.m_coord * &value.rho / (&value.p * &value.r_coord);
        let mut a_star = &r_cubed / (GRAV * &value.m_coord) * &value.nsqrd;
        let mut u = 4.0 * PI * &value.rho * &r_cubed / &value.m_coord;
        let mut c1 = &r_cubed / value.radius.powi(3) * value.mass / &value.m_coord;
        let x = &value.r_coord / value.radius;

        v[0] = 0.0;
        a_star[0] = 0.0;
        u[0] = 3.0;
        c1[0] = value.mass / value.radius.powi(3) * 3.0 / (4.0 * PI * value.rho[0]);

        let v_gamma_interp = Interp1DBuilder::new(&v / &value.gamma1)
            .x(x.clone())
            .strategy(CubicSpline::new())
            .build()?
            .interp_array(&Array1::linspace(0., 1., components.len()))?;

        let u_interp = Interp1DBuilder::new(u)
            .x(x.clone())
            .strategy(CubicSpline::new())
            .build()?
            .interp_array(&Array1::linspace(0., 1., components.len()))?;

        let a_star_interp = Interp1DBuilder::new(a_star)
            .x(x.clone())
            .strategy(CubicSpline::new())
            .build()?
            .interp_array(&Array1::linspace(0., 1., components.len()))?;

        for (i, component) in components.iter_mut().enumerate() {
            component[0][0] = v_gamma_interp[i] - 1.0 - ell;
            component[1][0] = -v_gamma_interp[i];
            component[2][0] = 0.0;
            component[3][0] = 0.0;

            component[0][1] = -a_star_interp[i];
            component[1][1] = a_star_interp[i] - u_interp[i] + 3. - ell;
            component[2][1] = 0.;
            component[3][1] = -1.;

            component[0][2] = 0.0;
            component[1][2] = 0.0;
            component[2][2] = 3. - u_interp[i] - ell;
            component[3][2] = 1.;

            component[0][3] = u_interp[i] * a_star_interp[i];
            component[1][3] = u_interp[i] * v_gamma_interp[i];
            component[2][3] = ell * (ell + 1.);
            component[3][3] = -u_interp[i] + 2. - ell;
        }

        let c1_interp = Interp1DBuilder::new(c1)
            .x(x.clone())
            .strategy(CubicSpline::new())
            .build()?
            .interp_array(&Array1::linspace(0., 1., components.len()))?;

        Ok(NonRotating1D {
            grid_power,
            components,
            c1: c1_interp.into_raw_vec(),
            ell,
            u_upper: *u_interp.last().unwrap(),
        })
    }
}

impl Interpolator<f64, 4> for NonRotating1D {
    fn evaluate(&self, location: f64, frequency: f64) -> Matrix<f64, 4, 4> {
        let index = (location * self.grid_power as f64).floor() as usize;
        let fraction = (location * self.grid_power as f64) - index as f64;

        let first_point = self.components[index];
        let second_point = self.components[index + 1];
        let omega_sqrd = frequency * frequency;

        let mut out =
            first_point * ((1.0 - fraction) / location) + second_point * (fraction / location);

        out[1][0] += ((1. - fraction) / self.c1[index] + fraction / self.c1[index + 1])
            / omega_sqrd
            * self.ell
            * (self.ell + 1.)
            / location;
        out[2][0] += ((1. - fraction) / self.c1[index] + fraction / self.c1[index + 1])
            / omega_sqrd
            * self.ell
            * (self.ell + 1.)
            / location;
        out[0][1] += ((1. - fraction) * self.c1[index] + fraction * self.c1[index + 1])
            * omega_sqrd
            / location;

        out
    }
}

impl System<f64, 4, 2, 2> for NonRotating1D {
    fn inner_boundary(&self, frequency: f64) -> Matrix<f64, 2, 4> {
        [
            self.c1[0] * frequency * frequency,
            0.,
            -self.ell,
            0.,
            0.,
            self.ell,
            0.,
            -1.,
        ]
        .into()
    }

    fn outer_boundary(&self, _frequency: f64) -> Matrix<f64, 2, 4> {
        [-1., self.u_upper, -1., 0., 1., self.ell + 1.0, 0., 1.].into()
    }
}

pub(crate) struct StretchedString {
    pub speed_generator: fn(f64) -> f64,
}

pub(crate) fn constant_speed(_location: f64) -> f64 {
    1.0
}

pub(crate) fn parabola(location: f64) -> f64 {
    2.0 * (location - 0.5).powi(2) + 0.5
}

pub(crate) fn linear_piecewise(location: f64) -> f64 {
    0.1 * (location - 0.5).abs() + 0.5
}

pub(crate) fn smoothened_linear_piecewise(location: f64) -> f64 {
    let offset = location - 0.5;
    0.5 + 0.1 * (offset.powi(2) + 0.00001).sqrt()
}

impl Interpolator<f64, 2> for StretchedString {
    fn evaluate(&self, location: f64, frequency: f64) -> Matrix<f64, 2, 2> {
        [[0.0, -frequency.powi(2) / (self.speed_generator)(location).powi(2)], [1.0, 0.0]].into()
    }
}

impl System<f64, 2, 1, 1> for StretchedString {
    fn inner_boundary(&self, _frequency: f64) -> Matrix<f64, 1, 2> {
        [[1.0], [0.0]].into()
    }

    fn outer_boundary(&self, _frequency: f64) -> Matrix<f64, 1, 2> {
        [[1.0], [0.0]].into()
    }
}
