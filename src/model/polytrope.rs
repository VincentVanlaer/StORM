use std::{f64::consts::PI, vec};

use itertools::Itertools;

use crate::{
    bracket::{BracketOptimizer, InverseQuadratic, Point, Precision},
    model::{DiscreteModelSegment, Segment},
};

use super::{ContinuousModel, DimensionedProperties, DimensionlessProperties, DiscreteModel};

fn eval_lane_emden(xi: f64, phi: f64, theta: f64, alpha: f64, n: f64) -> (f64, f64) {
    if xi == 0. {
        (alpha * theta.powf(n) * xi.powi(2), 0.)
    } else {
        (alpha * theta.powf(n) * xi.powi(2), -phi / (xi * xi))
    }
}

fn rk4(xi: f64, phi: f64, theta: f64, alpha: f64, n: f64, step: f64) -> (f64, f64) {
    let hstep = step / 2.;

    let k1 = eval_lane_emden(xi, phi, theta, alpha, n);
    let k2 = eval_lane_emden(
        xi + hstep,
        phi + hstep * k1.0,
        theta + hstep * k1.1,
        alpha,
        n,
    );
    let k3 = eval_lane_emden(
        xi + hstep,
        phi + hstep * k2.0,
        theta + hstep * k2.1,
        alpha,
        n,
    );
    let k4 = eval_lane_emden(xi + step, phi + step * k3.0, theta + step * k3.1, alpha, n);

    (
        phi + step / 6. * (k1.0 + 2. * k2.0 + 2. * k3.0 + k4.0),
        theta + step / 6. * (k1.1 + 2. * k2.1 + 2. * k3.1 + k4.1),
    )
}

/// Description of a segmented polytrope
pub struct IndexSegments {
    indices: Vec<f64>,
    ratios: Vec<f64>,
    cuts: Vec<f64>,
}

#[derive(Debug, Clone, Copy)]
enum IterSegments {
    Integrate {
        x2: f64,
        ratio: f64,
        index1: f64,
        index2: f64,
    },
    End {
        index: f64,
    },
}

impl IndexSegments {
    /// Central segment of the segmented polytrope
    ///
    /// For a polytrope with only a single segment, this covers the entire polytrope
    pub fn central(index: f64) -> IndexSegments {
        assert!(index < 5.);
        assert!(index >= 0.);

        IndexSegments {
            indices: vec![index],
            ratios: vec![],
            cuts: vec![],
        }
    }

    /// Add a segment to the polytrope
    ///
    /// The density ratio is defined as the density after the jump divided by the density before
    /// the jump
    pub fn add(mut self, location: f64, index: f64, density_ratio: f64) -> Self {
        assert!(index < 5.);
        assert!(index >= 0.);

        self.indices.push(index);
        self.cuts.push(location);
        self.ratios.push(density_ratio);

        self
    }

    fn into_iter(&self) -> impl Iterator<Item = IterSegments> {
        (0..self.indices.len() - 1)
            .map(|i| IterSegments::Integrate {
                x2: self.cuts[i],
                ratio: self.ratios[i],
                index1: self.indices[i],
                index2: self.indices[i + 1],
            })
            .chain([IterSegments::End {
                index: *self.indices.last().unwrap(),
            }])
    }

    fn integrate_poly(&self, step_size: f64) -> Vec<PolytropeSegment> {
        let mut xi = 0.;
        let mut theta = 1.;
        let mut phi = 0.;
        let mut scale = 1.;

        let mut solution: Vec<_> = self
            .into_iter()
            .map(|segment| match segment {
                IterSegments::Integrate {
                    x2,
                    ratio,
                    index1,
                    index2,
                } => {
                    let mut thetas = Vec::new();
                    let mut phis = Vec::new();
                    let mut xis = Vec::new();
                    let mut nsteps = 0;

                    loop {
                        let xi = xi + (nsteps as f64) * step_size;

                        thetas.push(theta);
                        phis.push(phi);
                        xis.push(xi);

                        if x2 - xi <= step_size {
                            (phi, theta) = rk4(xi, phi, theta, scale, index1, x2 - xi);
                            break;
                        } else {
                            (phi, theta) = rk4(xi, phi, theta, scale, index1, step_size);
                        }

                        nsteps += 1;
                    }

                    thetas.push(theta);
                    phis.push(phi);
                    xis.push(x2);

                    let gamma2 =
                        ratio * ratio * (index1 + 1.) / (index2 + 1.) * theta.powf(index1 - 1.);

                    xi = x2;
                    phi *= gamma2 / ratio / theta.powf(index1);
                    theta = 1.;
                    let s = scale;
                    scale *= gamma2;

                    PolytropeSegment {
                        xi: xis,
                        theta: thetas,
                        phi: phis,
                        index: index1,
                        gamma2: s,
                        delta_max: 1.,
                    }
                }
                IterSegments::End { index } => {
                    let mut thetas = Vec::new();
                    let mut phis = Vec::new();
                    let mut xis = Vec::new();
                    let mut nsteps = 0;

                    while theta > 0. {
                        let xi = xi + (nsteps as f64) * step_size;

                        thetas.push(theta);
                        phis.push(phi);
                        xis.push(xi);

                        (phi, theta) = rk4(xi, phi, theta, scale, index, step_size);

                        nsteps += 1;
                    }

                    xi = xi + (nsteps as f64 - 1.) * step_size;

                    let bracket_result = (InverseQuadratic {})
                        .optimize(
                            Point {
                                x: 0.,
                                f: *thetas.last().unwrap(),
                            },
                            Point {
                                x: step_size,
                                f: theta,
                            },
                            |val| -> Result<f64, ()> {
                                Ok(rk4(
                                    xi,
                                    *phis.last().unwrap(),
                                    *thetas.last().unwrap(),
                                    scale,
                                    index,
                                    val,
                                )
                                .1)
                            },
                            Precision::ULP(1.try_into().unwrap()),
                            None,
                        )
                        .unwrap();

                    (phi, theta) = rk4(
                        xi,
                        *phis.last().unwrap(),
                        *thetas.last().unwrap(),
                        scale,
                        index,
                        bracket_result.lower.x,
                    );

                    thetas.push(theta);
                    phis.push(phi);
                    xis.push(bracket_result.lower.x + xi);

                    PolytropeSegment {
                        xi: xis,
                        theta: thetas,
                        phi: phis,
                        gamma2: scale,
                        index,
                        delta_max: 1.,
                    }
                }
            })
            .collect();

        solution
            .iter_mut()
            .zip(self.ratios.iter())
            .rev()
            .fold(1., |mut acc, (segment, ratio)| {
                acc /= segment.theta.last().unwrap().powf(segment.index) * ratio;
                segment.delta_max = acc;

                acc
            });

        solution
    }
}

#[derive(Debug, Clone)]
struct PolytropeSegment {
    pub xi: Vec<f64>,
    pub theta: Vec<f64>,
    pub phi: Vec<f64>,
    pub index: f64,
    pub gamma2: f64,
    pub delta_max: f64,
}

/// Create a polytrope model from the polytropic index, first adiabatic exponent and integration
/// step size.
///
/// The resulting model will be dimensionless
pub fn construct_polytrope(segments: IndexSegments, gamma1: f64, step_size: f64) -> DiscreteModel {
    let segments: Vec<_> = segments.integrate_poly(step_size);

    let max_xi = *segments.last().unwrap().xi.last().unwrap();
    let max_phi = *segments.last().unwrap().phi.last().unwrap();
    let max_gamma2 = segments.last().unwrap().gamma2;

    let segments = segments
        .into_iter()
        .map(|segment| {
            let PolytropeSegment {
                xi,
                theta,
                phi,
                index,
                gamma2,
                delta_max,
            } = segment;

            let (xi, theta, phi) = if *xi.last().unwrap() == max_xi {
                (
                    &xi[..xi.len() - 1],
                    &theta[..theta.len() - 1],
                    &phi[..phi.len() - 1],
                )
            } else {
                (xi.as_slice(), theta.as_slice(), phi.as_slice())
            };

            let mut u = xi
                .iter()
                .zip(theta.iter())
                .zip(phi.iter())
                .map(|((&xi, &theta), &phi)| xi.powi(3) * theta.powf(index) / phi * gamma2)
                .collect_vec();

            if xi[0] == 0. {
                u[0] = 3.;
            }

            let mut v = xi
                .iter()
                .zip(theta.iter())
                .zip(phi.iter())
                .map(|((&xi, &theta), &phi)| (index + 1.) * phi / (xi * theta))
                .collect_vec();

            if xi[0] == 0. {
                v[0] = 0.;
            }

            let m_coord = phi
                .iter()
                .map(|&phi| (delta_max * max_gamma2 / gamma2 / max_phi) * phi)
                .collect_vec();

            let mut c1 = xi
                .iter()
                .zip(m_coord.iter())
                .map(|(&xi, &m_coord)| xi.powi(3) / max_xi.powi(3) / m_coord)
                .collect_vec();

            if xi[0] == 0. {
                c1[0] = (3. * max_phi * gamma2) / (max_xi.powi(3) * delta_max * max_gamma2);
            }

            let mut a_star = xi
                .iter()
                .zip(theta.iter())
                .zip(phi.iter())
                .map(|((&xi, &theta), &phi)| -((index + 1.) / gamma1 - index) * phi / (xi * theta))
                .collect_vec();

            if xi[0] == 0. {
                a_star[0] = 0.;
            }

            let x = xi.iter().map(|xi| xi / max_xi).collect_vec();

            let rho = theta
                .iter()
                .map(|r| {
                    r.powf(index) * max_xi.powi(3) * delta_max * max_gamma2 / (max_phi * 4. * PI)
                })
                .collect_vec();

            let p = theta
                .iter()
                .map(|theta| {
                    max_xi.powi(4) * max_gamma2 * max_gamma2 * delta_max * delta_max
                        / (4. * PI * max_phi * max_phi * (index + 1.) * gamma2)
                        * theta.powf(index + 1.)
                })
                .collect_vec();

            DiscreteModelSegment {
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
                metric: None,
            }
        })
        .collect();

    DiscreteModel {
        segments,
        perturbed: None,
        scale: None,
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
    fn eval(&self, segment: usize, grid: &[f64]) -> DiscreteModelSegment {
        assert_eq!(segment, 0);

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

        DiscreteModelSegment {
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
            metric: None,
        }
    }

    fn dimensions(&self) -> Option<DimensionedProperties> {
        None
    }

    fn segments(&self) -> Vec<super::Segment> {
        vec![Segment {
            lower: 0.,
            upper: 1.,
        }]
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZero;

    use itertools::Itertools;

    use crate::{
        bracket::Precision,
        dynamic_interface::{DifferenceSchemes, ErasedSolver},
        model::interpolate::LinearInterpolator,
        system::adiabatic::Rotating1D,
    };

    use super::{IndexSegments, Polytrope0, construct_polytrope};

    #[test]
    fn test_segmented_construction() {
        let polytrope = construct_polytrope(
            IndexSegments::central(2.).add(0.5 * Polytrope0::MAX_X, 1., 0.5),
            5. / 3.,
            0.01,
        );

        let grid: Vec<_> = polytrope
            .segments
            .iter()
            .map(|x| &*x.dimensionless.r_coord)
            .collect();

        let solver = ErasedSolver::new(
            &LinearInterpolator::new(&polytrope),
            Rotating1D::new(0, 0),
            DifferenceSchemes::Colloc6,
            &grid,
        );

        let solutions = solver
            .scan_and_optimize(
                [1., 2., 3., 4., 5.],
                Precision::ULP(NonZero::new(1).unwrap()),
            )
            .into_iter()
            .map(|res| res.root)
            .collect_vec();

        assert_eq!(solutions, [1.445054727775158, 3.5183813886332884])
    }
}
