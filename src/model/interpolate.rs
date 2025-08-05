use super::{
    ContinuousModel, DimensionedProperties, DimensionlessProperties, DiscreteModel, PerturbedMetric,
};

/// Linear interpolator for a [DiscreteModel]
pub struct LinearInterpolator<'model> {
    model: &'model DiscreteModel,
}

impl<'model> LinearInterpolator<'model> {
    /// Construct a new interpolator from a model
    pub fn new(model: &'model DiscreteModel) -> Self {
        LinearInterpolator { model }
    }
}

impl ContinuousModel for LinearInterpolator<'_> {
    fn eval(&self, grid: &[f64]) -> DiscreteModel {
        let mut old_grid_iter = self.model.dimensionless.r_coord.iter().enumerate();
        let mut new_grid_iter = grid.iter().enumerate();

        let mut dimensionless = DimensionlessProperties {
            r_coord: vec![0.; grid.len()].into(),
            m_coord: vec![0.; grid.len()].into(),
            rho: vec![0.; grid.len()].into(),
            p: vec![0.; grid.len()].into(),
            v: vec![0.; grid.len()].into(),
            u: vec![0.; grid.len()].into(),
            gamma1: vec![0.; grid.len()].into(),
            a_star: vec![0.; grid.len()].into(),
            c1: vec![0.; grid.len()].into(),
            rot: vec![0.; grid.len()].into(),
        };

        let mut metric = if let Some(m) = &self.model.metric {
            Some(PerturbedMetric {
                beta: vec![0.; grid.len()].into(),
                dbeta: vec![0.; grid.len()].into(),
                ddbeta: vec![0.; grid.len()].into(),
                rot: m.rot,
            })
        } else {
            None
        };

        let mut prev = (0, &self.model.dimensionless.r_coord[0]);
        let mut next = (0, &self.model.dimensionless.r_coord[0]);
        let mut current_new_grid_point = new_grid_iter.next().unwrap();

        loop {
            if next.1 == current_new_grid_point.1 {
                macro_rules! cp {
                    ($s: expr, $d: expr, $($e: ident),+) => {
                        $($s.$e[current_new_grid_point.0] = $d.$e[next.0];)*
                    };
                }

                cp!(
                    dimensionless,
                    self.model.dimensionless,
                    r_coord,
                    m_coord,
                    rho,
                    p,
                    v,
                    u,
                    gamma1,
                    a_star,
                    c1,
                    rot
                );

                if let Some(l) = &mut metric
                    && let Some(r) = &self.model.metric
                {
                    cp!(l, r, beta, dbeta, ddbeta);
                }

                if let Some(c) = new_grid_iter.next() {
                    current_new_grid_point = c;
                    continue;
                } else {
                    break;
                };
            } else if next.1 > current_new_grid_point.1 {
                macro_rules! interp {
                    ($($e: ident),+) => {
                        let m = &self.model.dimensionless;
                        let pos = (current_new_grid_point.1 - prev.1) / (next.1 - prev.1);
                        $(dimensionless.$e[current_new_grid_point.0] =
                            m.$e[prev.0] +
                            pos * (m.$e[next.0] - m.$e[prev.0]);)*
                    };
                }

                dimensionless.r_coord[current_new_grid_point.0] = *current_new_grid_point.1;
                interp!(m_coord, rho, p, v, u, gamma1, a_star, c1, rot);

                if let Some(c) = new_grid_iter.next() {
                    current_new_grid_point = c;
                    continue;
                } else {
                    break;
                };
            }

            prev = next;
            if let Some(n) = old_grid_iter.next() {
                next = n;
            } else {
                break;
            };
        }

        DiscreteModel {
            dimensionless,
            scale: self.model.scale,
            metric,
        }
    }

    fn inner(&self) -> f64 {
        self.model.dimensionless.r_coord[0]
    }

    fn outer(&self) -> f64 {
        *self.model.dimensionless.r_coord.last().unwrap()
    }

    fn dimensions(&self) -> Option<DimensionedProperties> {
        self.model.scale
    }
}
