use super::{
    ContinuousModel, DimensionedProperties, DimensionlessProperties, DiscreteModel,
    DiscreteModelSegment, PerturbedMetric, Segment,
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
    fn eval(&self, segment: usize, grid: &[f64]) -> DiscreteModelSegment {
        let base_model = &self.model.segments[segment];

        let mut dimensionless = DimensionlessProperties::zeroed(grid.len());

        let mut metric = if base_model.metric.is_some() {
            Some(PerturbedMetric::zeroed(grid.len()))
        } else {
            None
        };

        interpolate(
            base_model.dimensionless.r_coord.iter().cloned(),
            grid.iter().cloned(),
            |lower_idx1, upper_idx1, idx2, pos| {
                macro_rules! linear {
                    ($s: expr, $d: expr, $($e: ident),+) => {
                        $($d.$e[idx2] = (1. - pos) * $s.$e[lower_idx1] + pos * $s.$e[upper_idx1];)*
                    };
                }

                linear!(
                    base_model.dimensionless,
                    dimensionless,
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

                if let Some(m2) = &mut metric
                    && let Some(m1) = &base_model.metric
                {
                    linear!(m1, m2, alpha, dalpha, ddalpha, beta, dbeta, ddbeta);
                }
            },
        );

        DiscreteModelSegment {
            dimensionless,
            metric,
        }
    }

    fn dimensions(&self) -> Option<DimensionedProperties> {
        self.model.scale
    }

    fn segments(&self) -> Vec<super::Segment> {
        self.model
            .segments
            .iter()
            .map(|x| Segment {
                lower: *x.dimensionless.r_coord.first().unwrap(),
                upper: *x.dimensionless.r_coord.last().unwrap(),
            })
            .collect()
    }
}

fn interpolate(
    old_grid_iter: impl Iterator<Item = f64>,
    new_grid_iter: impl Iterator<Item = f64>,
    mut interpolate: impl FnMut(usize, usize, usize, f64),
) {
    let mut old_grid_iter = old_grid_iter.enumerate();
    let mut new_grid_iter = new_grid_iter.enumerate();

    let mut prev = old_grid_iter.next().unwrap();
    let mut next = old_grid_iter.next().unwrap();
    let mut current_new_grid_point = new_grid_iter.next().unwrap();

    loop {
        if next.1 >= current_new_grid_point.1 {
            // Interpolate the point
            interpolate(
                prev.0,
                next.0,
                current_new_grid_point.0,
                (current_new_grid_point.1 - prev.1) / (next.1 - prev.1),
            );
            if let Some(c) = new_grid_iter.next() {
                current_new_grid_point = c;
            } else {
                break;
            };
        } else {
            prev = next;
            if let Some(n) = old_grid_iter.next() {
                next = n;
            } else {
                todo!("Extrapolation not supported");
            };
        }
    }
}
