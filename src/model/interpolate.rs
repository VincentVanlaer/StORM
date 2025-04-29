use super::{DimensionlessProperties, Model};

pub(crate) trait InterpolatingModel {
    type ModelPoint;

    fn len(&self) -> usize;
    fn pos(&self, idx: usize) -> f64;

    fn eval(&self, idx: usize, pos: f64) -> Self::ModelPoint;
    fn eval_exact(&self, idx: usize) -> Self::ModelPoint;
}

pub struct LinearInterpolator<'model, M> {
    model: &'model M,
}

impl<'model, M: Model> LinearInterpolator<'model, M> {
    pub fn new(model: &'model M) -> Self {
        LinearInterpolator { model }
    }
}

impl<M: Model<ModelPoint = DimensionlessProperties>> InterpolatingModel
    for LinearInterpolator<'_, M>
{
    type ModelPoint = M::ModelPoint;

    fn len(&self) -> usize {
        self.model.len()
    }

    fn pos(&self, idx: usize) -> f64 {
        self.model.pos(idx)
    }

    fn eval(&self, idx: usize, pos: f64) -> Self::ModelPoint {
        let lower = self.model.eval(idx);
        let upper = self.model.eval(idx + 1);

        macro_rules! interp {
            ($e: ident) => {
                lower.$e + pos * (upper.$e - lower.$e)
            };
        }

        DimensionlessProperties {
            v_gamma: interp!(v_gamma),
            a_star: interp!(a_star),
            u: interp!(u),
            c1: interp!(c1),
            rot: interp!(rot),
        }
    }

    fn eval_exact(&self, idx: usize) -> Self::ModelPoint {
        self.model.eval(idx)
    }
}
