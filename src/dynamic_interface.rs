//! Main interface for computing the determinant for a trial frequency

use nalgebra::{Const, DefaultAllocator, Dim, Dyn};

use crate::bracket::{
    BracketOptimizer as _, BracketResult, FilterSignSwap, InverseQuadratic, Point, Precision,
};
use crate::linalg::storage::ArrayAllocator;
use crate::model::interpolate::LinearInterpolator;
use crate::model::{DimensionlessProperties, Model};
use crate::solver::{
    self, DeterminantAllocs, UpperResult, determinant, determinant_explicit, determinant_with_upper,
};
use crate::stepper::{
    Colloc2, Colloc4, ExplicitStepper, ImplicitStepper, Magnus2, Magnus4, Magnus6, Magnus8,
};
use crate::system::adiabatic::Rotating1D;
use crate::system::discretized::{DiscretizedSystem, DiscretizedSystemImpl};

/// Supported difference schemes
#[derive(clap::ValueEnum, Clone, Copy, Debug)]
pub enum DifferenceSchemes {
    /// Second-order collocation method
    Colloc2,
    /// Fourth-order collocation method
    Colloc4,
    /// Second-order magnus method
    Magnus2,
    /// Fourth-order magnus method
    Magnus4,
    /// Sixth-order magnus method
    Magnus6,
    /// Eight-order magnus method
    Magnus8,
}

/// Type erased interface for computing the determinant and the eigenvector of a problem
pub struct ErasedSolver {
    det: Box<dyn Fn(f64) -> f64>,
    eigenvector: Box<dyn Fn(f64) -> (f64, Vec<f64>)>,
}

impl ErasedSolver {
    /// Construct from a system, difference scheme and grid definition
    pub fn new(
        model: &impl Model<ModelPoint = DimensionlessProperties>,
        system: Rotating1D,
        scheme: DifferenceSchemes,
        solver_grid: Option<&[f64]>,
    ) -> ErasedSolver {
        match scheme {
            DifferenceSchemes::Colloc2 => {
                get_solvers_inner(model, system, || Colloc2 {}, solver_grid)
            }
            DifferenceSchemes::Colloc4 => {
                get_solvers_inner(model, system, || Colloc4 {}, solver_grid)
            }
            DifferenceSchemes::Magnus2 => {
                get_solvers_inner_explicit(model, system, || Magnus2 {}, solver_grid)
            }
            DifferenceSchemes::Magnus4 => {
                get_solvers_inner_explicit(model, system, || Magnus4 {}, solver_grid)
            }
            DifferenceSchemes::Magnus6 => {
                get_solvers_inner_explicit(model, system, || Magnus6 {}, solver_grid)
            }
            DifferenceSchemes::Magnus8 => {
                get_solvers_inner_explicit(model, system, || Magnus8 {}, solver_grid)
            }
        }
    }

    /// Compute the determinant for a certain frequency
    pub fn det(&self, freq: f64) -> f64 {
        (self.det)(freq)
    }

    /// Compute the eigenvectors for a certain frequency.
    ///
    /// This assumes that freq are close to a solution. This is less efficient than
    /// [ErasedSolver::det], so only use this after bracketing has completed.
    pub fn eigenvector(&self, freq: f64) -> Vec<f64> {
        (self.eigenvector)(freq).1
    }

    /// Scan all points given by `freq_grid` and optimize the resulting brackets to `precision`
    pub fn scan_and_optimize<'a>(
        &'a self,
        freq_grid: impl IntoIterator<Item = f64> + 'a,
        precision: Precision,
    ) -> impl Iterator<Item = BracketResult> + 'a {
        freq_grid
            .into_iter()
            .map(|x| Point { x, f: self.det(x) })
            .filter_sign_swap()
            .map(move |(point1, point2)| {
                (InverseQuadratic {})
                    .optimize(
                        point1,
                        point2,
                        |point| Ok::<_, !>(self.det(point)),
                        precision,
                        None,
                    )
                    .into_ok()
            })
    }
}

fn get_solvers_inner<T: ImplicitStepper + 'static>(
    model: &impl Model<ModelPoint = DimensionlessProperties>,
    system: Rotating1D,
    stepper: impl Fn() -> T,
    solver_grid: Option<&[f64]>,
) -> ErasedSolver
where
    DefaultAllocator:
        DeterminantAllocs<Const<4>, Const<2>> + ArrayAllocator<Const<4>, Const<4>, Dyn>,
    DefaultAllocator: ArrayAllocator<Const<4>, Const<4>, T::Points>,
{
    let system1 = DiscretizedSystemImpl::new(
        &LinearInterpolator::new(model),
        stepper(),
        system,
        solver_grid,
    );
    let system2 = DiscretizedSystemImpl::new(
        &LinearInterpolator::new(model),
        stepper(),
        system,
        solver_grid,
    );

    ErasedSolver {
        det: Box::new(move |freq: f64| determinant(&system1, freq)),
        eigenvector: Box::new(move |freq: f64| {
            let mut upper = UpperResult::new(system2.shape().value(), system2.len());

            let det = determinant_with_upper(&system2, freq, &mut upper);

            (det, upper.eigenvectors())
        }),
    }
}

fn get_solvers_inner_explicit<T: ExplicitStepper + 'static>(
    model: &impl Model<ModelPoint = DimensionlessProperties>,
    system: Rotating1D,
    stepper: impl Fn() -> T,
    solver_grid: Option<&[f64]>,
) -> ErasedSolver
where
    DefaultAllocator:
        DeterminantAllocs<Const<4>, Const<2>> + ArrayAllocator<Const<4>, Const<4>, Dyn>,
    DefaultAllocator: ArrayAllocator<Const<4>, Const<4>, T::Points>,
{
    let system1 = DiscretizedSystemImpl::new(
        &LinearInterpolator::new(model),
        stepper(),
        system,
        solver_grid,
    );
    let system2 = DiscretizedSystemImpl::new(
        &LinearInterpolator::new(model),
        stepper(),
        system,
        solver_grid,
    );

    ErasedSolver {
        det: Box::new(move |freq: f64| determinant_explicit(&system1, freq)),
        eigenvector: Box::new(move |freq: f64| {
            let mut upper = UpperResult::new(system2.shape().value(), system2.len());

            let det = determinant_with_upper(&system2, freq, &mut upper);

            (det, upper.eigenvectors())
        }),
    }
}

#[cfg(test)]
mod test {
    use std::{num::NonZeroU64, path::PathBuf};

    use itertools::Itertools;

    use crate::{bracket::Precision, model::gsm::StellarModel, system::adiabatic::Rotating1D};

    use super::{DifferenceSchemes, ErasedSolver};

    fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
        (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
    }

    fn compute_frequencies_radial(scheme: DifferenceSchemes) -> Vec<f64> {
        let model = {
            let main_dir: PathBuf = std::env::var("CARGO_MANIFEST_DIR").unwrap().into();
            let model_file = main_dir.join("test-data/test-model-zams.GSM");

            StellarModel::from_gsm(model_file).unwrap()
        };

        let system = Rotating1D::new(0, 0);
        let determinant = ErasedSolver::new(&model, system, scheme, None);
        let points = linspace(1.0, 25.0, 25);

        determinant
            .scan_and_optimize(
                points,
                Precision::ULP(const { NonZeroU64::new(1).unwrap() }),
            )
            .map(|res| res.root)
            .collect_vec()
    }

    #[test]
    fn test_frequencies_colloc2() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc2);

        assert_eq!(
            frequencies,
            [
                3.3047663569130536,
                4.266727223492225,
                5.171680659690917,
                6.112879114681654,
                7.202715354186323,
                8.382714828522186,
                9.59266836206982,
                10.775310463253373,
                11.920082692026774,
                13.053178735669679,
                14.210638397586763,
                15.393986102230592,
                16.5880292511668,
                17.785834047646592,
                18.98772299363563,
                20.19772931960939,
                21.41491994242974,
                22.63651632763216,
                23.861757854250527
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc4() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc4);
        assert_eq!(
            frequencies,
            [
                3.304759390925722,
                4.266719191005982,
                5.171684714065866,
                6.1128868978790365,
                7.202717475295715,
                8.382699981515483,
                9.59262895920255,
                10.775237833272321,
                11.91996604974901,
                13.052996583668008,
                14.210369118538319,
                15.39361003916573,
                16.58752825992596,
                17.785182622168932,
                18.98689209331154,
                20.19669235911406,
                21.41364351257505,
                22.634973746991786,
                23.85991347325063
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus2() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus2);
        assert_eq!(
            frequencies,
            [
                3.3047713704245685,
                4.266743680981154,
                5.171702299941156,
                6.112905983585803,
                7.202738950833405,
                8.38272397719048,
                9.59265226357175,
                10.775260427749853,
                11.919989125372584,
                13.053025064791559,
                14.210402390525093,
                15.393646992355476,
                16.58756669977432,
                17.785225703318673,
                18.986943176671076,
                20.19674853915816,
                21.413709426356597,
                22.635043899316315,
                23.85999187918837
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus4() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus4);
        assert_eq!(
            frequencies,
            [
                3.304759391648108,
                4.26671919274662,
                5.171684715567709,
                6.112886899795952,
                7.202717477992449,
                8.382699985122144,
                9.592628963845488,
                10.775237838963534,
                11.919966056603885,
                13.052996592400572,
                14.210369129464656,
                15.393610052143487,
                16.587528274555247,
                17.785182638743187,
                18.986892112467753,
                20.196692379988274,
                21.41364353540086,
                22.634973769188907,
                23.85991349447958
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus6() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus6);
        assert_eq!(
            frequencies,
            [
                3.3047593946264073,
                4.26671919874191,
                5.1716847197538005,
                6.1128869038819245,
                7.202717482211952,
                8.38269998899417,
                9.592628967346545,
                10.775237841907025,
                11.919966058823984,
                13.052996593538865,
                14.210369129178003,
                15.393610050060138,
                16.587528270322863,
                17.78518263130526,
                18.986892100317895,
                20.19669236243637,
                21.413643510617305,
                22.63497373698138,
                23.859913452528314
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus8() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus8);
        assert_eq!(
            frequencies,
            [
                3.3047593946234777,
                4.26671919873601,
                5.171684719749659,
                6.112886903877847,
                7.202717482207721,
                8.382699988990208,
                9.592628967342927,
                10.775237841903934,
                11.919966058821533,
                13.052996593537092,
                14.210369129177169,
                15.393610050060662,
                16.587528270325205,
                17.785182631309826,
                18.986892100325054,
                20.1966923624467,
                21.413643510631335,
                22.634973737000053,
                23.859913452552167
            ]
        );
    }
}
