//! Main interface for computing the determinant for a trial frequency

use nalgebra::{Const, DefaultAllocator, Dim, Dyn};

use crate::bracket::{
    BracketOptimizer as _, BracketResult, FilterSignSwap, InverseQuadratic, Point, Precision,
};
use crate::linalg::storage::ArrayAllocator;
use crate::model::interpolate::LinearInterpolator;
use crate::model::{DimensionlessProperties, Model};
use crate::solver::{DeterminantAllocs, UpperResult, determinant, determinant_with_upper};
use crate::stepper::{Colloc2, Colloc4, ImplicitStepper, Magnus2, Magnus4, Magnus6, Magnus8};
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
    ) -> ErasedSolver {
        match scheme {
            DifferenceSchemes::Colloc2 => get_solvers_inner(model, system, || Colloc2 {}),
            DifferenceSchemes::Colloc4 => get_solvers_inner(model, system, || Colloc4 {}),
            DifferenceSchemes::Magnus2 => get_solvers_inner(model, system, || Magnus2 {}),
            DifferenceSchemes::Magnus4 => get_solvers_inner(model, system, || Magnus4 {}),
            DifferenceSchemes::Magnus6 => get_solvers_inner(model, system, || Magnus6 {}),
            DifferenceSchemes::Magnus8 => get_solvers_inner(model, system, || Magnus8 {}),
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
) -> ErasedSolver
where
    DefaultAllocator:
        DeterminantAllocs<Const<4>, Const<2>> + ArrayAllocator<Const<4>, Const<4>, Dyn>,
    DefaultAllocator: ArrayAllocator<Const<4>, Const<4>, T::Points>,
{
    let system1 = DiscretizedSystemImpl::new(&LinearInterpolator::new(model), stepper(), system);
    let system2 = DiscretizedSystemImpl::new(&LinearInterpolator::new(model), stepper(), system);

    ErasedSolver {
        det: Box::new(move |freq: f64| determinant(&system1, freq)),
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
        let determinant = ErasedSolver::new(&model, system, scheme);
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
                7.2027153541863225,
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
                7.2027174752957155,
                8.382699981515483,
                9.59262895920255,
                10.775237833272321,
                11.919966049749013,
                13.052996583668008,
                14.21036911853832,
                15.393610039165731,
                16.58752825992596,
                17.785182622168932,
                18.98689209331154,
                20.196692359114056,
                21.413643512575046,
                22.63497374699179,
                23.859913473250632
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus2() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus2);
        assert_eq!(
            frequencies,
            [
                3.3047713704245694,
                4.266743680981153,
                5.171702299941156,
                6.112905983585802,
                7.202738950833407,
                8.382723977190482,
                9.592652263571749,
                10.77526042774985,
                11.919989125372588,
                13.05302506479156,
                14.21040239052509,
                15.39364699235548,
                16.587566699774317,
                17.785225703318673,
                18.986943176671076,
                20.196748539158158,
                21.4137094263566,
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
                3.3047593916481066,
                4.266719192746616,
                5.171684715567708,
                6.112886899795954,
                7.202717477992444,
                8.382699985122144,
                9.592628963845492,
                10.775237838963534,
                11.919966056603883,
                13.052996592400568,
                14.21036912946466,
                15.393610052143487,
                16.587528274555247,
                17.785182638743187,
                18.986892112467753,
                20.196692379988278,
                21.41364353540085,
                22.634973769188907,
                23.859913494479585
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus6() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus6);
        assert_eq!(
            frequencies,
            [
                3.3047593946264087,
                4.266719198741911,
                5.171684719753799,
                6.112886903881922,
                7.202717482211953,
                8.382699988994167,
                9.592628967346549,
                10.775237841907023,
                11.919966058823988,
                13.052996593538865,
                14.210369129178005,
                15.393610050060138,
                16.587528270322863,
                17.78518263130526,
                18.9868921003179,
                20.196692362436366,
                21.4136435106173,
                22.634973736981387,
                23.859913452528318
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus8() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus8);
        assert_eq!(
            frequencies,
            [
                3.304759394623479,
                4.26671919873601,
                5.171684719749656,
                6.112886903877849,
                7.202717482207723,
                8.382699988990208,
                9.592628967342932,
                10.775237841903934,
                11.919966058821533,
                13.052996593537092,
                14.210369129177167,
                15.393610050060659,
                16.587528270325205,
                17.785182631309826,
                18.98689210032505,
                20.1966923624467,
                21.413643510631335,
                22.63497373700006,
                23.859913452552167
            ]
        );
    }
}
