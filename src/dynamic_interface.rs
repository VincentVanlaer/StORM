//! Main interface for computing the determinant for a trial frequency

use nalgebra::{Const, DefaultAllocator, Dim, Dyn};

use crate::bracket::{
    BracketOptimizer as _, BracketResult, FilterSignSwap, InverseQuadratic, Point, Precision,
};
use crate::linalg::storage::ArrayAllocator;
use crate::model::DimensionlessProperties;
use crate::model::interpolate::InterpolatingModel;
use crate::solver::{
    DeterminantAllocs, UpperResult, determinant, determinant_explicit, determinant_with_upper,
};
use crate::stepper::{
    Colloc2, Colloc4, Colloc6, Colloc8, ExplicitStepper, ImplicitStepper, Magnus2, Magnus4,
    Magnus6, Magnus8,
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
    /// Sixth-order collocation method
    Colloc6,
    /// Eight-order collocation method
    Colloc8,
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
        model: &impl InterpolatingModel<ModelPoint = DimensionlessProperties>,
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
            DifferenceSchemes::Colloc6 => {
                get_solvers_inner(model, system, || Colloc6 {}, solver_grid)
            }
            DifferenceSchemes::Colloc8 => {
                get_solvers_inner(model, system, || Colloc8 {}, solver_grid)
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
    model: &impl InterpolatingModel<ModelPoint = DimensionlessProperties>,
    system: Rotating1D,
    stepper: impl Fn() -> T,
    solver_grid: Option<&[f64]>,
) -> ErasedSolver
where
    DefaultAllocator:
        DeterminantAllocs<Const<4>, Const<2>> + ArrayAllocator<Const<4>, Const<4>, Dyn>,
    DefaultAllocator: ArrayAllocator<Const<4>, Const<4>, T::Points>,
{
    let system1 = DiscretizedSystemImpl::new(model, stepper(), system, solver_grid);
    let system2 = DiscretizedSystemImpl::new(model, stepper(), system, solver_grid);

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
    model: &impl InterpolatingModel<ModelPoint = DimensionlessProperties>,
    system: Rotating1D,
    stepper: impl Fn() -> T,
    solver_grid: Option<&[f64]>,
) -> ErasedSolver
where
    DefaultAllocator:
        DeterminantAllocs<Const<4>, Const<2>> + ArrayAllocator<Const<4>, Const<4>, Dyn>,
    DefaultAllocator: ArrayAllocator<Const<4>, Const<4>, T::Points>,
{
    let system1 = DiscretizedSystemImpl::new(model, stepper(), system, solver_grid);
    let system2 = DiscretizedSystemImpl::new(model, stepper(), system, solver_grid);

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

    use crate::{
        bracket::Precision,
        model::{gsm::StellarModel, interpolate::LinearInterpolator},
        system::adiabatic::Rotating1D,
    };

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
        let determinant = ErasedSolver::new(&LinearInterpolator::new(&model), system, scheme, None);
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
                3.3047706224288893,
                4.266736429284347,
                5.171687592680076,
                6.112886512313282,
                7.202723887616655,
                8.382723981660297,
                9.592678313985463,
                10.775321077733329,
                11.920094055050019,
                13.053191486119086,
                14.21065295628957,
                15.394002573234957,
                16.58804769210533,
                17.785854624588193,
                18.987745988393094,
                20.19775496393888,
                21.414948539294045,
                22.636548053365317,
                23.861792937203628
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc4() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc4);
        assert_eq!(
            frequencies,
            [
                3.3047720152869338,
                4.266745963140683,
                5.171704819321423,
                6.112908350218141,
                7.2027421166306125,
                8.38272633680723,
                9.592657598295146,
                10.775268427670367,
                11.919998897183374,
                13.053033512478905,
                14.21041121274171,
                15.393657591778869,
                16.587581426994063,
                17.785241953530686,
                18.98695839666501,
                20.1967662378476,
                21.41372576241617,
                22.635064902126697,
                23.860014216829438
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc6() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc6);
        assert_eq!(
            frequencies,
            [
                3.304772015160266,
                4.2667459628569375,
                5.171704819103981,
                6.1129083500378405,
                7.202742116538021,
                8.382726336875065,
                9.592657598467904,
                10.77526842787161,
                11.91999889728694,
                13.053033512335519,
                14.21041121201184,
                15.39365758987505,
                16.587581423130356,
                17.785241946815045,
                18.986958386024746,
                20.19676622158744,
                21.413725738741416,
                22.635064868154704,
                23.86001416975952
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc8() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc8);
        assert_eq!(
            frequencies,
            [
                3.3047720151599753,
                4.266745962856291,
                5.171704819103482,
                6.1129083500372925,
                7.202742116537371,
                8.382726336874333,
                9.59265759846707,
                10.77526842787068,
                11.919998897285893,
                13.053033512334292,
                14.210411212010365,
                15.393657589873284,
                16.587581423128256,
                17.785241946812558,
                18.986958386021772,
                20.196766221583868,
                21.413725738737106,
                22.635064868149517,
                23.860014169753267
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus2() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus2);
        assert_eq!(
            frequencies,
            [
                3.30477564486192,
                4.2667529056557845,
                5.171709247093282,
                6.112913396187004,
                7.202747501398336,
                8.382733148640332,
                9.59266223539249,
                10.77527106351291,
                11.920000511225336,
                13.053037840875717,
                14.21041697853087,
                15.393663496634268,
                16.587585178101516,
                17.78524632216862,
                18.986966218448746,
                20.19677423610987,
                21.413738082082,
                22.63507569061922,
                23.86002703503698
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus4() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus4);
        assert_eq!(
            frequencies,
            [
                3.3047720391382653,
                4.2667460137112645,
                5.171704856969687,
                6.112908390164325,
                7.202742162680113,
                8.382726386518636,
                9.592657652794458,
                10.775268486353088,
                11.919998960620605,
                13.053033584573424,
                14.21041129591667,
                15.393657686411643,
                16.587581532970084,
                17.785242071913313,
                18.986958529411734,
                20.19676638516724,
                21.413725925939005,
                22.63506507996895,
                23.86001440955365
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus6() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus6);
        assert_eq!(
            frequencies,
            [
                3.304771998741276,
                4.266745928166955,
                5.171704793392827,
                6.112908322941874,
                7.20274208557778,
                8.382726303849733,
                9.592657562621858,
                10.77526838960843,
                11.91999885623533,
                13.053033466120043,
                14.210411159011505,
                15.393657529585496,
                16.58758135520472,
                17.78524187050612,
                18.98695830014606,
                20.19676612511259,
                21.41372563034037,
                22.635064746960737,
                23.86001403469456
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus8() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus8);
        assert_eq!(
            frequencies,
            [
                3.304772023512105,
                4.2667459805024395,
                5.171704832182101,
                6.112908363820135,
                7.202742132285545,
                8.382726353672554,
                9.592657616699562,
                10.775268447332078,
                11.919998918164925,
                13.053033535838988,
                14.210411238964875,
                15.393657620533789,
                16.58758145767078,
                17.785241985616697,
                18.986958429690507,
                20.19676627063857,
                21.41372579385351,
                22.635064929767445,
                23.86001423842012
            ]
        );
    }
}
