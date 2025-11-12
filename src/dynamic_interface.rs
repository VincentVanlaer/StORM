//! Main interface for computing the determinant for a trial frequency

use std::convert::Infallible;

use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, DimAdd, DimMul, DimSub, Dyn};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::bracket::{
    BracketOptimizer as _, BracketResult, FilterSignSwap, InverseQuadratic, Point, Precision,
};
use crate::linalg::storage::ArrayAllocator;
use crate::model::ContinuousModel;
use crate::solver::{UpperResult, determinant, determinant_explicit, determinant_with_upper};
use crate::stepper::{
    Colloc2, Colloc4, Colloc6, Colloc8, ExplicitStepper, ImplicitStepper, Magnus2, Magnus4,
    Magnus6, Magnus8,
};
use crate::system::adiabatic::Rotating1D;
use crate::system::discretized::DiscretizedSystemImpl;

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
    #[deprecated(
        note = "magnus methods do not converge properly, use the collocation methods instead"
    )]
    #[clap(hide = true)]
    Magnus2,
    /// Fourth-order magnus method
    #[deprecated(
        note = "magnus methods do not converge properly, use the collocation methods instead"
    )]
    #[clap(hide = true)]
    Magnus4,
    /// Sixth-order magnus method
    #[deprecated(
        note = "magnus methods do not converge properly, use the collocation methods instead"
    )]
    #[clap(hide = true)]
    Magnus6,
    /// Eight-order magnus method
    #[deprecated(
        note = "magnus methods do not converge properly, use the collocation methods instead"
    )]
    #[clap(hide = true)]
    Magnus8,
}

/// Type erased interface for computing the determinant and the eigenvector of a problem
pub struct ErasedSolver {
    det: Box<dyn Fn(f64) -> f64 + Sync>,
    eigenvector: Box<dyn Fn(f64) -> (f64, Vec<f64>) + Sync>,
}

impl ErasedSolver {
    /// Construct from a system, difference scheme and grid definition
    pub fn new(
        model: &(impl ContinuousModel + ?Sized),
        system: Rotating1D,
        scheme: DifferenceSchemes,
        solver_grid: &[&[f64]],
    ) -> ErasedSolver {
        #[allow(deprecated)]
        match scheme {
            DifferenceSchemes::Colloc2 => {
                get_solvers_inner_explicit(model, system, || Colloc2 {}.as_explicit(), solver_grid)
            }
            DifferenceSchemes::Colloc4 => {
                get_solvers_inner_explicit(model, system, || Colloc4 {}.as_explicit(), solver_grid)
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
                get_solvers_inner_explicit(model, system, || Colloc6 {}.as_explicit(), solver_grid)
            }
            DifferenceSchemes::Colloc8 => {
                get_solvers_inner_explicit(model, system, || Colloc8 {}.as_explicit(), solver_grid)
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
    pub fn scan_and_optimize(
        &self,
        freq_grid: impl IntoIterator<Item = f64, IntoIter: Send>,
        precision: Precision,
    ) -> Vec<BracketResult> {
        let mut has_invalid_numbers = false;
        let results = freq_grid
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|x| Point { x, f: self.det(x) })
            .collect::<Vec<_>>()
            .into_iter()
            .filter_sign_swap()
            .filter(|pair| {
                if pair.0.f.is_finite() && pair.1.f.is_finite() {
                    true
                } else {
                    has_invalid_numbers = true;
                    false
                }
            })
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(move |(point1, point2)| {
                (InverseQuadratic {})
                    .optimize(
                        point1,
                        point2,
                        |point| Ok::<_, Infallible>(self.det(point)),
                        precision,
                        None,
                    )
                    .unwrap()
            })
            .collect();

        if has_invalid_numbers {
            println!(
                "Invalid numbers encountered in initial scan, suspected rotation too high for scan."
            )
        }

        results
    }
}

fn get_solvers_inner<T: ImplicitStepper + Sync + 'static>(
    model: &(impl ContinuousModel + ?Sized),
    system: Rotating1D,
    stepper: impl Fn() -> T,
    solver_grid: &[&[f64]],
) -> ErasedSolver
where
    DefaultAllocator: Allocator<Const<4>, Const<4>>
        + Allocator<Const<2>, Const<4>>
        + Allocator<<Const<4> as DimSub<Const<2>>>::Output, Const<4>>
        + Allocator<<Const<4> as DimMul<Const<2>>>::Output, <Const<4> as DimAdd<Const<2>>>::Output>
        + Allocator<<Const<4> as DimMul<Const<2>>>::Output, Const<1>>
        + ArrayAllocator<Const<4>, Const<4>, Dyn>
        + ArrayAllocator<Const<4>, Const<4>, T::Points>,
{
    let system1 = DiscretizedSystemImpl::new(model, stepper(), system, solver_grid);
    let system2 = DiscretizedSystemImpl::new(model, stepper(), system, solver_grid);

    ErasedSolver {
        det: Box::new(move |freq: f64| determinant(&system1, freq)),
        eigenvector: Box::new(move |freq: f64| {
            let mut upper = UpperResult::new_from_system(&system2);

            let det = determinant_with_upper(&system2, freq, &mut upper);

            (det, upper.eigenvectors())
        }),
    }
}

fn get_solvers_inner_explicit<T: ExplicitStepper + Sync + 'static>(
    model: &(impl ContinuousModel + ?Sized),
    system: Rotating1D,
    stepper: impl Fn() -> T,
    solver_grid: &[&[f64]],
) -> ErasedSolver
where
    DefaultAllocator: Allocator<Const<4>, Const<4>>
        + Allocator<Const<2>, Const<4>>
        + Allocator<<Const<4> as DimSub<Const<2>>>::Output, Const<4>>
        + Allocator<<Const<4> as DimMul<Const<2>>>::Output, <Const<4> as DimAdd<Const<2>>>::Output>
        + Allocator<<Const<4> as DimMul<Const<2>>>::Output, Const<1>>
        + ArrayAllocator<Const<4>, Const<4>, Dyn>
        + ArrayAllocator<Const<4>, Const<4>, T::Points>,
{
    let system1 = DiscretizedSystemImpl::new(model, stepper(), system, solver_grid);
    let system2 = DiscretizedSystemImpl::new(model, stepper(), system, solver_grid);

    ErasedSolver {
        det: Box::new(move |freq: f64| determinant_explicit(&system1, freq)),
        eigenvector: Box::new(move |freq: f64| {
            let mut upper = UpperResult::new_from_system(&system2);

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
        model::{DiscreteModel, interpolate::LinearInterpolator, polytrope::Polytrope0},
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

            DiscreteModel::from_gsm(model_file).unwrap()
        };

        let system = Rotating1D::new(0, 0);
        let determinant = ErasedSolver::new(
            &LinearInterpolator::new(&model),
            system,
            scheme,
            &model
                .segments
                .iter()
                .map(|x| x.dimensionless.r_coord.as_ref())
                .collect_vec(),
        );
        let points = linspace(1.0, 25.0, 25);

        determinant
            .scan_and_optimize(
                points,
                Precision::ULP(const { NonZeroU64::new(1).unwrap() }),
            )
            .into_iter()
            .map(|res| res.root)
            .collect_vec()
    }

    #[test]
    fn test_frequencies_colloc2() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc2);

        assert_eq!(
            frequencies,
            [
                3.3047705823634703,
                4.266736343870047,
                5.171687595587034,
                6.112886592324839,
                7.202723916218958,
                8.382723891526915,
                9.592678008203873,
                10.775320522017816,
                11.920093303555236,
                13.053190710872078,
                14.210652204099809,
                15.394001964368739,
                16.588047356802633,
                17.78585476922645,
                18.987746641247288,
                20.197756136081438,
                21.414950182959394,
                22.636550178635105,
                23.861795525801593
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc4() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc4);
        assert_eq!(
            frequencies,
            [
                3.304771988576298,
                4.26674590619742,
                5.171704821259237,
                6.112908403558029,
                7.202742135696432,
                8.382726276718957,
                9.592657394452718,
                10.775268057219629,
                11.919998396210874,
                13.05303299565686,
                14.210410711288217,
                15.393657185848332,
                16.58758120337435,
                17.785242049776528,
                18.986958831631526,
                20.196767018928398,
                21.41372685773869,
                22.63506631842194,
                23.86001594191292
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc6() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc6);
        assert_eq!(
            frequencies,
            [
                3.3047719884495614,
                4.26674590591349,
                5.171704821041756,
                6.112908403377997,
                7.202742135604173,
                8.382726276786672,
                9.592657394623771,
                10.775268057417655,
                11.919998396310769,
                13.053032995510076,
                14.210410710551296,
                15.393657183936186,
                16.587581199503894,
                17.785242043062084,
                18.986958820998424,
                20.196767002682407,
                21.413726834087406,
                22.635066284488612,
                23.86001589490206
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc8() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc8);
        assert_eq!(
            frequencies,
            [
                3.3047719884492697,
                4.266745905912844,
                5.171704821041253,
                6.11290840337745,
                7.202742135603518,
                8.382726276785938,
                9.592657394622938,
                10.77526805741672,
                11.919998396309726,
                13.053032995508843,
                14.210410710549818,
                15.39365718393442,
                16.5875811995018,
                17.785242043059586,
                18.986958820995447,
                20.196767002678833,
                21.413726834083093,
                22.635066284483422,
                23.860015894895806
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus2() {
        #[allow(deprecated)]
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus2);
        assert_eq!(
            frequencies,
            [
                3.3047756048134764,
                4.266752820276893,
                5.171709250026434,
                6.112913476223278,
                7.202747530027676,
                8.38273305853886,
                9.592661929656748,
                10.77527050786098,
                11.91999975979777,
                13.053037065684244,
                14.210416226382524,
                15.393662887763972,
                16.58758484269586,
                17.785246466563454,
                18.986966870916923,
                20.19677540774143,
                21.41373972509688,
                22.635077815112005,
                23.86002962270805
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus4() {
        #[allow(deprecated)]
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus4);
        assert_eq!(
            frequencies,
            [
                3.304772012427685,
                4.266745956768099,
                5.171704858907576,
                6.112908443504287,
                7.202742181746013,
                8.38272632643045,
                9.592657448952124,
                10.775268115902444,
                11.919998459648216,
                13.053033067751512,
                14.210410794463336,
                15.393657280481298,
                16.587581309350604,
                17.78524216815943,
                18.986958964378562,
                20.196767166248403,
                21.413727021261945,
                22.635066496264685,
                23.860016134637675
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus6() {
        #[allow(deprecated)]
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus6);
        assert_eq!(
            frequencies,
            [
                3.304771972030555,
                4.266745871223452,
                5.171704795330556,
                6.112908376281986,
                7.20274210464388,
                8.382726243761287,
                9.592657358777672,
                10.775268019154412,
                11.919998355259096,
                13.053032949294519,
                14.210410657550867,
                15.393657123646523,
                16.587581131578123,
                17.785241966752977,
                18.986958735119533,
                20.19676690620731,
                21.41372672568609,
                22.63506616329432,
                23.860015759836728
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus8() {
        #[allow(deprecated)]
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus8);
        assert_eq!(
            frequencies,
            [
                3.3047719968014135,
                4.266745923559007,
                5.171704834119887,
                6.112908417160295,
                7.202742151351703,
                8.382726293584165,
                9.592657412855438,
                10.775268076878126,
                11.919998417188772,
                13.053033019013561,
                14.210410737504352,
                15.393657214594954,
                16.587581234044354,
                17.785242081863778,
                18.98695886466423,
                20.196767051733598,
                21.41372688919957,
                22.635066346101432,
                23.860015963562738
            ]
        );
    }

    #[test]
    fn test_polytrope() {
        let model = Polytrope0 { gamma1: 5. / 3. };

        let solver = ErasedSolver::new(
            &model,
            Rotating1D::new(0, 0),
            DifferenceSchemes::Colloc6,
            &[&linspace(0., 1., 10000).collect_vec()],
        );

        let results =
            solver.scan_and_optimize([3., 4.].into_iter(), Precision::ULP(1.try_into().unwrap()));

        assert_eq!(results.len(), 1);
        assert!(dbg!(results[0].root / model.exact(0, 1) - 1.) < 1e-13);
    }
}
