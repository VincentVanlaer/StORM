//! Main interface for computing the determinant for a trial frequency

use std::convert::Infallible;

use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, DimAdd, DimMul, DimSub, Dyn};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::bracket::{
    BracketError, BracketOptimizer as _, BracketResult, FilterSignSwap, InverseQuadratic, Point,
    Precision,
};
use crate::linalg::storage::ArrayAllocator;
use crate::model::ContinuousModel;
use crate::solver::{UpperResult, determinant, determinant_with_upper};
use crate::stepper::{Colloc2, Colloc4, Colloc6, Colloc8, ImplicitStepper};
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
                get_solvers_inner(model, system, || Colloc2 {}, solver_grid)
            }
            DifferenceSchemes::Colloc4 => {
                get_solvers_inner(model, system, || Colloc4 {}, solver_grid)
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
            .filter_map(move |(point1, point2)| {
                (InverseQuadratic {})
                    .optimize(
                        point1,
                        point2,
                        |point| Ok::<_, Infallible>(self.det(point)),
                        precision,
                        None,
                    )
                    .inspect_err(|err| match err {
                        BracketError::Eval(_) => {
                            unreachable!()
                        }
                        BracketError::Singularity => println!(
                            "Singularity between {:?} and {:?}, increase resolution",
                            point1, point2
                        ),
                    })
                    .ok()
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
                3.3047705823634708,
                4.266736343870042,
                5.171687595587037,
                6.1128865923248386,
                7.202723916218962,
                8.382723891526913,
                9.592678008203873,
                10.775320522017815,
                11.920093303555236,
                13.05319071087208,
                14.210652204099809,
                15.394001964368744,
                16.588047356802633,
                17.785854769226447,
                18.98774664124729,
                20.197756136081438,
                21.414950182959398,
                22.636550178635108,
                23.86179552580159
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc4() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc4);
        assert_eq!(
            frequencies,
            [
                3.304771988576296,
                4.266745906197422,
                5.171704821259237,
                6.112908403558026,
                7.202742135696427,
                8.382726276718955,
                9.592657394452715,
                10.775268057219623,
                11.919998396210872,
                13.053032995656858,
                14.210410711288215,
                15.393657185848332,
                16.58758120337435,
                17.785242049776524,
                18.986958831631522,
                20.1967670189284,
                21.413726857738695,
                22.63506631842195,
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
                3.304771988449559,
                4.266745905913491,
                5.171704821041752,
                6.112908403377998,
                7.202742135604172,
                8.382726276786668,
                9.592657394623767,
                10.775268057417655,
                11.91999839631077,
                13.053032995510076,
                14.21041071055129,
                15.393657183936188,
                16.587581199503898,
                17.785242043062077,
                18.986958820998424,
                20.196767002682407,
                21.413726834087406,
                22.635066284488612,
                23.860015894902055
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc8() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc8);
        assert_eq!(
            frequencies,
            [
                3.304771988449267,
                4.266745905912847,
                5.171704821041254,
                6.1129084033774514,
                7.2027421356035175,
                8.38272627678594,
                9.592657394622933,
                10.77526805741672,
                11.919998396309726,
                13.053032995508845,
                14.210410710549814,
                15.393657183934423,
                16.5875811995018,
                17.78524204305959,
                18.986958820995444,
                20.196767002678833,
                21.413726834083093,
                22.635066284483425,
                23.86001589489581
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

    #[test]
    fn test_singularities() {
        let model = Polytrope0 { gamma1: 5. / 3. };

        let solver = ErasedSolver::new(
            &model,
            Rotating1D::new(0, 0),
            DifferenceSchemes::Colloc6,
            &[&linspace(0., 1., 10).collect_vec()],
        );

        let results = solver
            .scan_and_optimize(
                [3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13.].into_iter(),
                Precision::Relative(1e-2),
            )
            .into_iter()
            .map(|res| res.root)
            .collect_vec();

        assert_eq!(results.len(), 4);
    }
}
