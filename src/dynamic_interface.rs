//! Main interface for computing the determinant for a trial frequency

use std::convert::Infallible;

use nalgebra::allocator::Allocator;
use nalgebra::{Const, DefaultAllocator, DimAdd, DimMul, DimSub, Dyn};
#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::bracket::{
    BracketError, BracketOptimizer as _, BracketResult, FilterSignSwap, InverseQuadratic, Point,
    Precision,
};
use crate::linalg::storage::ArrayAllocator;
use crate::model::ContinuousModel;
use crate::solver::{UpperResult, determinant, determinant_with_upper};
use crate::stepper::{Colloc2, Colloc4, Colloc6, Colloc8, ImplicitStepper};
use crate::system::adiabatic::{Radial, Rotating1D};
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
        degree: u64,
        order: i64,
        scheme: DifferenceSchemes,
        solver_grid: &[&[f64]],
    ) -> ErasedSolver {
        #[allow(deprecated)]
        match scheme {
            DifferenceSchemes::Colloc2 => {
                get_solvers_inner(model, degree, order, || Colloc2 {}, solver_grid)
            }
            DifferenceSchemes::Colloc4 => {
                get_solvers_inner(model, degree, order, || Colloc4 {}, solver_grid)
            }
            DifferenceSchemes::Colloc6 => {
                get_solvers_inner(model, degree, order, || Colloc6 {}, solver_grid)
            }
            DifferenceSchemes::Colloc8 => {
                get_solvers_inner(model, degree, order, || Colloc8 {}, solver_grid)
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
        let mut singularity_lower = f64::INFINITY;
        let mut singularity_upper = f64::NEG_INFINITY;
        let mut singularity_count = 0;

        let find_brackets = |pair: &(Point, Point)| {
            if pair.0.f.is_finite() && pair.1.f.is_finite() {
                true
            } else {
                has_invalid_numbers = true;
                false
            }
        };

        let bracket = |(point1, point2)| {
            (
                point1,
                point2,
                (InverseQuadratic {}).optimize(
                    point1,
                    point2,
                    |point| Ok::<_, Infallible>(self.det(point)),
                    precision,
                    None,
                ),
            )
        };

        let process_results = |(point1, point2, result): (Point, Point, Result<_, _>)| {
            result
                .inspect_err(|err| match err {
                    BracketError::Eval(_) => {
                        unreachable!()
                    }
                    BracketError::Singularity => {
                        if singularity_lower > point1.x {
                            singularity_lower = point1.x;
                        }

                        if singularity_upper < point2.x {
                            singularity_upper = point2.x
                        }
                        singularity_count += 1;
                    }
                })
                .ok()
        };

        #[cfg(feature = "parallel")]
        let results = freq_grid
            .into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|x| Point { x, f: self.det(x) })
            .collect::<Vec<_>>()
            .into_iter()
            .filter_sign_swap()
            .filter(find_brackets)
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(bracket)
            .collect::<Vec<_>>()
            .into_iter()
            .filter_map(process_results)
            .collect();

        #[cfg(not(feature = "parallel"))]
        let results = freq_grid
            .into_iter()
            .map(|x| Point { x, f: self.det(x) })
            .filter_sign_swap()
            .filter(find_brackets)
            .map(bracket)
            .filter_map(process_results)
            .collect();

        if has_invalid_numbers {
            println!(
                "Invalid numbers encountered in initial scan, suspected low resolution or rotation too high for scan."
            )
        }

        if singularity_count > 0 {
            println!(
                "{} singularities encountered between {} and {}",
                singularity_count, singularity_lower, singularity_upper
            );
            println!(
                "This typically indicates that the model grid resolution is too low or the frequency scan range is too low for the rotation rate. See the FAQ page on the documentation website for more info."
            )
        }

        results
    }
}

fn get_solvers_inner<T: ImplicitStepper + Sync + 'static>(
    model: &(impl ContinuousModel + ?Sized),
    degree: u64,
    order: i64,
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
        + ArrayAllocator<Const<4>, Const<4>, T::Points>
        + Allocator<Const<2>, Const<2>>
        + Allocator<Const<1>, Const<2>>
        + Allocator<<Const<2> as DimSub<Const<1>>>::Output, Const<2>>
        + Allocator<<Const<2> as DimMul<Const<2>>>::Output, <Const<2> as DimAdd<Const<1>>>::Output>
        + Allocator<<Const<2> as DimMul<Const<2>>>::Output, Const<1>>
        + ArrayAllocator<Const<2>, Const<2>, Dyn>
        + ArrayAllocator<Const<2>, Const<2>, T::Points>,
{
    let (det, eigenvector): (
        Box<dyn Fn(f64) -> f64 + Sync>,
        Box<dyn Fn(f64) -> (f64, Vec<f64>) + Sync>,
    ) = if degree == 0 {
        let system = DiscretizedSystemImpl::new(model, stepper(), Radial {}, solver_grid);
        let det = Box::new(move |freq: f64| determinant(&system, freq));

        let system = DiscretizedSystemImpl::new(model, stepper(), Radial {}, solver_grid);
        let eigenvector = Box::new(move |freq: f64| {
            let mut upper = UpperResult::new_from_system(&system);
            let det = determinant_with_upper(&system, freq, &mut upper);

            (det, upper.eigenvectors())
        });

        (det, eigenvector)
    } else {
        let system = DiscretizedSystemImpl::new(
            model,
            stepper(),
            Rotating1D::new(degree, order),
            solver_grid,
        );
        let det = Box::new(move |freq: f64| determinant(&system, freq));

        let system = DiscretizedSystemImpl::new(
            model,
            stepper(),
            Rotating1D::new(degree, order),
            solver_grid,
        );

        let eigenvector = Box::new(move |freq: f64| {
            let mut upper = UpperResult::new_from_system(&system);
            let det = determinant_with_upper(&system, freq, &mut upper);

            (det, upper.eigenvectors())
        });

        (det, eigenvector)
    };

    ErasedSolver { det, eigenvector }
}

#[cfg(test)]
mod test {
    use std::{num::NonZeroU64, path::PathBuf};

    use itertools::Itertools;

    use crate::{
        bracket::Precision,
        model::{
            DiscreteModel, interpolate::LinearInterpolator, loader::ModelFormat,
            polytrope::Polytrope0,
        },
    };

    use super::{DifferenceSchemes, ErasedSolver};

    fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
        (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
    }

    fn compute_frequencies(scheme: DifferenceSchemes, degree: u64) -> Vec<f64> {
        let model = {
            let main_dir: PathBuf = std::env::var("CARGO_MANIFEST_DIR").unwrap().into();
            let model_file = main_dir.join("test-data/test-model-zams.GSM");

            DiscreteModel::load(&model_file, ModelFormat::GSM).unwrap()
        };

        let determinant = ErasedSolver::new(
            &LinearInterpolator::new(&model),
            degree,
            0,
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
        let frequencies = compute_frequencies(DifferenceSchemes::Colloc2, 1);

        assert_eq!(
            frequencies,
            [
                3.535201011228379,
                4.708493801114231,
                5.622283033792152,
                6.637081303839896,
                7.787238819506522,
                8.975102056603092,
                10.16890639225985,
                11.328480127162212,
                12.456756151200729,
                13.596499939016413,
                14.765682238401588,
                15.953908717795468,
                17.148826879008027,
                18.34731613455383,
                19.55253105426822,
                20.76487199954969,
                21.984359921453198,
                23.20766192966448,
                24.433553704284236
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc4() {
        let frequencies = compute_frequencies(DifferenceSchemes::Colloc4, 1);
        assert_eq!(
            frequencies,
            [
                3.535204154312691,
                4.708502672978331,
                5.622297145286964,
                6.6370918782803265,
                7.787237364007961,
                8.975079647601149,
                10.16885600533052,
                11.328393100682678,
                12.456616220927778,
                13.596281034199258,
                14.765368167270632,
                15.953480477256946,
                17.148266528872618,
                18.34659411985872,
                19.55161739736722,
                20.76373718225713,
                21.982978060494954,
                23.206002871144456,
                24.431580821844044
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc6() {
        let frequencies = compute_frequencies(DifferenceSchemes::Colloc6, 1);
        assert_eq!(
            frequencies,
            [
                3.5352041543258275,
                4.708502672999094,
                5.622297145335113,
                6.637091878432071,
                7.787237364303527,
                8.975079648031652,
                10.168856005817386,
                11.328393101106105,
                12.456616221162692,
                13.596281034107673,
                14.765368166293726,
                15.953480474754993,
                17.148266523870348,
                18.346594111497154,
                19.55161738446003,
                20.763737162956268,
                21.98297803239097,
                23.206002831397097,
                24.431580767727027
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc8() {
        let frequencies = compute_frequencies(DifferenceSchemes::Colloc8, 1);
        assert_eq!(
            frequencies,
            [
                3.5352041543258292,
                4.708502672999098,
                5.622297145335121,
                6.637091878432084,
                7.787237364303547,
                8.975079648031677,
                10.168856005817425,
                11.328393101106153,
                12.456616221162752,
                13.596281034107754,
                14.765368166293829,
                15.953480474755118,
                17.1482665238705,
                18.346594111497332,
                19.551617384460208,
                20.76373716295644,
                21.982978032391106,
                23.206002831397146,
                24.431580767726913
            ]
        );
    }

    #[test]
    fn test_frequencies_radial() {
        let frequencies = compute_frequencies(DifferenceSchemes::Colloc2, 0);
        assert_eq!(
            frequencies,
            [
                3.3047809428365422,
                4.266748394071127,
                5.171693107854907,
                6.112890318645434,
                7.202726484657045,
                8.382725271699115,
                9.592678470414542,
                10.775320438533122,
                11.92009299796021,
                13.053190370072313,
                14.210651892830096,
                15.394001607990237,
                16.588046889370663,
                17.785854228084066,
                18.98774612289027,
                20.197755749714883,
                21.41494993956211,
                22.63655003007807,
                23.861795461044103
            ]
        )
    }

    #[test]
    fn test_polytrope() {
        let model = Polytrope0 { gamma1: 5. / 3. };

        let solver = ErasedSolver::new(
            &model,
            0,
            0,
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
            0,
            0,
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
