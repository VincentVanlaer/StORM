//! Main interface for computing the determinant for a trial frequency

use nalgebra::{Const, DefaultAllocator, Dim};

use crate::bracket::{
    BracketOptimizer as _, BracketResult, FilterSignSwap, InverseQuadratic, Point, Precision,
};
use crate::solver::{DeterminantAllocs, UpperResult, determinant, determinant_with_upper};
use crate::stepper::{Colloc2, Colloc4, Magnus2, Magnus4, Magnus6, Magnus8, Stepper};
use crate::system::System;
use crate::system::adiabatic::{GridScale, Rotating1D};

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
    /// Sixt-order magnus method
    Magnus6,
    /// Eigth-order magnus method
    Magnus8,
}

/// Type erased interface for computing the determinant and the eigenvector of a problem
pub struct MultipleShooting<'system_and_grid> {
    det: Box<dyn Fn(f64) -> f64 + 'system_and_grid>,
    eigenvector: Box<dyn Fn(f64) -> (f64, Vec<f64>) + 'system_and_grid>,
}

impl<'system_and_grid> MultipleShooting<'system_and_grid> {
    /// Construct from a system, difference scheme and grid definition
    pub fn new(
        system: &'system_and_grid Rotating1D,
        scheme: DifferenceSchemes,
        grid: &'system_and_grid GridScale,
    ) -> MultipleShooting<'system_and_grid> {
        match scheme {
            DifferenceSchemes::Colloc2 => get_solvers_inner(system, grid, || Colloc2 {}),
            DifferenceSchemes::Colloc4 => get_solvers_inner(system, grid, || Colloc4 {}),
            DifferenceSchemes::Magnus2 => get_solvers_inner(system, grid, || Magnus2 {}),
            DifferenceSchemes::Magnus4 => get_solvers_inner(system, grid, || Magnus4 {}),
            DifferenceSchemes::Magnus6 => get_solvers_inner(system, grid, || Magnus6 {}),
            DifferenceSchemes::Magnus8 => get_solvers_inner(system, grid, || Magnus8 {}),
        }
    }

    /// Compute the determinant for a certain frequency
    pub fn det(&self, freq: f64) -> f64 {
        (self.det)(freq)
    }

    /// Compute the eigenvectors for a certain frequency.
    ///
    /// This assumes that freq are close to a solution. This is less efficient than
    /// [MultipleShooting::det], so only use this after bracketing has completed.
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

fn get_solvers_inner<
    'a,
    N: Dim
        + nalgebra::DimSub<NInner>
        + nalgebra::DimMul<nalgebra::Const<2>>
        + nalgebra::DimAdd<NInner>,
    NInner: Dim,
    const ORDER: usize,
    G: ?Sized,
    S: System<f64, G, N, NInner, Const<ORDER>>,
    T: Stepper<f64, N, Const<ORDER>> + 'static,
>(
    system: &'a S,
    grid: &'a G,
    stepper: impl Fn() -> T,
) -> MultipleShooting<'a>
where
    DefaultAllocator: DeterminantAllocs<N, NInner, Const<ORDER>>,
{
    let stepper1 = stepper();
    let stepper2 = stepper();
    MultipleShooting {
        det: Box::new(move |freq: f64| determinant(system, &stepper1, grid, freq)),
        eigenvector: Box::new(move |freq: f64| {
            let mut upper = UpperResult::new(system.shape().value(), system.len(grid));

            let det = determinant_with_upper(system, &stepper2, grid, freq, &mut upper);

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
        model::StellarModel,
        system::adiabatic::{GridScale, Rotating1D},
    };

    use super::{DifferenceSchemes, MultipleShooting};

    fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
        (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
    }

    fn compute_frequencies_radial(scheme: DifferenceSchemes) -> Vec<f64> {
        let model = {
            let main_dir: PathBuf = std::env::var("CARGO_MANIFEST_DIR").unwrap().into();
            let model_file = main_dir.join("test-data/test-model-zams.GSM");

            StellarModel::from_gsm(model_file).unwrap()
        };

        let system = Rotating1D::from_model(&model, 0, 0);
        let determinant = MultipleShooting::new(&system, scheme, &GridScale { scale: 0 });
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
                3.30479694279111,
                4.2667915306564,
                5.17172880766053,
                6.112930394061321,
                7.202774101005002,
                8.382777523385197,
                9.592736355494658,
                10.77538297607734,
                11.920160441669331,
                13.05326602375103,
                14.210737686345766,
                15.394098074086251,
                16.58815426385916,
                17.78597344082322,
                18.98787865895907,
                20.197902625841344,
                21.41511269856594,
                22.63672979399207,
                23.86199363895272
            ]
        );
    }

    #[test]
    fn test_frequencies_colloc4() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Colloc4);
        assert_eq!(
            frequencies,
            [
                3.3047896449679843,
                4.266782871411036,
                5.171732424611369,
                6.112937739951343,
                7.202775746645694,
                8.382762195864952,
                9.592696471082316,
                10.77530988332594,
                11.920043357996246,
                13.053083431404987,
                14.210467959285369,
                15.393721553518894,
                16.587652805411693,
                17.785321533083316,
                18.98704725549496,
                20.196865138499653,
                21.413835715224057,
                22.635186632037843,
                23.860148646314116
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus2() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus2);
        assert_eq!(
            frequencies,
            [
                3.3048019432659745,
                4.266807960554894,
                5.171750428386139,
                6.112957242387451,
                7.202797673892647,
                8.382786646338415,
                9.592720228514478,
                10.775332909504652,
                11.920066840614448,
                13.053112312466066,
                14.210501630785924,
                15.393758906716727,
                16.587691644607055,
                17.78536501668777,
                18.987098747787538,
                20.19692173423978,
                21.41390205096295,
                22.635257211212295,
                23.86022748366069
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus4() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus4);
        assert_eq!(
            frequencies,
            [
                3.3047896558958634,
                4.266782894678752,
                5.171732442025188,
                6.112937758574142,
                7.202775768331735,
                8.382762219588976,
                9.592696497375957,
                10.775309911902587,
                11.920043389135222,
                13.053083467140631,
                14.210468000766209,
                15.393721600726396,
                16.587652857959355,
                17.785321591447847,
                18.987047320701805,
                20.196865209951476,
                21.41383579350626,
                22.635186714586958,
                23.860148732846703
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus6() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus6);
        assert_eq!(
            frequencies,
            [
                3.304789641149825,
                4.266782863369539,
                5.17173241868035,
                6.112937733797447,
                7.202775739799653,
                8.382762188825506,
                9.592696463664621,
                10.775309875571157,
                11.920043349735453,
                13.053083422056023,
                14.210467948252706,
                15.393721540216877,
                16.58765278909604,
                17.785321512864854,
                18.987047230243707,
                20.19686510650507,
                21.413835674669965,
                22.63518658015305,
                23.860148580415117
            ]
        );
    }

    #[test]
    fn test_frequencies_magnus8() {
        let frequencies = compute_frequencies_radial(DifferenceSchemes::Magnus8);
        assert_eq!(
            frequencies,
            [
                3.304789641128967,
                4.266782863325164,
                5.1717324186472275,
                6.112937733762265,
                7.202775739759051,
                8.382762188781655,
                9.592696463616349,
                10.775309875518843,
                11.920043349678426,
                13.053083421990706,
                14.210467948176362,
                15.393721540128249,
                16.587652788994053,
                17.78532151274762,
                18.987047230108402,
                20.19686510634906,
                21.413835674489754,
                22.635186579946158,
                23.86014858017804
            ]
        );
    }
}
