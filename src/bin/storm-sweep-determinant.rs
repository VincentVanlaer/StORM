use clap::Parser;
use color_eyre::{
    Result,
    eyre::{Context, eyre},
};
use itertools::Itertools;
use storm::{
    dynamic_interface::{DifferenceSchemes, ErasedSolver},
    model::{
        ContinuousModel, DimensionedProperties, DiscreteModel, interpolate::LinearInterpolator,
        loader::ModelFormat,
    },
};

#[derive(Debug, Parser)]
struct Command {
    /// Location of the stellar model. The stellar model should be an HDF5 GYRE model file.
    file: String,
    /// How many times should each datapoint of the input model be subdivided.
    #[arg(long, default_value = "1")]
    resample: usize,
    /// Format of the file
    #[arg(long, default_value = "gsm")]
    format: ModelFormat,
    /// Spherical degree
    ell: u64,
    /// Azimuthal order
    #[arg(allow_negative_numbers = true)]
    m: i64,
    /// Lower frequency of the scan range
    #[arg(allow_negative_numbers = true)]
    lower: f64,
    /// Upper frequency of the scan range
    #[arg(allow_negative_numbers = true)]
    upper: f64,
    /// Number of scanning steps
    steps: usize,
    /// Whether to do steps between lower and upper linear in period (inverse) or in frequency
    #[arg(long)]
    inverse: bool,
    /// Relative precision required.
    ///
    /// Due to the bracketing method, the actual precision of the result can be a couple of orders of magnitude better.
    /// Unless comparing different oscillation codes or methods of computation, a reasonable
    /// precision is 1e-8, which is the default.
    #[arg(long, default_value = "1e-8")]
    precision: f64,
    /// Units of lower and upper
    #[arg(long, default_value = "dynamical")]
    frequency_units: FrequencyUnits,
    /// The file to write the data to
    output: String,
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum FrequencyUnits {
    /// Dynamical frequency of the star [sqrt(GM/R^3)]
    Dynamical,
    /// Hertz [1/s]
    Hertz,
    /// Cycles per day [1/d]
    CyclesPerDay,
}

impl FrequencyUnits {
    fn scale_factor(&self, model: &Option<DimensionedProperties>) -> Result<f64> {
        match self {
            FrequencyUnits::Dynamical => Ok(1.),
            FrequencyUnits::Hertz => model
                .map(|model| model.freq_scale() / 2. / std::f64::consts::PI)
                .ok_or(eyre!(
                    "Input model is dimensionless, only dynamical frequency is supported"
                )),
            FrequencyUnits::CyclesPerDay => model
                .map(|model| model.freq_scale() * 86400. / 2. / std::f64::consts::PI)
                .ok_or(eyre!(
                    "Input model is dimensionless, only dynamical frequency is supported"
                )),
        }
    }

    fn convert_to_natural(&self, freq: f64, model: &Option<DimensionedProperties>) -> Result<f64> {
        self.scale_factor(model).map(|s| freq / s)
    }
}

fn main() -> Result<()> {
    let args = Command::parse();

    let file = DiscreteModel::load(args.file.as_ref(), args.format)
        .wrap_err(eyre!("Failed to load model"))?;

    eprintln!(
        "Loaded model with {} points ({} segments)",
        file.segments
            .iter()
            .map(|s| s.dimensionless.r_coord.len())
            .sum::<usize>(),
        file.segments.len()
    );

    let model = LinearInterpolator::new(&file);
    let upper = args
        .frequency_units
        .convert_to_natural(args.upper, &model.dimensions())?;
    let lower = args
        .frequency_units
        .convert_to_natural(args.lower, &model.dimensions())?;

    let output = hdf5::File::create(args.output)?;
    for diff in [
        DifferenceSchemes::Colloc2,
        DifferenceSchemes::Colloc4,
        DifferenceSchemes::Colloc6,
        DifferenceSchemes::Colloc8,
    ] {
        let determinant = ErasedSolver::new(
            &model,
            args.ell,
            args.m,
            diff,
            &file
                .segments
                .iter()
                .map(|x| {
                    x.dimensionless
                        .r_coord
                        .windows(2)
                        .flat_map(|a| linspace(a[0], a[1], args.resample + 1).take(args.resample))
                        .chain([*x.dimensionless.r_coord.last().unwrap()].into_iter())
                        .collect_vec()
                })
                .collect_vec()
                .iter()
                .map(AsRef::as_ref)
                .collect_vec(),
        );

        let points = if args.inverse {
            &mut rev_linspace(lower, upper, args.steps) as &mut (dyn Iterator<Item = f64> + Send)
        } else {
            &mut linspace(lower, upper, args.steps) as &mut (dyn Iterator<Item = f64> + Send)
        };

        let results = points.map(|x| determinant.det(x)).collect_vec();

        output
            .new_dataset_builder()
            .with_data(&results)
            .create(format!("{:?}", diff).as_ref())?;
    }

    Ok(())
}

fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
    (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
}

fn rev_linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
    linspace(1. / lower, 1. / upper, n).map(|x| 1. / x)
}
