#![feature(slice_as_chunks)]
#![feature(never_type)]
#![feature(generic_const_exprs)]
#![expect(incomplete_features)]

use clap::Parser;
use color_eyre::eyre::{eyre, Context};
use color_eyre::Result;
use ndarray::aview0;

use storm::bracket::Precision;
use storm::dynamic_interface::{DifferenceSchemes, MultipleShooting};
use storm::model::StellarModel;
use storm::postprocessing::Rotating1DPostprocessing;
use storm::system::adiabatic::{GridScale, Rotating1D};

#[derive(Parser)]
#[command()]
struct Main {
    #[arg(long)]
    input: String,
    #[arg(long)]
    overlay_rot: Option<String>,
    #[arg(long)]
    output: String,
    #[arg(long)]
    lower: f64,
    #[arg(long)]
    upper: f64,
    #[arg(long)]
    n_steps: usize,
    #[arg(long)]
    degree: u64,
    #[arg(long)]
    order: i64,
    #[arg(long)]
    difference_scheme: DifferenceSchemes,
    #[arg(long)]
    eigenfunctions: bool,
}

fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
    (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
}

fn main() -> Result<()> {
    color_eyre::install()?;

    let args = Main::parse();
    let mut model = StellarModel::from_gsm(&args.input)
        .wrap_err(eyre!("Could not read model {}", &args.input))?;

    if let Some(overlay) = args.overlay_rot {
        model
            .overlay_rot(&overlay)
            .wrap_err(eyre!("Could not read overlay {overlay}"))?;
    }

    let system = Rotating1D::from_model(&model, args.degree, args.order);
    let grid = &GridScale { scale: 0 };
    let determinant = MultipleShooting::new(&system, args.difference_scheme, grid);

    let solutions: Vec<_> = determinant
        .scan_and_optimize(
            linspace(args.lower, args.upper, args.n_steps),
            Precision::Relative(0.),
        )
        .collect();

    let output = hdf5::File::create(args.output)?;

    for (i, solution) in solutions.iter().enumerate() {
        let group = output.create_group(format!("{i}").as_str())?;

        group
            .new_attr_builder()
            .with_data(aview0(&solution.root))
            .create("freq")?;

        if args.eigenfunctions {
            let eigenvector = determinant.eigenvector(solution.root);

            let postprocessing = Rotating1DPostprocessing::new(
                solution.root,
                &eigenvector,
                args.degree,
                args.order,
                &model,
            );

            println!(
                "{:>3}: {:>23.20} {:>2} {:>6.3} {:>6.3} {:>+7.3} {}",
                i + 1,
                solution.root,
                solution.evals,
                postprocessing.clockwise_winding,
                postprocessing.counter_clockwise_winding,
                postprocessing.clockwise_winding - postprocessing.counter_clockwise_winding,
                (postprocessing.clockwise_winding - postprocessing.counter_clockwise_winding)
                    .round(),
            );

            group
                .new_dataset_builder()
                .with_data(&postprocessing.x)
                .create("x")?;

            group
                .new_dataset_builder()
                .with_data(&postprocessing.y1)
                .create("y1")?;
            group
                .new_dataset_builder()
                .with_data(&postprocessing.y2)
                .create("y2")?;
            group
                .new_dataset_builder()
                .with_data(&postprocessing.y3)
                .create("y3")?;
            group
                .new_dataset_builder()
                .with_data(&postprocessing.y4)
                .create("y4")?;
            group
                .new_dataset_builder()
                .with_data(&postprocessing.xi_h)
                .create("xi_h")?;
            group
                .new_dataset_builder()
                .with_data(&postprocessing.xi_r)
                .create("xi_r")?;

            group
                .new_attr_builder()
                .with_data(aview0(
                    &((postprocessing.clockwise_winding - postprocessing.counter_clockwise_winding)
                        .round() as i64),
                ))
                .create("n_pg")?;

            group
                .new_attr_builder()
                .with_data(aview0(&args.degree))
                .create("degree")?;

            group
                .new_attr_builder()
                .with_data(aview0(&args.order))
                .create("order")?;
        } else {
            println!("{:.20} {}", solution.root, solution.evals);
        }
    }

    Ok(())
}
