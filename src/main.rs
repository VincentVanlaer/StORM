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
        println!("{:.20} {}", solution.root, solution.evals);

        let group = output.create_group(format!("{i:05}").as_str())?;

        group
            .new_attr_builder()
            .with_data(aview0(&solution.root))
            .create("freq")?;

        if args.eigenfunctions {
            let eigenvector = determinant.eigenvector(solution.root);

            let (chunks, _) = eigenvector.as_chunks::<4>();

            let mut vec1 = vec![0.0; chunks.len()];
            let mut vec2 = vec![0.0; chunks.len()];
            let mut vec3 = vec![0.0; chunks.len()];
            let mut vec4 = vec![0.0; chunks.len()];

            for (i, chunk) in chunks.iter().enumerate() {
                vec1[i] = chunk[0];
                vec2[i] = chunk[1];
                vec3[i] = chunk[2];
                vec4[i] = chunk[3];
            }
            group
                .new_dataset_builder()
                .with_data(vec1.as_slice())
                .create("y1")?;
            group
                .new_dataset_builder()
                .with_data(vec2.as_slice())
                .create("y2")?;
            group
                .new_dataset_builder()
                .with_data(vec3.as_slice())
                .create("y3")?;
            group
                .new_dataset_builder()
                .with_data(vec4.as_slice())
                .create("y4")?;
        }
    }

    Ok(())
}
