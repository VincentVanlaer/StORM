#![allow(clippy::too_many_arguments)]
use clap::{Parser, Subcommand};
use color_eyre::Result;
use color_eyre::eyre::{Context, ContextCompat, OptionExt, Report, eyre};
use core::f64;
use hdf5::H5Type;
use itertools::Itertools;
use nalgebra::ComplexField;
use ndarray::aview0;
use nshare::AsNdarray2;
use std::io::{self, IsTerminal};
use std::process::ExitCode;
use storm::dynamic_interface::{DifferenceSchemes, MultipleShooting};
use storm::postprocessing::{
    ModeCoupling, ModeToPerturb, PerturbedMetric, Rotating1DPostprocessing, perturb_deformed,
    perturb_structure,
};

use storm::bracket::{BracketResult, Precision};
use storm::model::StellarModel;
use storm::system::adiabatic::{GridScale, Rotating1D};

struct Solution {
    bracket: BracketResult,
    eigenvector: Vec<f64>,
    ell: u64,
    m: i64,
}

fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
    (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
}

fn rev_linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
    linspace(1. / lower, 1. / upper, n).map(|x| 1. / x)
}

#[derive(Debug, H5Type, PartialEq, Clone, Copy)]
#[repr(C)]
struct H5Complex {
    pub re: f64,
    pub im: f64,
}

fn main() -> ExitCode {
    let mut state = StormState::default();
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    let interactive_input = io::stdin().is_terminal();
    color_eyre::install().unwrap();

    loop {
        let readline = rl.readline("[storm] > ");
        let command = match readline {
            Ok(line) => {
                let _ = rl.add_history_entry(&line);
                line
            }
            Err(rustyline::error::ReadlineError::Interrupted) => {
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("Error: {:?}", err);
                break;
            }
        };

        let command = command.trim().to_owned();

        if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
            println!("[storm] > {}", command);
        }

        let command = match parse_command(&command) {
            Ok(x) => x,
            Err(error) => {
                match error {
                    ParseCommandError::QuoteError => eprintln!("Invalid quotation"),
                    ParseCommandError::ClapError(error) => error.print().unwrap(),
                }

                // In an interactive use case, you can retry. This is not the case for a piped input.
                if interactive_input {
                    continue;
                }

                return 2.into();
            }
        };

        let Some(command) = command else {
            continue;
        };

        if let Err(error) = command.run_command(&mut state) {
            eprintln!("{error:?}");

            if !interactive_input {
                return 1.into();
            }
        }
    }

    0.into()
}

fn parse_command(command: &str) -> Result<Option<StormCommands>, ParseCommandError> {
    if command.is_empty() {
        return Ok(None);
    }

    let Some(args) = shlex::split(command) else {
        return Err(ParseCommandError::QuoteError);
    };

    StormCli::try_parse_from(args)
        .map_err(ParseCommandError::ClapError)
        .map(|cli| Some(cli.command))
}

#[derive(Debug, Parser)]
#[command(multicall = true)]
struct StormCli {
    #[command(subcommand)]
    command: StormCommands,
}

#[derive(Debug)]
enum ParseCommandError {
    QuoteError,
    ClapError(clap::Error),
}

#[derive(Debug, Subcommand)]
enum StormCommands {
    /// Use a stellar model as input
    Input { file: String },
    /// Replace the rotation profile of a model
    SetRotationOverlay {
        /// HDF5 file containing the rotation profile. The structure should be the same as the
        /// normal input model
        file: String,
    },
    /// Set the rotation profile to a constant
    SetRotationConstant {
        /// Angular rotation frequency [rad/s]
        value: f64,
    },
    /// Perform a frequency scan
    Scan {
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
        /// Whether to do inverse steps or linear steps between lower and upper
        #[arg(long)]
        inverse: bool,
        /// Relative precision required
        precision: f64,
    },
    /// Compute the P2 deformation of the stellar model
    Deform {
        /// Angular rotation frequency [rad/s]
        ///
        /// This parameter should match the rotation frequency of the model.
        rotation: f64,
    },
    /// Compute derived properties from the eigenfunctions
    ///
    /// This includes the radial and horizontal displacements, the density and pressure
    /// perturbations, mode identification, ...
    PostProcess {},
    /// Perturb the mode frequencies and eigenfunctions to match the deformed star
    PerturbDeformed {
        /// Azimuthal order to do the perturbations for.
        /// This will filter the modes than have been obtain with scan to only those that have the
        /// same azimuthal order as selected here
        #[arg(allow_negative_numbers = true)]
        m: i64,
    },
    /// Write the results to an HDF5 file
    Output {
        /// The file to write the data to
        file: String,
    },
}

impl StormCommands {
    fn run_command(self, state: &mut StormState) -> Result<(), Report> {
        match self {
            Self::Input { file } => state.input(&file),
            Self::SetRotationOverlay { file } => state.set_rotation_overlay(&file),
            Self::SetRotationConstant { value } => state.set_rotation_constant(value),
            Self::Scan {
                ell,
                m,
                lower,
                upper,
                steps,
                inverse,
                precision,
            } => state.scan(ell, m, lower, upper, steps, inverse, precision),
            Self::Deform { rotation } => state.deform(rotation),
            Self::PostProcess {} => state.post_process(),
            Self::PerturbDeformed { m } => state.perturb_deformed(m),
            Self::Output { file } => state.output(file),
        }
    }
}

#[derive(Default)]
struct StormState {
    input: Option<StellarModel>,
    solutions: Vec<Solution>,
    perturbed_structure: Option<PerturbedMetric>,
    postprocessing: Option<Vec<Rotating1DPostprocessing>>,
    perturbed_frequencies: Vec<ModeCoupling>,
}

impl StormState {
    fn input(&mut self, file: &str) -> Result<(), Report> {
        let file = StellarModel::from_gsm(file).wrap_err(eyre!("Failed to load model"))?;

        eprintln!("Loaded model with {} points", file.r_coord.len());

        self.input = Some(file);

        Ok(())
    }

    fn set_rotation_overlay(&mut self, file: &str) -> Result<(), Report> {
        let input = self.input.as_mut().ok_or_eyre(
            "Input was not set. Please run `input` before setting the rotation profile.",
        )?;

        input
            .overlay_rot(file)
            .wrap_err(eyre!("Failed to set rotation profile"))?;

        Ok(())
    }

    fn set_rotation_constant(&mut self, value: f64) -> Result<(), Report> {
        let input = self.input.as_mut().ok_or_eyre(
            "Input was not set. Please run `input` before setting the rotation profile.",
        )?;

        input.rot.fill(value);

        Ok(())
    }

    fn scan(
        &mut self,
        ell: u64,
        m: i64,
        lower: f64,
        upper: f64,
        steps: usize,
        inverse: bool,
        precision: f64,
    ) -> Result<(), Report> {
        let input = self
            .input
            .as_mut()
            .ok_or_eyre("Input was not set. Please run `input` before running a scan.")?;

        let system = Rotating1D::from_model(input, ell, m);
        let determinant =
            MultipleShooting::new(&system, DifferenceSchemes::Magnus6, &GridScale { scale: 0 });
        let points = if inverse {
            &mut rev_linspace(lower, upper, steps) as &mut dyn Iterator<Item = f64>
        } else {
            &mut linspace(lower, upper, steps) as &mut dyn Iterator<Item = f64>
        };

        self.solutions.extend(
            determinant
                .scan_and_optimize(points, Precision::Relative(precision))
                .map(|res| Solution {
                    eigenvector: determinant.eigenvector(res.root),
                    bracket: res,
                    ell,
                    m,
                }),
        );

        Ok(())
    }

    fn deform(&mut self, rotation: f64) -> Result<(), Report> {
        let input = self
            .input
            .as_mut()
            .ok_or_eyre("Input was not set. Please run `input` before running deform.")?;

        self.perturbed_structure = Some(perturb_structure(input, rotation));

        Ok(())
    }

    fn post_process(&mut self) -> Result<(), Report> {
        let input = self.input.as_mut().ok_or_eyre(
            "Input was not set. Please run `input` and `scan` before running `post-process`.",
        )?;

        self.postprocessing = Some(
            self.solutions
                .iter()
                .map(|sol| -> Result<Rotating1DPostprocessing> {
                    Ok(Rotating1DPostprocessing::new(
                        sol.bracket.root,
                        &sol.eigenvector,
                        sol.ell,
                        sol.m,
                        input,
                    ))
                })
                .try_collect()?,
        );

        Ok(())
    }

    fn perturb_deformed(&mut self, m: i64) -> Result<(), Report> {
        let input = self.input.as_mut().ok_or_eyre(
            "Input was not set. Please run `input`, `deform`, and `scan` before running `perturb-deformed`.",
        )?;

        let perturbed_structure = self.perturbed_structure.as_ref().wrap_err(
            "Deformed structure was not computed. Please run `deform` before `perturb-deformed`",
        )?;

        let postprocessing = self.postprocessing.as_mut().ok_or_eyre(
            "Post processing was not ran. Please run `post-process` before running `perturb-deformed`.",
        )?;

        for mode_coupling in &self.perturbed_frequencies {
            if mode_coupling.m == m {
                return Err(eyre!("Already computed perturbation for m = {m}"));
            }
        }

        self.perturbed_frequencies.push(perturb_deformed(
            input,
            self.solutions
                .iter()
                .zip(postprocessing.iter())
                .filter(|(sol, _)| sol.m == m)
                .map(|(sol, post)| ModeToPerturb {
                    ell: sol.ell,
                    freq: sol.bracket.root,
                    post_processing: post,
                })
                .collect_vec()
                .as_ref(),
            m,
            perturbed_structure,
        ));

        Ok(())
    }

    fn output(&mut self, file: String) -> Result<(), Report> {
        let output = hdf5::File::create(file)?;

        let input = self.input.as_mut().ok_or_eyre(
            "Input was not set. Please run `input`, `scan`, and potentially `post-process` before running `output`.",
        )?;

        output
            .new_dataset_builder()
            .with_data(&input.r_coord)
            .create("r")?;

        if let Some(ref perturbed_structure) = self.perturbed_structure {
            output
                .new_dataset_builder()
                .with_data(&perturbed_structure.beta)
                .create("beta")?;
            output
                .new_dataset_builder()
                .with_data(&perturbed_structure.dbeta)
                .create("dbeta")?;
            output
                .new_dataset_builder()
                .with_data(&perturbed_structure.ddbeta)
                .create("ddbeta")?;
            output
                .new_attr_builder()
                .with_data(aview0(&perturbed_structure.rot))
                .create("rot")?;
        }

        let solution_group = output.create_group("solutions")?;

        for (i, solution) in self.solutions.iter().enumerate() {
            let group = solution_group.create_group(format!("{i:05}").as_str())?;

            group
                .new_attr_builder()
                .with_data(aview0(&solution.bracket.root))
                .create("freq")?;

            group
                .new_attr_builder()
                .with_data(aview0(&solution.ell))
                .create("ell")?;

            group
                .new_attr_builder()
                .with_data(aview0(&solution.m))
                .create("m")?;

            if let Some(ref postprocessing) = self.postprocessing {
                let postprocessing = &postprocessing[i];
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
                    .with_data(&postprocessing.xi_r)
                    .create("xi_r")?;
                group
                    .new_dataset_builder()
                    .with_data(&postprocessing.xi_h)
                    .create("xi_h")?;
            }
        }

        if !self.perturbed_frequencies.is_empty() {
            let group = output.create_group("perturbations")?;

            for mode_coupling in &self.perturbed_frequencies {
                let group = group.create_group(&format!("{}", mode_coupling.m))?;
                group
                    .new_dataset_builder()
                    .with_data(
                        &mode_coupling
                            .freqs
                            .iter()
                            .map(|x| H5Complex {
                                re: x.real(),
                                im: x.imaginary(),
                            })
                            .collect_vec(),
                    )
                    .create("frequencies")?;
                group
                    .new_dataset_builder()
                    .with_data(
                        &mode_coupling
                            .coupling
                            .map(|x| H5Complex {
                                re: x.real(),
                                im: x.imaginary(),
                            })
                            .as_ndarray2()
                            .as_standard_layout(),
                    )
                    .create("eigenvectors")?;

                group
                    .new_dataset_builder()
                    .with_data(&mode_coupling.d.as_ndarray2().as_standard_layout())
                    .create("d")?;
                group
                    .new_dataset_builder()
                    .with_data(&mode_coupling.r.as_ndarray2().as_standard_layout())
                    .create("r")?;
                group
                    .new_dataset_builder()
                    .with_data(&mode_coupling.l.as_ndarray2().as_standard_layout())
                    .create("l")?;
            }
        }

        self.solutions.clear();
        self.postprocessing = None;
        self.perturbed_structure = None;
        self.perturbed_frequencies.clear();

        Ok(())
    }
}
