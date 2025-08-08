#![allow(clippy::too_many_arguments)]
use clap::Parser;
use color_eyre::Result;
use color_eyre::eyre::{Context, ContextCompat, OptionExt, Report, eyre};
use hdf5::H5Type;
use itertools::Itertools;
use nalgebra::ComplexField;
use ndarray::aview0;
use nshare::AsNdarray2;
use std::io::{self, IsTerminal};
use std::process::ExitCode;
use std::usize;
use storm::bracket::{BracketResult, Precision};
use storm::dynamic_interface::{DifferenceSchemes, ErasedSolver};
use storm::model::interpolate::LinearInterpolator;
use storm::model::polytrope::{Polytrope0, construct_polytrope};
use storm::model::{ContinuousModel, DimensionedProperties, DiscreteModel};
use storm::perturbed::{ModeCoupling, ModeToPerturb, perturb_deformed, perturb_structure};
use storm::postprocessing::Rotating1DPostprocessing;
use storm::system::adiabatic::Rotating1D;

fn main() -> ExitCode {
    let mut state = StormState::default();
    let mut rl = rustyline::DefaultEditor::new().unwrap();
    let interactive_input = io::stdin().is_terminal();
    color_eyre::install().unwrap();

    let mut arg = std::env::args().nth(1).map(|f| {
        std::fs::read_to_string(f)
            .unwrap()
            .lines()
            .map(String::from)
            .collect_vec()
            .into_iter()
    });

    loop {
        let command = if let Some(ref mut arg) = arg {
            let Some(command) = arg.next() else { break };

            command
        } else {
            let readline = rl.readline("[storm] > ");
            match readline {
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
            }
        };

        let command = command.trim().to_owned();

        if !io::stdout().is_terminal() || !io::stdin().is_terminal() {
            println!("[storm] > {}", command);
        }

        // Inline comments are handled by clap. If there is only a comment on the line, then clap
        // will show the help text (as there is no command), which is not wat we want.
        if command.starts_with("#") {
            continue;
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

    StormCommands::try_parse_from(args)
        .map_err(ParseCommandError::ClapError)
        .map(Some)
}

#[derive(Debug)]
enum ParseCommandError {
    QuoteError,
    ClapError(clap::Error),
}

#[derive(Debug, Parser)]
#[command(multicall = true)]
enum StormCommands {
    /// Load a stellar model
    Input {
        /// Location of the stellar model. The stellar model should be an HDF5 GYRE model file.
        file: String,
        /// How many times should each datapoint of the input model be subdivided.
        #[arg(long, default_value = "1")]
        resample: usize,
    },
    /// Load a polytrope model
    InputPoly {
        /// Polytropic index
        index: f64,
        /// Step size in intergation of the polytrope model
        dx: f64,
        /// First adiabatic exponent
        #[arg(long, default_value = "1.66666666666666")]
        gamma1: f64,
        /// How many times should each datapoint of the input model be subdivided.
        #[arg(long, default_value = "1")]
        resample: usize,
    },
    /// Load the analytical index-zero polytrope. This is primarily used for convergence testing
    /// and has some restrictions (no rotation).
    InputPoly0 {
        /// Number of sample points
        npoints: usize,
        /// First adiabatic exponent
        #[arg(long, default_value = "1.66666666666666")]
        gamma1: f64,
    },
    /// Replace the rotation profile of a model
    SetRotationOverlay {
        /// HDF5 file containing the rotation profile. The structure should be the same as the
        /// normal input model
        file: String,
    },
    /// Set the rotation profile to a constant
    SetRotationConstant {
        /// Angular rotation frequency
        value: f64,
        /// Units of value
        #[arg(long, default_value = "dynamical")]
        frequency_units: FrequencyUnits,
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
        /// Difference scheme
        #[arg(long, default_value = "colloc4")]
        difference_scheme: DifferenceSchemes,
        /// Units of lower and upper
        #[arg(long, default_value = "dynamical")]
        frequency_units: FrequencyUnits,
    },
    /// Compute the P2 deformation of the stellar model
    Deform {
        /// Rotation frequency
        ///
        /// This parameter should match the rotation frequency of the model.
        rotation: f64,
        /// Units of rotation
        #[arg(long, default_value = "dynamical")]
        frequency_units: FrequencyUnits,
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
    /// Write the results to an HDF5 file. Unless `--keep-data` is passed, this will clear all data except for the input model.
    Output {
        /// The file to write the data to
        file: String,
        /// All frequencies will be outputted in these units. This includes the mode frequencies,
        /// but also the rotation frequency
        #[arg(long, default_value = "dynamical")]
        frequency_units: FrequencyUnits,
        /// Mode properties to include in the output
        ///
        /// Some of these properties require post-processing to be available. The properties will
        /// be available as datasets in the root group, unless specified otherwise.
        #[arg(long, value_delimiter = ',')]
        properties: Vec<ModePropertyFlags>,
        /// Profiles to include in the output. Requires post-processing.
        ///
        /// These profiles can be found in sub groups of the `mode-profiles` group. The name of the
        /// group is the index in the main solution arrays (e.g. `frequency`). Values can be separated
        /// by commas.
        ///
        /// All profiles are normalized.
        #[arg(long, value_delimiter = ',')]
        profiles: Vec<ProfileFlags>,
        /// Model properties.
        ///
        /// These properties can be found in the `model` group. Values can be separated by commas.
        #[arg(long, value_delimiter = ',')]
        model_properties: Vec<ModelPropertyFlags>,
        /// Do not delete current computation results
        #[arg(long)]
        keep_data: bool,
    },
    /// Remove all computations thus far. Needed to load a new input model after doing computations
    /// without saving results.
    Clear,
    #[command(hide(true))]
    GenerateMarkdown,
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

#[derive(clap::ValueEnum, Debug, Clone, Copy, PartialEq)]
enum ProfileFlags {
    /// Radial coordinate of the points in the other datasets. Since this will be the same for
    /// all the modes, it can be found in the root group.
    RadialCoordinate,
    /// Dimensionless perturbation
    Y1,
    /// Dimensionless perturbation
    Y2,
    /// Dimensionless perturbation
    Y3,
    /// Dimensionless perturbation
    Y4,
    /// Radial displacement
    #[clap(name = "xi_r")]
    XiR,
    /// Horizontal displacement
    #[clap(name = "xi_h")]
    XiH,
    /// Toroidal displacement (l + 1)
    #[clap(name = "xi_tp")]
    XiTp,
    /// Toroidal displacement (l - 1)
    #[clap(name = "xi_tn")]
    XiTn,
    /// Pressure perturbation
    Pressure,
    /// Density perturbation
    Density,
    /// Gravity potential perturbation
    GravityPotential,
    /// Gravity acceleration perturbation
    GravityAcceleration,
    /// Divergence of the displacement
    Divergence,
    /// Location of the radial nodes of the mode
    Nodes,
}

#[derive(Default)]
struct Profiles {
    radial_coordinate: bool,
    y1: bool,
    y2: bool,
    y3: bool,
    y4: bool,
    xi_r: bool,
    xi_h: bool,
    xi_tn: bool,
    xi_tp: bool,
    pressure: bool,
    density: bool,
    gravity_potential: bool,
    gravity_acceleration: bool,
    divergence: bool,
    nodes: bool,
}

impl Profiles {
    fn needs_post_processing(&self) -> bool {
        self.y1
            || self.y2
            || self.y3
            || self.y4
            || self.xi_r
            || self.xi_h
            || self.xi_tp
            || self.xi_tn
            || self.pressure
            || self.density
            || self.gravity_potential
            || self.gravity_acceleration
            || self.divergence
            || self.nodes
    }
}

impl From<Vec<ProfileFlags>> for Profiles {
    fn from(value: Vec<ProfileFlags>) -> Self {
        let mut prof = Self::default();

        for val in value {
            match val {
                ProfileFlags::RadialCoordinate => prof.radial_coordinate = true,
                ProfileFlags::Y1 => prof.y1 = true,
                ProfileFlags::Y2 => prof.y2 = true,
                ProfileFlags::Y3 => prof.y3 = true,
                ProfileFlags::Y4 => prof.y4 = true,
                ProfileFlags::XiR => prof.xi_r = true,
                ProfileFlags::XiH => prof.xi_h = true,
                ProfileFlags::Pressure => prof.pressure = true,
                ProfileFlags::Density => prof.density = true,
                ProfileFlags::GravityPotential => prof.gravity_potential = true,
                ProfileFlags::GravityAcceleration => prof.gravity_acceleration = true,
                ProfileFlags::Divergence => prof.divergence = true,
                ProfileFlags::XiTp => prof.xi_tp = true,
                ProfileFlags::XiTn => prof.xi_tn = true,
                ProfileFlags::Nodes => prof.nodes = true,
            }
        }

        prof
    }
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum ModePropertyFlags {
    /// Mode frequency in units given by `frequency-units`
    Frequency,
    /// Spherical degree of the mode
    Degree,
    /// Azimuthal order of the mode
    AzimuthalOrder,
    /// Radial order of the mode
    RadialOrder,
    /// Perturbed frequencies, units are given by `frequency-units`. It is stored in a subgroup of
    /// the `deformation` group. The name of that subgroup is given by the azimuthal order selected
    /// for the perturbative calculations.
    DeformedFrequency,
    /// Eigenvectors for the perturbed system of equations. These can be used to construct the
    /// perturbed eigenfunctions from the actual eigenfunctions. Each solution of the perturbed
    /// system is a column of this matrix, while the rows map to one of the eigenfunctions. It is
    /// stored as `eigenvector` in the same group as the `deformed-frequency` option.
    DeformedEigenvector,
    /// Coupling matrices `L`, `D`, and `R` for the deformation perturbation. They can be found in
    /// the previously mentioned subgroup as `l`, `d`, and `r` respectively.
    CouplingMatrix,
}

#[derive(Default)]
struct ModeProperties {
    frequency: bool,
    degree: bool,
    radial_order: bool,
    azimuthal_order: bool,
    deformed_frequency: bool,
    deformed_eigenvector: bool,
    coupling_matrix: bool,
}

impl ModeProperties {
    fn needs_post_processing(&self) -> bool {
        self.radial_order
    }

    fn needs_deformation(&self) -> bool {
        self.deformed_frequency || self.deformed_eigenvector
    }
}

impl From<Vec<ModePropertyFlags>> for ModeProperties {
    fn from(value: Vec<ModePropertyFlags>) -> Self {
        let mut prop = ModeProperties::default();

        for flag in value {
            match flag {
                ModePropertyFlags::Frequency => prop.frequency = true,
                ModePropertyFlags::Degree => prop.degree = true,
                ModePropertyFlags::AzimuthalOrder => prop.azimuthal_order = true,
                ModePropertyFlags::DeformedFrequency => prop.deformed_frequency = true,
                ModePropertyFlags::DeformedEigenvector => prop.deformed_eigenvector = true,
                ModePropertyFlags::CouplingMatrix => prop.coupling_matrix = true,
                ModePropertyFlags::RadialOrder => prop.radial_order = true,
            }
        }

        prop
    }
}

#[derive(clap::ValueEnum, Debug, Clone, Copy)]
enum ModelPropertyFlags {
    /// Dynamical frequency of the star (sqrt(GM/R^3)) [1/s]
    ///
    /// Contrary to other outputted frequencies, this is always given in Hz. Saved as an attribute,
    /// not a dataset.
    DynamicalFrequency,
    /// The P2 deformation of the stellar structure. This quantity is unitless. This is only
    /// available if the deformation command has been called.
    DeformationAlpha,
    /// The logarithmic derivative of alpha by a. It is stored in the `model` group. This is only
    /// available if the deformation command has been called.
    #[clap(name = "deformation-dalpha")]
    DeformationDAlpha,
    /// The second logarithmic derivative of alpha by a. It is stored in the `model` group. This is
    /// only available if the deformation command has been called.
    #[clap(name = "deformation-ddalpha")]
    DeformationDDAlpha,
    /// The P2 deformation of the stellar structure. This quantity is unitless. This is only
    /// available if the deformation command has been called.
    DeformationBeta,
    /// The logarithmic derivative of beta by a. It is stored in the `model` group. This is only
    /// available if the deformation command has been called.
    #[clap(name = "deformation-dbeta")]
    DeformationDBeta,
    /// The second logarithmic derivative of beta by a. It is stored in the `model` group. This is
    /// only available if the deformation command has been called.
    #[clap(name = "deformation-ddbeta")]
    DeformationDDBeta,
    /// Rotation frequency used in for the deformation calculations. This might be replaced by just
    /// the model rotation frequency if shellular differential rotation is supported in the
    /// deformation calculations. Saved as an attribute, not a dataset.
    DeformationRotationFrequency,
}

#[derive(Default)]
struct ModelProperties {
    dynamical_frequency: bool,
    deformation_alpha: bool,
    deformation_dalpha: bool,
    deformation_ddalpha: bool,
    deformation_beta: bool,
    deformation_dbeta: bool,
    deformation_ddbeta: bool,
    deformation_rotation_frequency: bool,
}

impl ModelProperties {
    fn needs_deformation(&self) -> bool {
        self.deformation_alpha
            || self.deformation_dalpha
            || self.deformation_ddalpha
            || self.deformation_beta
            || self.deformation_dbeta
            || self.deformation_ddbeta
            || self.deformation_rotation_frequency
    }
}

impl From<Vec<ModelPropertyFlags>> for ModelProperties {
    fn from(value: Vec<ModelPropertyFlags>) -> Self {
        let mut prop = Self::default();

        for val in value {
            match val {
                ModelPropertyFlags::DynamicalFrequency => prop.dynamical_frequency = true,
                ModelPropertyFlags::DeformationAlpha => prop.deformation_alpha = true,
                ModelPropertyFlags::DeformationDAlpha => prop.deformation_dalpha = true,
                ModelPropertyFlags::DeformationDDAlpha => prop.deformation_ddalpha = true,
                ModelPropertyFlags::DeformationBeta => prop.deformation_beta = true,
                ModelPropertyFlags::DeformationDBeta => prop.deformation_dbeta = true,
                ModelPropertyFlags::DeformationDDBeta => prop.deformation_ddbeta = true,
                ModelPropertyFlags::DeformationRotationFrequency => {
                    prop.deformation_rotation_frequency = true
                }
            }
        }

        prop
    }
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

    fn convert_from_natural(
        &self,
        freq: f64,
        model: &Option<DimensionedProperties>,
    ) -> Result<f64> {
        self.scale_factor(model).map(|s| freq * s)
    }
}

impl StormCommands {
    fn run_command(self, state: &mut StormState) -> Result<(), Report> {
        match self {
            Self::Input { file, resample } => state.input(&file, resample),
            Self::InputPoly {
                index,
                dx,
                gamma1,
                resample,
            } => state.input_poly(index, dx, gamma1, resample),
            Self::InputPoly0 { gamma1, npoints } => state.input_poly0(gamma1, npoints),
            Self::SetRotationOverlay { file } => state.set_rotation_overlay(&file),
            Self::SetRotationConstant {
                value,
                frequency_units,
            } => state.set_rotation_constant(value, frequency_units),
            Self::Scan {
                ell,
                m,
                lower,
                upper,
                steps,
                inverse,
                precision,
                difference_scheme,
                frequency_units,
            } => state.scan(
                ell,
                m,
                lower,
                upper,
                steps,
                inverse,
                precision,
                difference_scheme,
                frequency_units,
            ),
            Self::Deform {
                rotation,
                frequency_units,
            } => state.deform(rotation, frequency_units),
            Self::PostProcess {} => state.post_process(),
            Self::PerturbDeformed { m } => state.perturb_deformed(m),
            Self::Output {
                file,
                frequency_units,
                profiles,
                properties,
                model_properties,
                keep_data,
            } => state.output(
                file,
                frequency_units,
                properties.into(),
                profiles.into(),
                model_properties.into(),
                keep_data,
            ),
            Self::Clear => state.clear(),
            Self::GenerateMarkdown => Ok(clap_markdown::print_help_markdown::<StormCommands>()),
        }
    }
}

#[derive(Default)]
struct StormState {
    input: Option<Model>,
    solutions: Vec<Solution>,
    postprocessing: Option<Vec<Rotating1DPostprocessing>>,
    perturbed_frequencies: Vec<ModeCoupling>,
}

enum Model {
    Discrete(DiscreteModel, usize),
    Continuous(Box<dyn ContinuousModel>, usize),
}

impl Model {
    fn grid(&self) -> Box<[f64]> {
        match self {
            Model::Discrete(model, scale) => model
                .dimensionless
                .r_coord
                .windows(2)
                .flat_map(|a| linspace(a[0], a[1], scale + 1).take(*scale))
                .chain([*model.dimensionless.r_coord.last().unwrap()].into_iter())
                .collect_vec()
                .into(),
            Model::Continuous(model, scale) => {
                let lower = model.inner();
                let upper = model.outer();

                linspace(lower, upper, *scale).collect_vec().into()
            }
        }
    }

    fn as_continuous(&self) -> Box<dyn ContinuousModel + '_> {
        match self {
            Model::Discrete(discrete_model, _) => Box::new(LinearInterpolator::new(discrete_model)),
            Model::Continuous(continuous_model, _) => Box::new(continuous_model),
        }
    }
}

impl StormState {
    fn input(&mut self, file: &str, resample: usize) -> Result<(), Report> {
        if !self.solutions.is_empty() {
            return Err(eyre!(
                "Changing input models with already computed solutions is not supported. Either first write out the results with the `output` command or remove all results using `clear`."
            ));
        }

        let file = DiscreteModel::from_gsm(file).wrap_err(eyre!("Failed to load model"))?;

        eprintln!(
            "Loaded model with {} points",
            file.dimensionless.r_coord.len()
        );

        self.input = Some(Model::Discrete(file, resample));

        Ok(())
    }

    fn input_poly(
        &mut self,
        index: f64,
        dx: f64,
        gamma1: f64,
        resample: usize,
    ) -> Result<(), Report> {
        if !self.solutions.is_empty() {
            return Err(eyre!(
                "Changing input models with already computed solutions is not supported. Either first write out the results with the `output` command or remove all results using `clear`."
            ));
        }

        if index < 0. || index >= 5. {
            return Err(eyre!(
                "Invalid polytrope index {index}. The polytrope index n should be smaller than 5 and positive."
            ));
        }

        let model = construct_polytrope(index, gamma1, dx);

        eprintln!(
            "Loaded model with {} points",
            model.dimensionless.r_coord.len()
        );

        self.input = Some(Model::Discrete(model, resample));

        Ok(())
    }

    fn input_poly0(&mut self, gamma1: f64, npoints: usize) -> Result<(), Report> {
        if !self.solutions.is_empty() {
            return Err(eyre!(
                "Changing input models with already computed solutions is not supported. Either first write out the results with the `output` command or remove all results using `clear`."
            ));
        }

        self.input = Some(Model::Continuous(Box::new(Polytrope0 { gamma1 }), npoints));

        Ok(())
    }

    fn set_rotation_overlay(&mut self, file: &str) -> Result<(), Report> {
        let input = self.input.as_mut().ok_or_eyre(
            "Input was not set. Please run `input` before setting the rotation profile.",
        )?;

        let Model::Discrete(model, _) = input else {
            return Err(eyre!(
                "Setting the rotation profile for analytical polytropes is not supported"
            ));
        };

        model
            .overlay_rot(file)
            .wrap_err(eyre!("Failed to set rotation profile"))?;

        Ok(())
    }

    fn set_rotation_constant(
        &mut self,
        value: f64,
        frequency_units: FrequencyUnits,
    ) -> Result<(), Report> {
        let input = self.input.as_mut().ok_or_eyre(
            "Input was not set. Please run `input` before setting the rotation profile.",
        )?;

        let Model::Discrete(model, _) = input else {
            return Err(eyre!(
                "Setting the rotation profile for analytical polytropes is not supported"
            ));
        };

        model
            .dimensionless
            .rot
            .fill(frequency_units.convert_to_natural(value, &model.scale)?);

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
        difference_scheme: DifferenceSchemes,
        frequency_units: FrequencyUnits,
    ) -> Result<(), Report> {
        let input = self
            .input
            .as_mut()
            .ok_or_eyre("Input was not set. Please run `input` before running a scan.")?;

        let system = Rotating1D::new(ell, m);
        let model = input.as_continuous();
        let upper = frequency_units.convert_to_natural(upper, &model.dimensions())?;
        let lower = frequency_units.convert_to_natural(lower, &model.dimensions())?;

        let determinant = ErasedSolver::new(&model, system, difference_scheme, &input.grid());
        let points = if inverse {
            &mut rev_linspace(lower, upper, steps) as &mut dyn Iterator<Item = f64>
        } else {
            &mut linspace(lower, upper, steps) as &mut dyn Iterator<Item = f64>
        };

        let solutions = determinant
            .scan_and_optimize(points, Precision::Relative(precision))
            .map(|res| Solution {
                eigenvector: determinant.eigenvector(res.root),
                bracket: res,
                ell,
                m,
            })
            .collect_vec();

        eprintln!("Found {} modes", solutions.len());

        self.solutions.extend(solutions);

        Ok(())
    }

    fn deform(&mut self, rotation: f64, frequency_units: FrequencyUnits) -> Result<(), Report> {
        let input = self
            .input
            .as_mut()
            .ok_or_eyre("Input was not set. Please run `input` before running deform.")?;

        let Model::Discrete(model, _) = input else {
            return Err(eyre!(
                "Setting the rotation profile for analytical polytropes is not supported"
            ));
        };
        let rotation = frequency_units.convert_to_natural(rotation, &model.scale)?;

        model.metric = Some(perturb_structure(model, rotation));

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
                        &input.as_continuous().eval(&input.grid()),
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

        let input = input.as_continuous().eval(&input.grid());

        let perturbed_structure = input.metric.as_ref().wrap_err(
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
            &input,
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

    fn output(
        &mut self,
        file: String,
        frequency_units: FrequencyUnits,
        properties: ModeProperties,
        profiles: Profiles,
        model_properties: ModelProperties,
        keep_data: bool,
    ) -> Result<(), Report> {
        let output = hdf5::File::create(file)?;

        let input = self.input.as_mut().ok_or_eyre(
            "Input was not set. Please run `input`, `scan`, and potentially `post-process` before running `output`.",
        )?;

        let input = input.as_continuous().eval(&input.grid());

        if model_properties.needs_deformation() && input.metric.is_none() {
            eprintln!("Deformation was requested as output, but was not computed");
        }

        if properties.needs_deformation() && self.perturbed_frequencies.is_empty() {
            eprintln!(
                "Perturbation from rotational deformation was requested as output, but was not computed"
            );
        }

        if (properties.needs_post_processing() || profiles.needs_post_processing())
            && self.postprocessing.is_none()
        {
            eprintln!("Post-processing was requested as output, but was not computed");
        }

        macro_rules! dataset {
            ($group: ident, $name: expr, $val: expr) => {
                $group.new_dataset_builder().with_data($val).create($name)
            };
        }

        macro_rules! attr {
            ($group: ident, $name: expr, $val: expr) => {
                $group.new_attr_builder().with_data($val).create($name)
            };
        }

        if profiles.radial_coordinate {
            dataset!(output, "radial-coordinate", &input.dimensionless.r_coord)?;
        }

        let model_group = output.create_group("model")?;

        if model_properties.dynamical_frequency {
            if let Some(scale) = input.scale {
                attr!(
                    model_group,
                    "dynamical-frequency",
                    aview0(&scale.freq_scale())
                )?;
            } else {
                eprintln!(
                    "Dynamical frequency was requested as output, but input model is dimensionless"
                );
            }
        }

        if let Some(ref perturbed_structure) = input.metric {
            if model_properties.deformation_alpha {
                dataset!(model_group, "deformation-alpha", &perturbed_structure.alpha)?;
            }

            if model_properties.deformation_dalpha {
                dataset!(
                    model_group,
                    "deformation-dalpha",
                    &perturbed_structure.dalpha
                )?;
            }

            if model_properties.deformation_ddalpha {
                dataset!(
                    model_group,
                    "deformation-ddalpha",
                    &perturbed_structure.ddalpha
                )?;
            }

            if model_properties.deformation_beta {
                dataset!(model_group, "deformation-beta", &perturbed_structure.beta)?;
            }

            if model_properties.deformation_dbeta {
                dataset!(model_group, "deformation-dbeta", &perturbed_structure.dbeta)?;
            }

            if model_properties.deformation_ddbeta {
                dataset!(
                    model_group,
                    "deformation-ddbeta",
                    &perturbed_structure.ddbeta
                )?;
            }

            if model_properties.deformation_rotation_frequency {
                let rot =
                    frequency_units.convert_from_natural(perturbed_structure.rot, &input.scale)?;
                attr!(model_group, "deformation-rotation-frequency", aview0(&rot))?;
            }
        }

        let solution_group = output.create_group("mode-profiles")?;

        let mut freq = Vec::new();
        let mut ell = Vec::new();
        let mut m = Vec::new();
        let mut radial_order = Vec::new();

        for (i, solution) in self.solutions.iter().enumerate() {
            let group = solution_group.create_group(format!("{i}").as_str())?;

            freq.push(frequency_units.convert_from_natural(solution.bracket.root, &input.scale)?);
            ell.push(solution.ell);
            m.push(solution.m);

            if let Some(ref postprocessing) = self.postprocessing {
                let postprocessing = &postprocessing[i];
                radial_order.push(postprocessing.radial_order);

                if profiles.y1 {
                    dataset!(group, "y1", &postprocessing.y1)?;
                }

                if profiles.y2 {
                    dataset!(group, "y2", &postprocessing.y2)?;
                }

                if profiles.y3 {
                    dataset!(group, "y3", &postprocessing.y3)?;
                }

                if profiles.y4 {
                    dataset!(group, "y4", &postprocessing.y4)?;
                }

                if profiles.xi_r {
                    dataset!(group, "xi_r", &postprocessing.xi_r)?;
                }

                if profiles.xi_h {
                    dataset!(group, "xi_h", &postprocessing.xi_h)?;
                }

                if profiles.xi_tp {
                    dataset!(group, "xi_tp", &postprocessing.xi_tp)?;
                }

                if profiles.xi_tn {
                    dataset!(group, "xi_tn", &postprocessing.xi_tn)?;
                }

                if profiles.pressure {
                    dataset!(group, "pressure", &postprocessing.p)?;
                }

                if profiles.density {
                    dataset!(group, "density", &postprocessing.rho)?;
                }

                if profiles.gravity_potential {
                    dataset!(group, "gravity-acceleration", &postprocessing.dpsi)?;
                }

                if profiles.gravity_potential {
                    dataset!(group, "gravity-potential", &postprocessing.psi)?;
                }

                if profiles.divergence {
                    dataset!(group, "divergence", &postprocessing.chi)?;
                }

                if profiles.nodes {
                    dataset!(group, "nodes", &postprocessing.nodes)?;
                }
            }
        }

        if properties.frequency {
            dataset!(output, "frequency", &freq)?;
        }

        if properties.degree {
            dataset!(output, "degree", &ell)?;
        }

        if properties.radial_order && self.postprocessing.is_some() {
            dataset!(output, "radial-order", &radial_order)?;
        }

        if properties.azimuthal_order {
            dataset!(output, "azimuthal-order", &m)?;
        }

        let perturbation_group = output.create_group("deformation")?;

        for mode_coupling in &self.perturbed_frequencies {
            let group = perturbation_group.create_group(&format!("{}", mode_coupling.m))?;

            if properties.deformed_frequency {
                let freqs = &mode_coupling
                    .freqs
                    .iter()
                    .map(|x| -> Result<H5Complex> {
                        Ok(H5Complex {
                            re: frequency_units.convert_from_natural(x.real(), &input.scale)?,
                            im: frequency_units
                                .convert_from_natural(x.imaginary(), &input.scale)?,
                        })
                    })
                    .collect::<Result<Vec<H5Complex>>>()?;

                dataset!(group, "frequency", &freqs)?;
            }

            if properties.deformed_eigenvector {
                dataset!(
                    group,
                    "eigenvector",
                    &mode_coupling
                        .coupling
                        .map(|x| H5Complex {
                            re: x.real(),
                            im: x.imaginary(),
                        })
                        .as_ndarray2()
                        .as_standard_layout()
                )?;
            }

            if properties.coupling_matrix {
                dataset!(
                    group,
                    "d",
                    &mode_coupling.d.as_ndarray2().as_standard_layout()
                )?;
                dataset!(
                    group,
                    "r",
                    &mode_coupling.r.as_ndarray2().as_standard_layout()
                )?;
                dataset!(
                    group,
                    "l",
                    &mode_coupling.l.as_ndarray2().as_standard_layout()
                )?;
            }
        }

        if !keep_data {
            self.solutions.clear();
            self.postprocessing = None;
            self.perturbed_frequencies.clear();
        }

        Ok(())
    }

    fn clear(&mut self) -> Result<(), Report> {
        self.input = None;
        self.perturbed_frequencies.clear();
        self.solutions.clear();
        self.postprocessing = None;

        Ok(())
    }
}

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
