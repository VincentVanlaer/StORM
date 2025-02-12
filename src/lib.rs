//! Stellar Oscillations with Rotation and Magnetism
//!
//! ```no_run
//! # use storm::bracket::Precision;
//! # use storm::dynamic_interface::{DifferenceSchemes, MultipleShooting};
//! # use storm::model::StellarModel;
//! # use storm::system::adiabatic::{GridScale, Rotating1D};
//! let model = StellarModel::from_gsm("model.hdf5").unwrap();
//!
//! let lower = 1.0;
//! let upper = 25.0;
//! let n_steps = 25;
//! let degree = 1;
//! let order = 0;
//!
//! let system = Rotating1D::from_model(&model, degree, order);
//! let grid = GridScale { scale: 0 };
//! let solver = MultipleShooting::new(&system, DifferenceSchemes::Colloc2, &grid);
//!
//! let solutions: Vec<_> = solver
//!     .scan_and_optimize(
//!         linspace(lower, upper, n_steps),
//!         Precision::Relative(0.),
//!     )
//!     .collect();
//!
//! for s in solutions {
//!     println!("{}", s.root);
//! }
//! #
//! # fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
//! #     (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
//! # }
//! ```
#![feature(generic_const_exprs)]
#![feature(const_option)]
#![feature(iter_map_windows)]
#![feature(never_type)]
#![feature(unwrap_infallible)]
#![feature(unsigned_is_multiple_of)]
#![feature(trait_alias)]
#![expect(incomplete_features)]
#![allow(clippy::needless_range_loop)] // Makes math code less readable
#![allow(clippy::too_many_arguments)] // Bit silly
#![warn(missing_docs)]

extern crate lapack_src;

pub mod bracket;
pub mod dynamic_interface;
mod linalg;
pub mod model;
pub mod postprocessing;
mod solver;
mod stepper;
pub mod system;
