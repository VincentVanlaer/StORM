//! Stellar Oscillations with Rotation and Magnetism
#![allow(clippy::needless_range_loop)] // Makes math code less readable
#![allow(clippy::too_many_arguments)] // Bit silly
#![warn(missing_docs)]

pub mod bracket;
pub mod dynamic_interface;
mod gaunt;
mod linalg;
pub mod model;
pub mod perturbed;
pub mod postprocessing;
mod solver;
mod stepper;
pub mod system;
