//! Stellar Oscillations with Rotation and Magnetism
#![feature(generic_const_exprs)]
#![feature(const_option)]
#![feature(iter_map_windows)]
#![feature(never_type)]
#![feature(unwrap_infallible)]
#![expect(incomplete_features)]
#![allow(clippy::needless_range_loop)] // Makes math code less readable
#![warn(missing_docs)]

pub mod bracket;
pub mod dynamic_interface;
mod linalg;
pub mod model;
mod solver;
mod stepper;
pub mod system;
