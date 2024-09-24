#![feature(generic_const_exprs)]
#![feature(const_option)]
#![feature(iter_map_windows)]
#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]
#![allow(incomplete_features)]
#![allow(clippy::needless_range_loop)] // Makes math code less readable

pub mod bracket;
pub mod dynamic_interface;
pub mod helpers;
mod linalg;
pub mod model;
pub mod solver;
pub mod stepper;
pub mod system;

extern crate blas_src;
extern crate lapack_src;
