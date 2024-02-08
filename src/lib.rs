#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]
#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]
#![allow(incomplete_features)]

pub mod bracket;
mod linalg;
pub mod model;
pub mod solver;
pub mod stepper;
pub mod system;

extern crate blas_src;
extern crate lapack_src;
