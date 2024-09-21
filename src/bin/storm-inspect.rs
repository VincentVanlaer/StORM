#![feature(never_type)]
#![feature(generic_const_exprs)]

use std::cell::Cell;

use color_eyre::Result;
use ndarray::aview0;
use storm::{
    bracket::{
        Balanced, BalancedState, BracketResult, BracketSearcher, BrentState, Point,
        SearchBrackets as _,
    },
    dynamic_interface::{get_solvers, DifferenceSchemes},
    helpers::linspace,
    model::StellarModel,
    system::adiabatic::{ModelGrid, Rotating1D},
};

struct IntermediateState {
    brent: BrentState,
    determinants: Vec<Point>,
}

struct IntermediateStateBalanced {
    state: BalancedState,
    determinants: Vec<Point>,
}

struct Solution {
    bracket: BracketResult,
    bracketing: Vec<IntermediateState>,
}

fn main() -> Result<()> {
    let lower: f64 = 15.0;
    let upper: f64 = 16.0;
    let steps: usize = 2;
    let difference_scheme = DifferenceSchemes::Magnus4;

    let model = StellarModel::from_gsm(&hdf5::File::open("test-data/test-model.GSM")?)?;
    let system = Rotating1D::from_model(&model, 0, 0)?;
    let grid = &ModelGrid { scale: 0 };
    let searcher = &Balanced { rel_epsilon: 0. };
    let (system_matrix, determinant) = get_solvers(&system, difference_scheme, &grid);

    let dets: Vec<_> = linspace(lower, upper, steps)
        .map(|x| Point {
            x,
            f: determinant(x),
        })
        .collect();

    let solutions: Vec<_> = dets
        .iter()
        .brackets()
        .map(|(point1, point2)| {
            let mut bracket_state = Vec::new();
            let evals = Cell::<i64>::new(0);
            let bracket = searcher.search(
                *point1,
                *point2,
                |point| {
                    if evals.get() > 20 {
                        Err(())
                    } else {
                        Ok(determinant(point))
                    }
                },
                Some(&mut |state| {
                    println!(
                        "{} {} {} {} {}",
                        state.lower.x, state.lower.f, state.upper.x, state.upper.f, state.next_eval
                    );
                    let mut determinants = Vec::new();

                    for i in 0..101 {
                        let p =
                            state.lower.x + f64::from(i) * (state.upper.x - state.lower.x) / 100.;

                        determinants.push(Point {
                            x: p,
                            f: system_matrix(p).unwrap().determinant(),
                        });
                    }

                    bracket_state.push(IntermediateStateBalanced {
                        state,
                        determinants,
                    });

                    evals.set(evals.get() + 1);
                }),
            );

            (bracket, bracket_state)
        })
        .collect();

    // let solutions: Vec<_> = dets
    //     .windows(2)
    //     .filter_map(|window| {
    //         let pair1 = window[0];
    //         let pair2 = window[1];

    //         if pair1.f.signum() != pair2.f.signum() {
    //             Some((pair1, pair2))
    //         } else {
    //             None
    //         }
    //     })
    //     .collect::<Vec<_>>()
    //     .into_iter()
    //     .map(|(lower, upper)| {
    //         let mut bracket_state = Vec::new();
    //         let mut evals = 0;
    //         let bracket = (Brent { rel_epsilon: 1e-15 })
    //             .search(
    //                 lower,
    //                 upper,
    //                 |point| system_matrix(point).map(|x| x.determinant()),
    //                 Some(&mut |state| {
    //                     println!("{} {} {} {} {} {}", state.current.x, state.current.f, state.counterpoint.x, state.counterpoint.f, state.previous.x, state.previous.f);
    //                     let mut determinants = Vec::new();
    //                     let mut counterpoint = state.counterpoint;
    //                     let previous = state.previous;
    //                     let current = state.current;

    //                     if counterpoint.f.signum() == current.f.signum() {
    //                         counterpoint = previous;
    //                     }

    //                     for i in 0..1001 {
    //                         let p = counterpoint.x
    //                             + f64::from(i) * (current.x - counterpoint.x) / 1000.;

    //                         determinants.push(Point {
    //                             x: p,
    //                             f: system_matrix(p).unwrap().determinant(),
    //                         });
    //                     }

    //                     bracket_state.push(IntermediateState {
    //                         brent: state,
    //                         determinants,
    //                     });

    //                     evals += 1;
    //                 }),
    //             )
    //             .expect("Bracket failed");

    //         Solution {
    //             bracket,
    //             bracketing: bracket_state,
    //         }
    //     })
    //     .collect();

    // let output = hdf5::File::create("test-data/generated/brent-inspect.hdf5")?;

    // let scan_group = output.create_group("scan")?;

    // scan_group
    //     .new_dataset_builder()
    //     .with_data(dets.iter().map(|x| x.x).collect::<Vec<_>>().as_slice())
    //     .create("freq")?;

    // scan_group
    //     .new_dataset_builder()
    //     .with_data(dets.iter().map(|x| x.f).collect::<Vec<_>>().as_slice())
    //     .create("value")?;

    // let sol_parent_group = output.create_group("sol")?;

    // for (i, sol) in solutions.iter().enumerate() {
    //     let sol_group = sol_parent_group.create_group(&format!("sol_{i}"))?;

    //     println!("{:.20} {}", sol.bracket.freq, sol.bracket.evals);
    //     sol_group
    //         .new_attr_builder()
    //         .with_data(aview0(&sol.bracket.evals))
    //         .create("evals")?;
    //     sol_group
    //         .new_attr_builder()
    //         .with_data(aview0(&sol.bracket.freq))
    //         .create("freq")?;

    //     for (j, eval) in sol.bracketing.iter().enumerate() {
    //         let eval_group = sol_group.create_group(&format!("eval_{j}"))?;

    //         eval_group
    //             .new_dataset_builder()
    //             .with_data(
    //                 eval.determinants
    //                     .iter()
    //                     .map(|x| x.x)
    //                     .collect::<Vec<_>>()
    //                     .as_slice(),
    //             )
    //             .create("freq")?;

    //         eval_group
    //             .new_dataset_builder()
    //             .with_data(
    //                 eval.determinants
    //                     .iter()
    //                     .map(|x| x.f)
    //                     .collect::<Vec<_>>()
    //                     .as_slice(),
    //             )
    //             .create("value")?;

    //         eval_group
    //             .new_attr_builder()
    //             .with_data(aview0(&eval.brent.current.x))
    //             .create("current")?;

    //         eval_group
    //             .new_attr_builder()
    //             .with_data(aview0(&eval.brent.previous.x))
    //             .create("previous")?;

    //         eval_group
    //             .new_attr_builder()
    //             .with_data(aview0(&eval.brent.counterpoint.x))
    //             .create("counterpoint")?;

    //         eval_group
    //             .new_attr_builder()
    //             .with_data(aview0(&eval.brent.next_eval))
    //             .create("next")?;

    //         eval_group
    //             .new_attr_builder()
    //             .with_data(aview0(&VarLenUnicode::from_str(
    //                 &eval.brent.method.to_string(),
    //             )?))
    //             .create("method")?;
    //     }
    // }

    let output = hdf5::File::create("test-data/generated/balanced-inspect.hdf5")?;

    let scan_group = output.create_group("scan")?;

    scan_group
        .new_dataset_builder()
        .with_data(dets.iter().map(|x| x.x).collect::<Vec<_>>().as_slice())
        .create("freq")?;

    scan_group
        .new_dataset_builder()
        .with_data(dets.iter().map(|x| x.f).collect::<Vec<_>>().as_slice())
        .create("value")?;

    let sol_parent_group = output.create_group("sol")?;

    for (i, sol) in solutions.iter().enumerate() {
        let sol_group = sol_parent_group.create_group(&format!("sol_{i}"))?;

        sol_group
            .new_attr_builder()
            .with_data(aview0(&sol.1.len()))
            .create("evals")?;
        if let Ok(sol) = &sol.0 {
            println!("{:.20} {}", sol.freq, sol.evals);
            sol_group
                .new_attr_builder()
                .with_data(aview0(&sol.freq))
                .create("freq")?;
        }

        for (j, eval) in sol.1.iter().enumerate() {
            let eval_group = sol_group.create_group(&format!("eval_{j}"))?;

            eval_group
                .new_dataset_builder()
                .with_data(
                    eval.determinants
                        .iter()
                        .map(|x| x.x)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .create("freq")?;

            eval_group
                .new_dataset_builder()
                .with_data(
                    eval.determinants
                        .iter()
                        .map(|x| x.f)
                        .collect::<Vec<_>>()
                        .as_slice(),
                )
                .create("value")?;

            eval_group
                .new_attr_builder()
                .with_data(aview0(&eval.state.upper.x))
                .create("upper")?;

            eval_group
                .new_attr_builder()
                .with_data(aview0(&eval.state.lower.x))
                .create("lower")?;

            eval_group
                .new_attr_builder()
                .with_data(aview0(&eval.state.offset))
                .create("offset")?;

            if let Some(p) = eval.state.previous {
                eval_group
                    .new_attr_builder()
                    .with_data(aview0(&p.x))
                    .create("previous")?;
            }

            eval_group
                .new_attr_builder()
                .with_data(aview0(&eval.state.next_eval))
                .create("next")?;
        }
    }

    Ok(())
}
