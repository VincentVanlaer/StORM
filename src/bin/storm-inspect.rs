#![feature(generic_const_exprs)]

use color_eyre::{
    eyre::{eyre, Context},
    Result,
};
use ndarray::aview0;
use storm::{
    bracket::{Balanced, BalancedState, BracketResult, BracketSearcher, BrentState, Point},
    model::StellarModel,
    solver::{decompose_system_matrix, DecomposedSystemMatrix},
    stepper::Colloc2,
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
    let lower: f64 = 1.0;
    let upper: f64 = 6.0;
    let steps: usize = 6;
    let model = StellarModel::from_gsm(&hdf5::File::open("test-data/test-model.GSM")?)?;
    let system = Rotating1D::from_model(&model, 0, 0)?;
    let system_matrix = |freq: f64| -> Result<DecomposedSystemMatrix> {
        decompose_system_matrix(&system, &Colloc2 {}, &ModelGrid { scale: 0 }, freq)
            .or(Err(eyre!("Failed determinant")))
    };

    let mut dets = vec![Point { x: 0.0, f: 0.0 }; steps];
    for i in 0..steps {
        let freq = lower + i as f64 / (steps - 1) as f64 * (upper - lower);
        dets[i] = Point {
            x: freq,
            f: system_matrix(freq)
                .wrap_err("Frequency scan failed")?
                .determinant(),
        };
    }

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

    let solutions: Vec<_> = dets
        .windows(2)
        .filter_map(|window| {
            let pair1 = window[0];
            let pair2 = window[1];

            if pair1.f.signum() != pair2.f.signum() {
                Some((pair1, pair2))
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|(lower, upper)| {
            let mut bracket_state = Vec::new();
            let mut evals = 0;
            let bracket = (Balanced { rel_epsilon: 1e-15 })
                .search(
                    lower,
                    upper,
                    |point| system_matrix(point).map(|x| x.determinant()),
                    Some(&mut |state| {
                        println!(
                            "{} {} {} {} {}",
                            state.lower.x,
                            state.lower.f,
                            state.upper.x,
                            state.upper.f,
                            state.next_eval
                        );
                        let mut determinants = Vec::new();

                        for i in 0..1001 {
                            let p = state.lower.x
                                + f64::from(i) * (state.upper.x - state.lower.x) / 1000.;

                            determinants.push(Point {
                                x: p,
                                f: system_matrix(p).unwrap().determinant(),
                            });
                        }

                        bracket_state.push(IntermediateStateBalanced {
                            state,
                            determinants,
                        });

                        evals += 1;
                    }),
                )
                .expect("Bracket failed");

            (bracket, bracket_state)
        })
        .collect();

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

        println!("{:.20} {}", sol.0.freq, sol.0.evals);
        sol_group
            .new_attr_builder()
            .with_data(aview0(&sol.0.evals))
            .create("evals")?;
        sol_group
            .new_attr_builder()
            .with_data(aview0(&sol.0.freq))
            .create("freq")?;

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
