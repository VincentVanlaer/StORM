#![feature(never_type)]
#![feature(generic_const_exprs)]
#![expect(incomplete_features)]

use std::cell::Cell;

use color_eyre::Result;
use ndarray::aview0;
use storm::{
    bracket::{
        BracketOptimizer, FilterSignSwap as _, InverseQuadratic, InverseQuadraticState, Point,
        Precision,
    },
    dynamic_interface::{DifferenceSchemes, MultipleShooting},
    model::StellarModel,
    system::adiabatic::Rotating1D,
};

struct IntermediateStateBalanced {
    state: InverseQuadraticState,
    determinants: Vec<Point>,
}

fn linspace(lower: f64, upper: f64, n: usize) -> impl Iterator<Item = f64> {
    (0..n).map(move |x| lower + (upper - lower) * (x as f64) / ((n - 1) as f64))
}

fn main() -> Result<()> {
    let lower: f64 = 1.0;
    let upper: f64 = 25.0;
    let steps: usize = 25;
    let difference_scheme = DifferenceSchemes::Colloc2;

    let model = StellarModel::from_gsm("test-data/test-model-tams.GSM")?;
    let system = Rotating1D::new(0, 0);
    let searcher = &InverseQuadratic {};
    let determinant = MultipleShooting::new(&model, system, difference_scheme);

    let dets: Vec<_> = linspace(lower, upper, steps)
        .map(|x| Point {
            x,
            f: determinant.det(x),
        })
        .collect();

    let solutions: Vec<_> = dets
        .iter()
        .cloned()
        .filter_sign_swap()
        .map(|(point1, point2)| {
            let mut bracket_state = Vec::new();
            let evals = Cell::<i64>::new(0);
            let bracket = searcher.optimize(
                point1,
                point2,
                |point| {
                    if evals.get() > 20 {
                        Err(())
                    } else {
                        Ok(determinant.det(point))
                    }
                },
                Precision::Relative(0.),
                Some(&mut |state| {
                    println!(
                        "{} {} {} {} {}",
                        state.lower.x, state.lower.f, state.upper.x, state.upper.f, state.next_eval
                    );
                    println!(
                        "ULPs between lower and upper: {}",
                        state.upper.x.to_bits() - state.lower.x.to_bits()
                    );
                    let mut determinants = Vec::new();

                    for i in 0..101 {
                        let p =
                            state.lower.x + f64::from(i) * (state.upper.x - state.lower.x) / 100.;

                        determinants.push(Point {
                            x: p,
                            f: determinant.det(p),
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
            println!("{:.20} {}", sol.root, sol.evals);
            sol_group
                .new_attr_builder()
                .with_data(aview0(&sol.root))
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
