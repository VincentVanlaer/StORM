---
title: Frequently asked questions
icon: question_mark
---

## I have a problem/found a bug

Report the problem on [GitHub](https://github.com/VincentVanlaer/StORM/issues). Provide sufficient information to help us diagnose the problem:

- Version of StORM you are running
- What commands did you run?
- What is the error message, or what did you expect to happen but didn't?
- Anything else you think might be relevant.

## How do I control the number of threads StORM uses?

By default, StORM uses all available threads on your system.
The environment variable `RAYON_NUM_THREADS` can be used to set the number of threads that StORM should use.
This is similar to how `OMP_NUM_THREADS` controls parallelism in OpenMP enabled software.
Note that since StORM does not use OpenMP, `OMP_NUM_THREADS` has no effect.

## How does parallelism in StORM work?

StORM will distribute the following tasks over multiple threads:

- Computing the determinant of each frequency in the scanning grid
- Root finding of the determinant after a sign swap has been found
- Post processing of each oscillation mode

Within these tasks, no additional parallelism is present.
Unless very large models are involved (~100 000 grid points), the baseline speed (~milliseconds for a single mode) of StORM is already fast enough that such additional parallelism is not needed. 
