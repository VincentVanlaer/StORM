==========================
Frequently asked questions
==========================

I have a problem/found a bug
============================

Report the problem on `GitHub <https://github.com/VincentVanlaer/StORM/issues>`_. Provide sufficient information to help us diagnose the problem:

- Version of StORM you are running
- What commands did you run?
- What is the error message, or what did you expect to happen but didn't?
- Anything else you think might be relevant.

How do I control the number of threads StORM uses?
==================================================

By default, StORM uses all available threads on your system.
The environment variable ``RAYON_NUM_THREADS`` can be used to set the number of threads that StORM should use.
This is similar to how ``OMP_NUM_THREADS`` controls parallelism in OpenMP enabled software.
Note that since StORM does not use OpenMP, ``OMP_NUM_THREADS`` has no effect.

How does parallelism in StORM work?
===================================

StORM will distribute the following tasks over multiple threads:

- Computing the determinant of each frequency in the scanning grid
- Root finding of the determinant after a sign swap has been found
- Post processing of each oscillation mode

Within these tasks, no additional parallelism is present.
Unless very large models are involved (~100 000 grid points), the baseline speed (~milliseconds for a single mode) of StORM is already fast enough that such additional parallelism is not needed.

What does "singularity detected" mean?
====================================

Sometimes you will encounter the following message:

   X singularities encountered between Y and Z
   This typically indicates that the model grid resolution is too low or the frequency scan range is too low for the rotation rate. See the FAQ page on the documentation website for more info.

So this is that FAQ entry. StORM finds oscillation modes looking for changes in sign of the determinant of a big matrix describing the oscillation problem. Solutions are where the determinant is zero. However, in certain scenarios, sign changes may occur by the determinant going to :math:`\pm\infty`, i.e. a singularity. There are two reasons for singularities:

- Very low model resolution (i.e. tens to hundreds of points). This can be fixed by resampling.
- Scanning frequencies for prograde modes at too low frequencies. In the limit :math:`n \to -\infty`, the frequencies approach a non-zero lower bound, given by the rotation rate and the asymptotic sensitivity of the modes to the rotation rate. Below this limit, singularities will pop up.

The second case cannot always be easily fixed, as the lower frequency bound is model and rotation-profile dependent, but in general the solution is to shift the scan range towards higher frequencies. Changing the model resolution will not change this. It is also important to keep in mind that if singularities are encountered, there will be modes that are not accurate. This is because the g modes become infinitely dense near the lower bound. A finite subset of these modes will be detected, but your model and the solver will not have enough resolution to resolve them. Frequencies, eigenfunctions, and mode identifications will be inaccurate.
