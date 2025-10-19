---
title: Rotation
icon: 360 
katex: true
---

Stellar rotation affects the pulsations frequencies of the oscillation modes.
This happens for various reasons:

- Rotation introduces an additional term (the centrifugal acceleration) in the equation of hydrostatic support, changing the structure and evolution of the stellar model
- Rotational shears may introduce additional forms of mixing in stellar interiors
- Additional terms for the Coriolis force and the stellar deformation are added to the oscillation equations. These terms are not spherically symmetric.

For StORM, this last aspect is the most relevant.
When the star is no longer spherically symmetric, the oscillation equations are no longer separable in $\theta$ and $r$.
Hence, a complete solution of the oscillation equations must be constructed from multiplet spherical harmonics.
This is a complicated endeavour, and most analyses are done using perturbative approaches, or otherwise constrained solution sets.

## Coriolis acceleration

The Coriolis acceleration is the only term that is first order in the rotation.
It is therefore typically the only term that is considered when performing stellar oscillation calculations with rotation.
When perturbing the oscillation equations with this term, one finds (e.g. [Aerts et al. 2010](https://ui.adsabs.harvard.edu/abs/2010aste.book.....A/abstract))
$$ \omega_{\rm rot} = \omega_0 + m \int_0^R K(r) \Omega(r) dr\,, $$
where $K(r)$ depends on the structure of the mode and describes its sensitivity to the rotation rate at radius $r$.
This assumes shellular rotation.

StORM includes the Coriolis acceleration in the oscillation equation solver, still assuming that the solution consists of a single spherical harmonic, without the toroidal components.
This results in frequencies that are similar to $\omega_{\rm rot}$ as describe above.
These effects are always included in the computations done by the `scan` parameter and, unless otherwise specified, will use the rotation rate set in the stellar model.
To change the rotation rate use either the `set-rotation-constant` command or the `set-rotation-overlay`.
The second command loads the rotation rate from an HDF5 file with the same GSM file format as the input model.

## Centrifugal deformation

The centrifugal acceleration counteracts part of the gravitional acceleration.
This increases the equatorial radius of the star compared to the polar radius, deforming the star.
These changes to the structure of the star affect the pulsation frequencies.
Whether this effect is relevant depends on the particular science case (and the properties of the star and the oscillation modes).

StORM computes the effect of the stellar deformation on the oscillation modes in a perturbative way.
However, in order to support 1D input models, we first need to compute how deformed the input model should be given a certain rotation rate.
This too is done perturbatively.
First, we parametrize the deformed star with the coordinates $\zeta$ and $\theta$, where $\zeta = 0$ is the center of the star, and $\zeta = 1$ is the surface of the star.
The physical radial coordinate is then given by

$$ r = \zeta(1 + \varepsilon(\zeta, \theta)) $$

The variable $\varepsilon$ represents the shift of the isobars due to the rotation, such that the isobars are at constant values of $\zeta$.
We assume that properties of the rotating star that depend on the radius, such as density and pressure, now only depend on $\zeta$.
The only variable that then needs to be determined is $\varepsilon$.
We use the Chandrasekhar-Milne expansion for this.
It defines $\varepsilon$ as a series of even Legendre polynomials:
$$ \varepsilon(\zeta, \theta) = \sum_{k \geq 0} \varepsilon_k(\zeta) P_{2k}(\theta) \,. $$
StORM includes the $P_0$ and $P_2$ term.

With the deformed structure, StORM perturbs the oscillation frequencies, assuming that the perturbed displacement functions can be written as a sum of unperturbed modes:

$$\bm{\tilde\xi}_k = \sum_{k'} a_{kk'} \bm{\xi}_{k'}\,.$$
 
In principle, this sum should range over all modes, including those of different degrees.
Since rotation preserves the azimuthal symmetry, it is not necessary to include modes with different azimuthal orders.
Additionally, only modes with either even or odd degree need to be included, as rotation only couples with $\ell \pm 2$ if the deformation is limited to the $P_2$ term as described above.
However, it is not possible to actually compute this infinite sum.
Instead, one must make a cut-off in radial order and spherical degree.

Apart from the deformation terms, StORM also includes the effect of the toroidal components of the oscillation modes on the oscillation frequencies.
These components are computed analytically from the poloidal components, which results in toroidal modes that are accurate up to first order in the rotation rate.

To then perturb the oscillation equations, StORM expands the oscillation equations with the sum defined above, which results in a matrix equation.
The eigenvalues and eigenvectors of this equations give us the perturbed oscillation frequencies and the parameters $a_{kk'}$ defined above.

## Example

This section contains an example that shows how to
- deform a stellar model
- computed the perturbed oscillation modes

As usual, we first need to load in a model:

```sh
[storm] > input model.GSM
```

We can set a specific constant rotation rate if required (the default units are the fraction of the critical rotation rate):

```sh
[storm] > set-rotation-constant 0.1
```

With this model loaded, we can search for the oscillation modes using the `scan` command.
All modes that are found here will be included in the sum of eigenfunctions defined above.

```sh
[storm] > scan  --frequency-units=cycles-per-day 0 0 10 18 30
[storm] > scan  --frequency-units=cycles-per-day 2 0 10 18 30
```

In this case we compute zonal modes of $\ell = 0$ and $\ell = 2$.
Since they differ by two, they will be coupled by the rotation.
To do this coupling, we need two things: the eigenfunctions and the deformed stellar model.
We compute this as follows:

```sh
[storm] > deform 0.1
[storm] > post-process
```

The deform command also requires the rotation rate, as the standard form of the Chandrasekhar-Milne expansion only supports constant rotation.
Since the deformation is strongest at the surface, ideally the rotation rate corresponds to the (near-)surface rotation rate.

Finally, we can run the actual perturbation of the oscillation modes:

```sh
[storm] > perturb-deformed 0
```

The single argument to this command is the azimuthal order to select, which in our case is zonal modes.

After these commands, the following additional outputs are enabled:

- `deformation-alpha`, `deformation-dalpha`, and `deformation-ddalpha`: these define the $P_0$ deformed structure of the star, and correspond to $\alpha(a)$ in Eq. (27) of [Lee and Baraffe (1995)](https://ui.adsabs.harvard.edu/abs/1995A&A...301..419L)
- `deformation-beta`, `deformation-dbeta`, and `deformation-ddbeta`: these define the $P_2$ deformed structure of the star, and correspond to $\beta(a)$ in Eq. (27) of [Lee and Baraffe (1995)](https://ui.adsabs.harvard.edu/abs/1995A&A...301..419L)
- `deformed-frequency`: Perturbed oscillation frequencies
- `deformed-eigenvector`: The parameters $a_{kk'}$

More details can be obtained from the `help output` command.

## Challenge: mode identification

Since the perturbation method mixes the oscillation modes together and just produces a list of oscillation frequencies and eigenfunctions, one may wonder how to map these frequencies back onto the non-rotation oscillation modes.
In general, this is not possible.
However, one can use some heuristics to determine which perturbed frequencies match the non-perturbed frequencies:

- Increase the rotation rate step by step, and match the modes based on frequencies
- Use the parameters $a_{kk'}$ to determine which non-perturbed mode contributes most to the perturbed oscillation mode.

These methods are currently not implemented in StORM.
They are expected to break down at higher rotation rates.
