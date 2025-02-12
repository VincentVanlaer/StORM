import sys
from pathlib import Path
from subprocess import run
from math import pi
import h5py
import matplotlib.pyplot as plt
import numpy as np
from ..legendre import legendre
from .polar import map_min_delta, plot_mode_grid

theta = np.linspace(0, pi, 180 * 2)
leg = legendre(theta)


def load_order(
    order: int, r_idx: np.ndarray, f: h5py.File
) -> tuple[np.ndarray, np.ndarray]:
    azimuthal = f["m"][:]
    sol = f["solutions"]
    degree = f["degree"][:]

    select = np.flatnonzero(azimuthal == order)

    modes = np.array(
        [
            np.outer(
                (
                    sol[str(key)]["pressure"][r_idx]
                    - sol[str(key)]["xi_r"][r_idx] * density[r_idx] * g[r_idx]
                )
                / pressure[r_idx],
                leg[(degree[key], order)],
            )
            for key in select
        ]
    )

    freqs = f["deformation"][str(order)]["frequency"]["re"]
    vectors = f["deformation"][str(order)]["eigenvector"]["re"][: len(modes), :]

    perturbed_eigenvectors = np.tensordot(vectors, modes, ((0,), (0,)))

    return freqs, perturbed_eigenvectors


def plot_order(r_coord: np.ndarray, name: str, freqs: np.ndarray, vectors: np.ndarray):
    skey = np.argsort(freqs)

    data = list(zip(vectors[skey], freqs[skey]))

    plot_mode_grid(data, r_coord, theta, name)


def load_and_plot_file(name: str, path: str) -> None:
    f = h5py.File(path)

    r_coord = f["radial-coordinate"][:]
    r_coord = r_coord / np.max(r_coord)
    idx = map_min_delta(r_coord, 0.0005)
    r_coord = r_coord[idx]

    zonal = load_order(0, idx, f)
    prograde = load_order(1, idx, f)
    retrograde = load_order(-1, idx, f)

    plot_order(r_coord, f"{name} retrograde", retrograde[0], retrograde[1])
    plot_order(r_coord, f"{name} zonal", zonal[0], zonal[1])
    plot_order(r_coord, f"{name} prograde", prograde[0], prograde[1])

    f.close()


model = "test-data/top-zams-model.GSM"

if len(sys.argv) == 2 and sys.argv[1] == "rerun":
    p = Path("test-data/generated/storm-zams-structure/")

    p.mkdir(exist_ok=True)

    input = """
    input test-data/top-zams-model.GSM
    set-rotation-constant .1

    scan --frequency-units=cycles-per-day 0 0 5. 30. 100
    scan --frequency-units=cycles-per-day 2 0 5. 30. 100
    scan --frequency-units=cycles-per-day 4 0 5. 30. 100
    scan --frequency-units=cycles-per-day 6 0 5. 30. 100

    scan --frequency-units=cycles-per-day 2 1 5. 30. 100
    scan --frequency-units=cycles-per-day 4 1 5. 30. 100
    scan --frequency-units=cycles-per-day 6 1 5. 30. 100

    scan --frequency-units=cycles-per-day 2 -1 5. 30. 100
    scan --frequency-units=cycles-per-day 4 -1 5. 30. 100
    scan --frequency-units=cycles-per-day 6 -1 5. 30. 100

    post-process

    deform .1
    perturb-deformed 0
    perturb-deformed -1
    perturb-deformed 1

    output test-data/generated/storm-zams-structure/even.hdf5 --frequency-units=cycles-per-day --properties frequency,degree,azimuthal-order,deformed-frequency,deformed-eigenvector,coupling-matrix --profiles radial-coordinate,y1,y2,y3,y4,xi_r,xi_h,pressure,density --model-properties dynamical-frequency,deformation-beta,deformation-dbeta,deformation-ddbeta,deformation-rotation-frequency

    scan --frequency-units=cycles-per-day 1 0 5. 30. 100
    scan --frequency-units=cycles-per-day 3 0 5. 30. 100
    scan --frequency-units=cycles-per-day 5 0 5. 30. 100

    scan --frequency-units=cycles-per-day 1 1 5. 30. 100
    scan --frequency-units=cycles-per-day 3 1 5. 30. 100
    scan --frequency-units=cycles-per-day 5 1 5. 30. 100

    scan --frequency-units=cycles-per-day 1 -1 5. 30. 100
    scan --frequency-units=cycles-per-day 3 -1 5. 30. 100
    scan --frequency-units=cycles-per-day 5 -1 5. 30. 100

    post-process

    deform .1
    perturb-deformed 0
    perturb-deformed -1
    perturb-deformed 1

    output test-data/generated/storm-zams-structure/odd.hdf5 --frequency-units=cycles-per-day --properties frequency,degree,azimuthal-order,deformed-frequency,deformed-eigenvector,coupling-matrix --profiles radial-coordinate,y1,y2,y3,y4,xi_r,xi_h,pressure,density --model-properties dynamical-frequency,deformation-beta,deformation-dbeta,deformation-ddbeta,deformation-rotation-frequency
    """

    run(["cargo", "run", "--release", "--bin=storm"], input=input, text=True)


f = h5py.File(model)

pressure = f["P"][:]
density = f["rho"][:]
G = 6.67430e-8
r = f["r"][:]
g = G * f["M_r"][:] / r**2
g[0] = 0


f.close()

load_and_plot_file("StORM: Even", "test-data/generated/storm-zams-structure/even.hdf5")
load_and_plot_file("StORM: Odd", "test-data/generated/storm-zams-structure/odd.hdf5")

plt.show()
