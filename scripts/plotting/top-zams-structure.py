import pickle
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from .polar import map_min_delta, plot_mode_grid

# Keys:
# 'm'   : azimuthal order TOP (m=-1 for prograde)
# 'n'   : pseudo radial order
# 'l'   : pseudo spherical degree
# 'freq': mode frequency in d^-1
# 'mode': 2D pressure perturbation
# 'vis' : mode visibility at i=45deg


def is_close(freq: float, prev_freqs: list[float]) -> bool:
    for prev_freq in prev_freqs:
        if abs(prev_freq - freq) < 0.001:
            return True

    return False


def rescale_and_cut(data: np.ndarray) -> np.ndarray:
    _, n_theta = data.shape
    n_theta = int(n_theta / 2)
    data = data[idx, :n_theta]

    return data


def do_plot(modes: list[tuple[np.ndarray, float]], name: str) -> None:
    modes = list(map(lambda x: (rescale_and_cut(x[0]), x[1]), modes))
    theta = np.linspace(0, pi, modes[0][0].shape[1])

    plot_mode_grid(modes, r_coord, theta, name)


def filter_modes(data: dict, m: int) -> list[tuple[np.ndarray, float]]:
    prev_freqs: list[float] = []
    modes = []

    for freq, g in zip(data["freq"][data["m"] == m], data["mode"][data["m"] == m]):
        if is_close(freq, prev_freqs):
            continue

        modes.append((g, freq))

        prev_freqs.append(freq)

    return sorted(modes, key=lambda x: x[1])


def load_and_plot(name: str, path: str):
    data = pickle.load(open(path, "rb"))

    do_plot(filter_modes(data, -1), f"{name} prograde")
    do_plot(filter_modes(data, 0), f"{name} zonal")
    do_plot(filter_modes(data, 1), f"{name} retrograde")


grid = pickle.load(open("test-data/generated/grid_O10.pkl", "rb"))
r_coord = grid["radius"][0][:, 0]

idx = map_min_delta(r_coord, 0.002)

r_coord = r_coord[idx]

load_and_plot("TOP: Odd", "test-data/generated/identified_npts60_nth10_O10_odd_vec.pkl")
load_and_plot(
    "TOP: Even", "test-data/generated/identified_npts60_nth10_O10_even_vec.pkl"
)

plt.show()
