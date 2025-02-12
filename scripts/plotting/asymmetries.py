import sys
from threading import Thread
from subprocess import Popen, PIPE
from math import sqrt, inf, pi
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
import h5py
import numpy as np
from pathlib import Path

G = 6.67430e-8


class ModeMap:
    def __init__(self, non_rotating_freqs: np.ndarray, degrees: np.ndarray) -> None:
        sort_idx = np.argsort(non_rotating_freqs)
        self._mode_track = [[freq] for freq in non_rotating_freqs[sort_idx]]
        self._rotation_frequencies: list[float] = [0.0]
        self._degrees = degrees[sort_idx]

    def push_results(
        self,
        rotation_frequency: float,
        mode_frequencies: np.ndarray,
        degrees_from_vector: np.ndarray,
    ) -> None:
        self._rotation_frequencies.append(rotation_frequency)

        sort_idx = np.argsort(mode_frequencies)
        mode_frequencies = mode_frequencies[sort_idx]
        degrees_from_vector = degrees_from_vector[sort_idx]

        track_freqs = np.array([track[-1] for track in self._mode_track])

        print(mode_frequencies)

        assert len(mode_frequencies) >= len(track_freqs)

        min_norm = inf
        min_idx = 0
        for i in range(len(mode_frequencies) - len(track_freqs) + 1):
            norm = np.linalg.norm(
                mode_frequencies[i : i + len(track_freqs)] - track_freqs
            )

            if norm < min_norm:
                min_norm = norm
                min_idx = i

        mode_frequencies = mode_frequencies[min_idx : min_idx + len(track_freqs)]
        degrees_from_vector = degrees_from_vector[min_idx : min_idx + len(track_freqs)]

        # check for crossings
        for idx in np.nonzero(np.diff(mode_frequencies) < 0.005)[0]:
            if degrees_from_vector[idx] != self._degrees[idx]:
                if not (
                    degrees_from_vector[idx] == self._degrees[idx + 1]
                    and degrees_from_vector[idx + 1] == self._degrees[idx]
                ):
                    # ship it
                    continue

                self._degrees[idx], self._degrees[idx + 1] = (
                    self._degrees[idx + 1],
                    self._degrees[idx],
                )
                self._mode_track[idx], self._mode_track[idx + 1] = (
                    self._mode_track[idx + 1],
                    self._mode_track[idx],
                )

        for track, freq in zip(self._mode_track, mode_frequencies):
            track.append(freq)


@dataclass
class Multiplet:
    negative: float | None
    zero: float | None
    positive: float | None


class MultipletMap:
    def __init__(
        self,
        degree: int | None,
        negative: ModeMap,
        positive: ModeMap,
        zero: ModeMap,
    ) -> None:
        assert negative._rotation_frequencies == positive._rotation_frequencies
        assert negative._rotation_frequencies == zero._rotation_frequencies

        self._combined_tracks = []
        self._rot = zero._rotation_frequencies.copy()

        for idx, track in enumerate(zero._mode_track):
            if degree is not None and degree != zero._degrees[idx]:
                continue

            for track_pos in positive._mode_track:
                if abs((track_pos[0] - track[0])) < 1e-6:
                    break
            else:
                continue

            for track_neg in negative._mode_track:
                if abs((track_neg[0] - track[0])) < 1e-6:
                    break
            else:
                continue

            self._combined_tracks.append(
                [
                    Multiplet(neg, zero, pos)
                    for zero, pos, neg in zip(track, track_pos, track_neg)
                ]
            )

    def asymmetries(self) -> tuple[list[float], list[list[float | None]]]:
        return (
            self._rot,
            [
                [
                    (
                        (mult.positive + mult.negative - 2 * mult.zero)
                        / (mult.positive - mult.negative)
                        if mult.zero is not None
                        and mult.positive is not None
                        and mult.negative is not None
                        else None
                    )
                    for mult, rot in zip(track, self._rot)
                ]
                for track in self._combined_tracks
            ],
        )

    def colors(self, upper: float, lower: float) -> list[float]:
        return [
            viridis((track[0].zero - lower) / (upper - lower))
            for track in self._combined_tracks
        ]


def load_modes(parity: str, m: int, limit_lower: float, limit_upper: float) -> ModeMap:
    base = f"test-data/generated/storm-asymmetries/{parity}-{m}"

    data = []

    for f_name in Path(base).iterdir():
        f = h5py.File(f_name)

        rot = f["model"].attrs["deformation-rotation-frequency"]
        # rot = float(f_name.stem.split("_")[1])
        orig_freqs = f["frequency"][:]
        freqs = f["deformation"][str(m)]["frequency"]["re"][:]
        vectors = f["deformation"][str(m)]["eigenvector"]["re"][:]
        degrees = f["degree"][:]

        degree_types = sorted(np.unique(degrees))
        norms = []

        for ell in degree_types:
            norms.append(
                np.linalg.norm(vectors[np.nonzero(degrees == ell), :], axis=1)[0]
            )

        inferred_degrees = []
        for degree_matches in zip(*norms):
            inferred_degrees.append(degree_types[np.argmax(degree_matches)])

        data.append((rot, freqs, np.array(inferred_degrees)))
        # data.append((rot, orig_freqs, degrees))

        f.close()

    data.sort(key=lambda x: x[0])
    mode_filter = (limit_lower <= data[0][1]) & (data[0][1] <= limit_upper)

    modes = ModeMap(data[0][1][mode_filter], data[0][2][mode_filter])

    for rot, freqs, degrees in data:
        modes.push_results(rot, freqs[freqs > 0], degrees)

    return modes


def model_freq_scale() -> float:
    f = h5py.File(model)

    M = f.attrs["M_star"]
    R = f.attrs["R_star"]

    return sqrt(G * M / R**3) / 2 / pi * 86400


model = "test-data/top-zams-model.GSM"
freq_scale = model_freq_scale()


def rot_input(rot: float) -> list[str]:
    return [f"set-rotation-constant {rot}", f"deform {rot}"]


def scan_input_even(order: int, rot: float) -> list[str]:
    if order == 0:
        zero = ["scan --frequency-units=cycles-per-day 0 0 2.0 30 300 --precision=1e-6"]
    else:
        zero = []

    return zero + [
        f"scan --frequency-units=cycles-per-day 2 {m} {2.0 + m * rot} {30.0 + m * rot} 300 --precision=1e-6",
        f"scan --frequency-units=cycles-per-day 4 {m} {2.0 + m * rot} {30.0 + m * rot} 300 --precision=1e-6",
        f"scan --frequency-units=cycles-per-day 6 {m} {2.0 + m * rot} {30.0 + m * rot} 300 --precision=1e-6",
        f"scan --frequency-units=cycles-per-day 2 {m} {0.8 + m * rot} {2.0 + m * rot} 300 --precision=1e-6 --inverse",
        f"scan --frequency-units=cycles-per-day 4 {m} {0.8 + m * rot} {2.0 + m * rot} 300 --precision=1e-6 --inverse",
        f"scan --frequency-units=cycles-per-day 6 {m} {0.8 + m * rot} {2.0 + m * rot} 300 --precision=1e-6 --inverse",
    ]


def scan_input_odd(order: int, rot: float) -> list[str]:
    return [
        f"scan --frequency-units=cycles-per-day 1 {m} {2.0 + m * rot} {30.0 + m * rot} 300 --precision=1e-6",
        f"scan --frequency-units=cycles-per-day 3 {m} {2.0 + m * rot} {30.0 + m * rot} 300 --precision=1e-6",
        f"scan --frequency-units=cycles-per-day 5 {m} {2.0 + m * rot} {30.0 + m * rot} 300 --precision=1e-6",
        f"scan --frequency-units=cycles-per-day 7 {m} {2.0 + m * rot} {30.0 + m * rot} 300 --precision=1e-6",
        f"scan --frequency-units=cycles-per-day 1 {m} {0.8 + m * rot} {2.0 + m * rot} 300 --precision=1e-6 --inverse",
        f"scan --frequency-units=cycles-per-day 3 {m} {0.8 + m * rot} {2.0 + m * rot} 300 --precision=1e-6 --inverse",
        f"scan --frequency-units=cycles-per-day 5 {m} {0.8 + m * rot} {2.0 + m * rot} 300 --precision=1e-6 --inverse",
        f"scan --frequency-units=cycles-per-day 7 {m} {0.8 + m * rot} {2.0 + m * rot} 300 --precision=1e-6 --inverse",
    ]


def finish_and_output(m: int, idx: int, rot: float, name: str) -> list[str]:
    return [
        "post-process",
        f"perturb-deformed {m}",
        f"output test-data/generated/storm-asymmetries/{name}-{m}/{i}_{rot}.hdf5 --frequency-units=cycles-per-day --properties frequency,degree,azimuthal-order,deformed-frequency,deformed-eigenvector,coupling-matrix --profiles radial-coordinate,y1,y2,y3,y4,xi_r,xi_h,pressure,density --model-properties dynamical-frequency,deformation-beta,deformation-dbeta,deformation-ddbeta,deformation-rotation-frequency",
    ]


def start_storm(input: list[str]) -> Popen:
    proc = Popen(["cargo", "run", "--release", "--bin=storm"], stdin=PIPE, text=True)

    def send_input():
        proc.stdin.write("\n".join(lines))
        proc.stdin.close()

    Thread(target=send_input).start()

    return proc


if len(sys.argv) == 2 and sys.argv[1] == "rerun":
    procs = []

    for m in range(-1, 2):
        output = Path(f"test-data/generated/storm-asymmetries/odd-{m}/")
        output.mkdir(parents=True, exist_ok=True)

        lines = [f"input {model}\n"]

        for i in range(101):
            rot = i / 100 * 0.2
            lines.extend(rot_input(rot))
            lines.extend(scan_input_odd(m, rot * freq_scale))
            lines.extend(finish_and_output(m, i, rot, "odd"))

        procs.append(start_storm(lines))

        output = Path(f"test-data/generated/storm-asymmetries/even-{m}/")
        output.mkdir(parents=True, exist_ok=True)

        lines = [f"input {model}\n"]

        for i in range(101):
            rot = i / 100 * 0.2
            lines.extend(rot_input(rot))
            lines.extend(scan_input_even(m, rot * freq_scale))
            lines.extend(finish_and_output(m, i, rot, "even"))

        procs.append(start_storm(lines))

    for p in procs:
        p.wait()

odd_lower = 0.1
odd_upper = 20.0
odd_zero = load_modes("odd", 0, odd_lower, odd_upper)
odd_negative = load_modes("odd", -1, odd_lower, odd_upper)
odd_positive = load_modes("odd", 1, odd_lower, odd_upper)

even_zero = load_modes("even", 0, 3.0, 20.0)
even_negative = load_modes("even", -1, 3.0, 20.0)
even_positive = load_modes("even", 1, 3.0, 20.0)

odd_multiplets = MultipletMap(1, odd_negative, odd_positive, odd_zero)
even_multiplets = MultipletMap(2, even_negative, even_positive, even_zero)

plt.figure(num="Odd modes")

for track in odd_multiplets._combined_tracks:
    plt.plot(
        np.array(odd_multiplets._rot) / freq_scale,
        list(map(lambda x: x.negative, track)),
        c="C0",
    )
    plt.plot(
        np.array(odd_multiplets._rot) / freq_scale,
        list(map(lambda x: x.zero, track)),
        c="C1",
    )
    plt.plot(
        np.array(odd_multiplets._rot) / freq_scale,
        list(map(lambda x: x.positive, track)),
        c="C2",
    )

plt.figure(num="Even modes")

for track in even_multiplets._combined_tracks:
    plt.plot(
        np.array(even_multiplets._rot) / freq_scale,
        list(map(lambda x: x.negative, track)),
        c="C0",
    )
    plt.plot(
        np.array(even_multiplets._rot) / freq_scale,
        list(map(lambda x: x.zero, track)),
        c="C1",
    )
    plt.plot(
        np.array(even_multiplets._rot) / freq_scale,
        list(map(lambda x: x.positive, track)),
        c="C2",
    )

plt.figure(num="Odd asymmetries")

rot, asymmetries = odd_multiplets.asymmetries()

for a, color in zip(asymmetries, viridis(np.linspace(0, 1, len(asymmetries)))):
    plt.plot(np.array(rot) / freq_scale, a, color=color)

plt.title(r"$\mathcal{A}_1$ asymmetry $\ell = 1$")
plt.xlabel(r"Rotation frequency [$\Omega_k$]")
plt.ylabel(r"Relative asymmetry")
plt.savefig("test-data/generated/asymmetries-odd.pdf")

plt.figure(num="Even asymmetries")

rot, asymmetries = even_multiplets.asymmetries()

for a in asymmetries:
    plt.plot(np.array(rot) / freq_scale, a, color="C0")

plt.title(r"$\mathcal{A}_1$ asymmetry $\ell = 2$")
plt.xlabel(r"Rotation frequency [$\Omega_k$]")
plt.ylabel(r"Relative asymmetry")
plt.savefig("test-data/generated/asymmetries-even.pdf")

plt.show()
