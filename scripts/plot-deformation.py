from math import inf, sqrt
from tqdm import tqdm
from dataclasses import dataclass
from collections.abc import Iterator, Callable
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, Normalize
import h5py
import numpy as np
from pathlib import Path

G = 6.67430e-8


class ModeMap:
    def __init__(self, match_distance: float, rot_mul: float) -> None:
        self._match_distance = match_distance
        self._mode_track: list[list[float | None]] = []
        self._rotation_frequencies: list[float] = []
        self._rot_mul = rot_mul

    @staticmethod
    def _get_last_not_none(l: list[float | None]) -> float:
        for idx in range(len(l) - 1, -1, -1):
            val = l[idx]
            if val is not None:
                return val

        raise Exception("List only contains None")

    def push_results(
        self, rotation_frequency: float, mode_frequencies: list[float]
    ) -> None:
        available_tracks = self._mode_track.copy()

        available_track_vals = [
            self._get_last_not_none(track) for track in available_tracks
        ]

        try:
            prev_rot = self._rotation_frequencies[-1]
        except IndexError:
            prev_rot = 0

        while mode_frequencies and available_tracks:
            closest_distance = inf
            closest_mode_track = 0
            closest_mode_idx = 0

            for j, mode in enumerate(mode_frequencies):
                for i, mode_track in enumerate(available_track_vals):
                    val = abs(mode_track - (
                        mode - self._rot_mul * rotation_frequency))
                    if closest_distance > val:
                        closest_distance = val
                        closest_mode_track = i
                        closest_mode_idx = j

            if closest_distance > self._match_distance:
                break

            available_tracks[closest_mode_track].append(
                mode_frequencies[closest_mode_idx] -
                self._rot_mul * rotation_frequency
            )

            del available_tracks[closest_mode_track]
            del available_track_vals[closest_mode_track]
            del mode_frequencies[closest_mode_idx]

        for track in available_tracks:
            track.append(None)

        for mode in mode_frequencies:
            self._mode_track.append(
                [None] * len(self._rotation_frequencies) + [mode - self._rot_mul * rotation_frequency
                                                            ])

        self._rotation_frequencies.append(rotation_frequency)


@dataclass
class Multiplet:
    negative: float | None
    zero: float | None
    positive: float | None


class MultipletMap:

    def __init__(
        self, filter: Callable[[float], bool], negative: ModeMap, positive: ModeMap, zero: ModeMap,
    ) -> None:
        assert negative._rotation_frequencies == positive._rotation_frequencies
        assert negative._rotation_frequencies == zero._rotation_frequencies

        self._combined_tracks = []
        self._rot = zero._rotation_frequencies.copy()

        for track in zero._mode_track:
            if track[0] is None or not filter(track[0]):
                continue

            for track_pos in positive._mode_track:
                if track_pos[0] == track[0]:
                    break
            else:
                continue

            for track_neg in negative._mode_track:
                if track_neg[0] == track[0]:
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
                        / (mult.positive - mult.negative + 2 / freq_scale * rot)
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


def load_modes(files: Iterator[Path], m: int) -> ModeMap:
    modes = ModeMap(0.02, m / freq_scale)

    for f_name in tqdm(list(files)):
        f = h5py.File(f_name)

        rot = float(f_name.stem.split("_")[1])

        freqs = f["perturbations"][str(m)]["frequencies"]["re"]

        modes.push_results(rot, [freq for freq in freqs if freq > 0])

    return modes


def collect_ells(f_name: Path) -> dict[int, list[float]]:
    f = h5py.File(f_name)

    ells = defaultdict(list)

    for sol in f["solutions"].values():
        ells[sol.attrs["ell"]].append(sol.attrs["freq"])

    return ells


def closest_ell(zero_rot_freqs: dict[int, list[float]], freq_to_match: float) -> int:
    distance = inf
    closest_ell = 0

    for ell, freq_list in zero_rot_freqs.items():
        for freq in freq_list:
            if distance > abs(freq - freq_to_match):
                distance = abs(freq - freq_to_match)
                closest_ell = ell

    return closest_ell


def model_freq_scale() -> float:
    f = h5py.File("test-data/joey-test-model.GSM")

    M = f.attrs["M_star"]
    R = f.attrs["R_star"]

    return sqrt(G * M / R**3)


freq_scale = model_freq_scale()

odd_zero_rot = collect_ells(next(
    Path("test-data/generated/").glob("deformed-odd-0/0_*.hdf5")))
even_zero_rot = collect_ells(next(
    Path("test-data/generated/").glob("deformed-even-0/0_*.hdf5")))


odd_negative = load_modes(
    Path("test-data/generated/").glob("deformed-odd--1/*.hdf5"), -1)
odd_zero = load_modes(
    Path("test-data/generated/").glob("deformed-odd-0/*.hdf5"), 0)
odd_positive = load_modes(
    Path("test-data/generated/").glob("deformed-odd-1/*.hdf5"), 1)
even_negative = load_modes(
    Path("test-data/generated/").glob("deformed-even--1/*.hdf5"), -1)
even_zero = load_modes(
    Path("test-data/generated/").glob("deformed-even-0/*.hdf5"), 0)
even_positive = load_modes(
    Path("test-data/generated/").glob("deformed-even-1/*.hdf5"), 1)


odd_multiplets = MultipletMap(
    lambda x: 0.5 < x < 10 and closest_ell(odd_zero_rot, x) == 1, odd_negative, odd_positive, odd_zero)
even_multiplets = MultipletMap(
    lambda x: 0.5 < x < 10 and closest_ell(even_zero_rot, x) == 2, even_negative, even_positive, even_zero)

plt.figure(num="Odd modes")

for track in odd_negative._mode_track:
    plt.plot(np.array(odd_negative._rotation_frequencies) /
             freq_scale, track, c="C0")

for track in odd_zero._mode_track:
    plt.plot(np.array(odd_zero._rotation_frequencies) /
             freq_scale, track, c="C1")

for track in odd_positive._mode_track:
    plt.plot(np.array(odd_positive._rotation_frequencies) /
             freq_scale, track, c="C2")

plt.figure(num="Even modes")

for track in even_negative._mode_track:
    plt.plot(np.array(even_negative._rotation_frequencies) /
             freq_scale, track, c="C0")

for track in even_zero._mode_track:
    plt.plot(np.array(even_zero._rotation_frequencies) /
             freq_scale, track, c="C1")

for track in even_positive._mode_track:
    plt.plot(np.array(even_positive._rotation_frequencies) /
             freq_scale, track, c="C2")

plt.figure(num="Odd asymmetries")

rot, asymmetries = odd_multiplets.asymmetries()

for a in asymmetries:
    plt.plot(np.array(rot) / freq_scale, a, color="C0")

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
