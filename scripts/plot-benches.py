import matplotlib.pyplot as plt
import h5py
import json
import numpy as np
import math

plt.rcParams["figure.constrained_layout.use"] = True

DIFF_SCHEMES = [
    "COLLOC_GL2",
    "COLLOC_GL4",
    "COLLOC_GL6",
]
STORM_SCHEMES = [
    "colloc2",
    "colloc4",
    "colloc6",
]

results_gyre9 = []
results_gyre8 = []
results_storm = []

freqs = {}
freqs_storm = {}

for d in DIFF_SCHEMES:
    try:
        band = json.load(open(f"test-data/generated/storm-gyre/{d}_gyre9.json"))[
            "results"
        ][0]["min"]
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        band = np.nan

    if not math.isnan(band):
        freq_band = h5py.File(f"test-data/generated/storm-gyre/{d}_gyre9/summary.h5")[
            "freq"
        ]["re"][:]
        freqs[f"{d}_BAND"] = freq_band

    results_gyre9.append(band)

for d in DIFF_SCHEMES:
    try:
        band = json.load(open(f"test-data/generated/storm-gyre/{d}_gyre8.json"))[
            "results"
        ][0]["min"]
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        band = np.nan

    if not math.isnan(band):
        freq_band = h5py.File(f"test-data/generated/storm-gyre/{d}_gyre8/summary.h5")[
            "freq"
        ]["re"][:]
        freqs[f"{d}_BAND"] = freq_band

    results_gyre8.append(band)

for d in STORM_SCHEMES:
    try:
        storm = json.load(open(f"test-data/generated/storm-gyre/{d}_storm.json"))[
            "results"
        ][0]["min"]
        storm_summary = h5py.File(
            f"test-data/generated/storm-gyre/{d}_storm/summary.hdf5"
        )

        freqs_storm[f"{d}_storm"] = np.array(sorted(storm_summary["frequency"]))
        storm_summary.close()

    except FileNotFoundError:
        storm = np.nan

    results_storm.append(storm)

sets = {"GYRE 8.1": results_gyre8, "GYRE 9.0": results_gyre9, "StORM": results_storm}

x = np.arange(len(DIFF_SCHEMES))
width = 0.3

plt.figure(layout="constrained")

for i, (attribute, measurement) in enumerate(sets.items()):
    offset = width * i
    rects = plt.bar(x + offset, measurement, 0.3, label=attribute)

plt.ylabel("Runtime [s], lower is better")
plt.xticks(x + width, DIFF_SCHEMES, rotation=45)
plt.legend()
plt.title(
    "Comparison of GYRE and StORM performance for various difference schemes. The stellar model is a 6200 point MESA model of a beta-Cepheid star (HD 192575). Input scan parameters are 1 to 25 dimless angular frequency with 25 scan points, dipole modes. In total, 19 frequencies are extracted up to machine precision. All regridding has been disabled. Ran on a single core of an i7-1185G7.",
    wrap=True,
    fontsize=8,
    pad=10,
    loc="left",
)
plt.yscale("log")
plt.savefig("test-data/generated/storm-gyre/gyre-performance.pdf")

plt.figure()

ref = freqs["COLLOC_GL6_BAND"]

for i, (k, v) in enumerate(freqs.items()):
    if len(ref) != len(v):
        print(f"Invalid output for {k}")
        plt.plot([], [], label=k)
        continue
    plt.plot(v + i / 10, v / ref - 1.0, ".", label=k)

plt.gca().set_prop_cycle(None)

for i, (k, v) in enumerate(freqs_storm.items()):
    if len(ref) != len(v):
        print(f"Invalid output for {k}")
        plt.plot([], [], label=k)
        continue
    plt.plot(v + i / 10, v / ref - 1.0, "x", label=k, markersize=4)

plt.xlabel("Frequency (dimensionless)")
plt.ylabel("Relative difference with GYRE COLLOC_GL6")
plt.legend()
plt.savefig("test-data/generated/storm-gyre/gyre-compare.pdf")
plt.show()
