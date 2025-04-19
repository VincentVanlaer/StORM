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
    "MAGNUS_GL2",
    "MAGNUS_GL4",
    "MAGNUS_GL6",
    "MAGNUS_GL8",
]
STORM_SCHEMES = [
    "colloc2",
    "colloc4",
    "colloc6",
    "magnus2",
    "magnus4",
    "magnus6",
    "magnus8",
]

results_band = []
results_storm = []

freqs = {}
freqs_storm = {}

for d in DIFF_SCHEMES:
    try:
        band = json.load(open(f"test-data/generated/{d}_BAND.json"))["results"][0][
            "min"
        ]
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        band = np.nan

    if not math.isnan(band):
        freq_band = h5py.File(f"test-data/generated/{d}_BAND/summary.h5")["freq"]["re"][
            :
        ]
        freqs[f"{d}_BAND"] = freq_band

    results_band.append(band)

for d in STORM_SCHEMES:
    try:
        storm = json.load(open(f"test-data/generated/{d}_storm.json"))["results"][0][
            "min"
        ]
        storm_summary = h5py.File(f"test-data/generated/{d}_storm/summary.hdf5")

        freqs_storm[f"{d}_storm"] = np.array(sorted(storm_summary["frequency"]))
        storm_summary.close()

    except FileNotFoundError:
        storm = np.nan

    results_storm.append(storm)

sets = {"GYRE band method": results_band, "StORM": results_storm}

x = np.arange(len(DIFF_SCHEMES))
width = 0.4

plt.figure(layout="constrained")

for i, (attribute, measurement) in enumerate(sets.items()):
    offset = width * i
    rects = plt.bar(x + offset, measurement, 0.3, label=attribute)

plt.ylabel("Runtime [s], lower is better")
plt.xticks(x + width, DIFF_SCHEMES, rotation=45)
plt.legend()
plt.title(
    "Comparison of GYRE and StORM performance for various difference schemes. The stellar model is a 6200 point MESA model of a beta-Cepheid star (HD 192575). Input scan parameters are 1 to 25 dimless angular frequency with 25 scan points, radial modes. In total, 19 frequencies are extracted up to machine precision. GYRE is using the Brent bracketing algorithm, while StORM is using the custom Balanced algorithm. All regridding has been disabled, and GYRE has been forced to use the full set of equations, rather than the second-order reduced equations. Ran on a single core of an i7-1185G7.",
    wrap=True,
    fontsize=8,
    pad=10,
    loc="left",
)
plt.yscale("log")
plt.savefig("test-data/generated/gyre-performance.pdf")

plt.figure()

ref = freqs.pop("COLLOC_GL6_BAND")

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
plt.savefig("test-data/generated/gyre-compare.pdf")
plt.show()
