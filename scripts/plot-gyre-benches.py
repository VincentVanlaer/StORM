import matplotlib.pyplot as plt
import json
import numpy as np

DIFF_SCHEMES = [ "COLLOC_GL2", "COLLOC_GL4", "COLLOC_GL6", "MAGNUS_GL2", "MAGNUS_GL4", "MAGNUS_GL6"]
STORM_SCHEMES = [ "colloc2", "colloc4", "colloc6", "magnus2", "magnus4", "magnus6" ]

results_block = []
results_band = []
results_storm = []

for d in DIFF_SCHEMES:
    block = json.load(open(f"test-data/generated/{d}_BLOCK.json"))["results"][0]["mean"]
    band = json.load(open(f"test-data/generated/{d}_BAND.json"))["results"][0]["mean"]

    results_block.append(block)
    results_band.append(band)

for d in STORM_SCHEMES:
    try:
        storm = json.load(open(f"test-data/generated/{d}_storm.json"))["results"][0]["mean"]
    except FileNotFoundError:
        storm = np.nan

    results_storm.append(storm)

sets = {"GYRE block method": results_block, "GYRE band method": results_band, "StORM": results_storm}

x = np.arange(len(DIFF_SCHEMES))
width = 0.3

plt.figure(layout="constrained")

for i, (attribute, measurement) in enumerate(sets.items()):
    offset = 0.3 * i
    rects = plt.bar(x + offset, measurement, 0.3, label=attribute)

plt.ylabel("Runtime [s], lower is better")
plt.xticks(x + width, DIFF_SCHEMES, rotation=45)
plt.legend()
plt.title("Comparison of GYRE and StORM performance for various difference schemes. The stellar model is a 6200 point MESA model of a beta-Cepheid star (HD 192575). Input scan parameters are 1 to 25 dimless angular frequency with 25 scan points, radial modes. In total, 19 frequencies are extracted up to machine precision. GYRE is using the Brent bracketing algorithm, while StORM is using the custom Balanced algorithm. All regridding has been disabled, and GYRE has been forced to use the full set of equations, rather than the second-order reduced equations. Ran on a single core of an i7-1185G7.", wrap=True, fontsize=8, pad=10, loc='left')
plt.savefig("test-data/generated/gyre-performance.pdf")
