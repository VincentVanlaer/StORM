from math import pi, sqrt
from pathlib import Path
import h5py

G = 6.67430e-8
model = "test-data/joey-test-model.GSM"

f = h5py.File(model)

rot_crit = sqrt(G * f.attrs["M_star"] / f.attrs["R_star"]**3)

f.close()

for m in range(-1, 2):
    lines = [
        f"input {model}\n"
    ]

    for i in range(1001):
        rot = i / 1000 * 0.20 * rot_crit
        lines.extend([
            f"set-rotation-constant {rot}\n",
            f"scan 1 {m} 1.0 30 300 1e-6\n",
            f"scan 1 {m} 0.5 1. 100 1e-6 --inverse\n",
            f"scan 3 {m} 1.0 30 300 1e-6\n",
            f"scan 3 {m} 0.5 1. 100 1e-6 --inverse\n",
            f"scan 5 {m} 1.0 30 300 1e-6\n",
            f"scan 5 {m} 0.5 1. 100 1e-6 --inverse\n",
            f"scan 7 {m} 1.0 30 300 1e-6\n",
            f"scan 7 {m} 0.5 1. 100 1e-6 --inverse\n",
            "post-process\n",
            f"deform {rot}\n",
            f"perturb-deformed {m}\n",
            f"output test-data/generated/deformed-odd-{m}/{i}_{rot}.hdf5\n",
        ])

    Path(
        f"test-data/generated/deformed-odd-{m}").mkdir(parents=True, exist_ok=True)

    with open(f"test-data/generated/deformed-odd-{m}/input", "w") as f:
        f.writelines(lines)

for m in range(-1, 2):
    lines = [
        f"input {model}\n"
    ]

    for i in range(1001):
        rot = i / 1000 * 0.20 * rot_crit
        lines.extend([
            f"set-rotation-constant {rot}\n",
            f"scan 0 {m} 1.0 30 300 1e-6\n",
            f"scan 2 {m} 1.0 30 300 1e-6\n",
            f"scan 2 {m} 0.5 1. 100 1e-6 --inverse\n",
            f"scan 4 {m} 1.0 30 300 1e-6\n",
            f"scan 4 {m} 0.5 1. 100 1e-6 --inverse\n",
            f"scan 6 {m} 1.0 30 300 1e-6\n",
            f"scan 6 {m} 0.5 1. 100 1e-6 --inverse\n",
            "post-process\n",
            f"deform {rot}\n",
            f"perturb-deformed {m}\n",
            f"output test-data/generated/deformed-even-{m}/{i}_{rot}.hdf5\n",
        ])

    Path(
        f"test-data/generated/deformed-even-{m}").mkdir(parents=True, exist_ok=True)
    with open(f"test-data/generated/deformed-even-{m}/input", "w") as f:
        f.writelines(lines)
