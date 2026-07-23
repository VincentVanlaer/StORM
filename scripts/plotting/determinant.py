import sys
from pathlib import Path
from subprocess import run
import h5py
import matplotlib.pyplot as plt

model = "test-model-zams.GSM"

if len(sys.argv) == 2 and sys.argv[1] == "rerun":
    run(
        [
            "cargo",
            "run",
            "--release",
            "--bin=storm-sweep-determinant",
            f"test-data/{model}",
            "1",
            "0",
            "0.001",
            "1",
            "10000",
            "test-data/generated/determinant.hdf5",
            "--inverse",
            "--resample=4",
        ],
        check=True,
    )


f = h5py.File("test-data/generated/determinant.hdf5")

plt.plot(f["Colloc2"][:])
plt.yscale("symlog", linthresh=1)
plt.figure()
plt.plot(f["Colloc4"][:])
plt.yscale("symlog", linthresh=1)
plt.figure()
plt.plot(f["Colloc6"][:])
plt.yscale("symlog", linthresh=1)
plt.figure()
plt.plot(f["Colloc8"][:])
plt.yscale("symlog", linthresh=1)

f.close()
plt.show()
