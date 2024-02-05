import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 8,
            "figure.titlesize": 10,
            "figure.labelsize": 10,
            "lines.linewidth": 3,
            "lines.markersize": 6,
            "axes.linewidth": 0.25,
            "xtick.major.width": 0.25,
            "ytick.major.width": 0.25,
            "axes.prop_cycle": cycler(
                "color",
                [
                    "#006BA4",
                    "#FF800E",
                    "#ABABAB",
                    "#595959",
                    "#5F9ED1",
                    "#C85200",
                    "#898989",
                ],
            )
            + cycler("linestyle", ["-", "--", ":", "-.", "-", "--", ":"])
            + cycler("markersize", [3, 4, 4, 4, 3, 4, 4]),
            "axes.formatter.limits": (-5, 5),
            "patch.facecolor": "006BA4",
            "figure.constrained_layout.use": True,
            "pgf.texsystem": "lualatex",
            "text.usetex": True,
            "pgf.rcfonts": False,
        }
    )


set_style()

for test_target in Path("target/criterion").iterdir():
    if test_target.name == "report":
        continue

    data = defaultdict(list)

    for result in test_target.glob("*/*/result"):
        error = float(result.read_text())
        npoints = float(result.parent.name)
        method = result.parent.parent.name
        timings = np.genfromtxt(
            result.parent / "new" / "raw.csv", delimiter=",", names=True
        )
        time = timings[-1]["sample_measured_value"] / timings[-1]["iteration_count"]

        data[method].append((npoints, error, time))

    plt.figure(test_target.name)

    for k, v in data.items():
        sort = tuple(zip(*sorted(v)))
        plt.plot(sort[2], np.abs(sort[1]), label=k)

    for i in range(1, 11):
        plt.axhline(i * sys.float_info.epsilon * 108.40719198544681, ls="--", lw=0.5)

    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
plt.show()
