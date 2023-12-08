import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import matplotlib as mpl

def set_style():
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
data = np.loadtxt("a.csv", delimiter=',')

plt.plot(data[:, 0], np.abs(data[:, 1:]), label=["GL2", "GL4", "GL6", "GL8"])
plt.yscale("log")
plt.xscale("log")
plt.legend()
plt.show()
