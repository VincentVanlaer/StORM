from math import pi
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import h5py

f = h5py.File("test-data/generated/test.hdf5")


def angle_diff(a):
    a, b = a
    diff = a - b

    if diff > pi:
        diff = diff - 2.0 * pi
    elif diff < -pi:
        diff = diff + 2.0 * pi

    return diff


winding_x = []
winding_y = []


for k in f.keys():
    if k == "x":
        continue

    y1 = f[k]["y1"][:]
    y2 = f[k]["y2"][:]
    y3 = f[k]["y3"][:]
    y4 = f[k]["y4"][:]
    xi_h = f[k]["xi_h"][:]
    xi_r = f[k]["xi_r"][:]

# for (k, f) in enumerate(Path("test-data/generated/l1-mode-id").glob("mode-*.h5")):
#     f = h5py.File(f)
# 
#     y1 = f["y_1"]["re"][1:]
#     y2 = f["y_2"]["re"][1:]
#     y3 = f["y_3"]["re"][1:]
#     y4 = f["y_4"]["re"][1:]
#     xi_r = f["xi_r"]["re"][1:]
#     xi_h = f["xi_h"]["re"][1:]
#
#     f.close()

    a = xi_h[1:]
    b = xi_r[1:]

    arg = np.arctan2(a, b)

    diff = list(map(angle_diff, zip(arg[1:], arg[:-1])))

    argp = 0
    argg = 0
    argg_idx = [False] * len(arg)

    zerop = 0
    zerog = 0

    for i in range(len(arg) - 1):
        if b[i] * b[i + 1] <= 0:
            if (b[i] < 0 and a[i] > 0) or (b[i] > 0 and a[i] < 0):
                zerog += 1
            else:
                zerop += 1

        if diff[i] > 0:
            argp += diff[i]
        else:
            argg -= diff[i]
            argg_idx[i] = True

    if int(k) in (17, 18, 19,):
        fig = plt.figure(layout="constrained")

        gs = GridSpec(2, 3, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(a, b)
        ax1.scatter(a[argg_idx], b[argg_idx], c="red")

        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(y1)
        ax2.plot(y2)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(y3)
        ax3.plot(y4)
        ax3 = fig.add_subplot(gs[1, 2])
        ax3.plot(xi_r)
        ax3.plot(xi_h)

    winding_x.append(argp / pi)
    winding_y.append(argg / pi)

    diff = np.nan_to_num(diff)

    print(
        "{}: {:+.5f}, {:+.5f}, {:.5f}, {:.5f}, {:.5f}, {}, {}".format(
            int(k) + 1,
            round(sum(diff) / pi),
            sum(diff) / pi,
            argp / pi,
            argg / pi,
            argp / pi + argg / pi,
            zerop,
            zerog,
        )
    )

f.close()


plt.figure()

plt.scatter(winding_x, winding_y)
plt.axvline(0)
plt.axhline(0)

plt.figure()
plt.plot(np.diff(np.array(winding_x) - np.array(winding_y)), ".")

plt.show()
