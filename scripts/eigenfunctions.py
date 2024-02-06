# storm --input test-data/Z0.016_M11.50_logD3.00_aov0.000_fov0.020_mn2186-MS.GSM --output scripts/eigenfunctions.hdf5 --lower 4 --upper 4.5 --n-steps 2 --degree 1 --overlay-rot test-data/overlay.GSM

import matplotlib.pyplot as plt
import h5py

f = h5py.File("scripts/eigenfunctions.hdf5")
f2 = h5py.File("test-data/gyre-reference.hdf5")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

print(f["00000_00"].attrs["freq"])
print(f2.attrs["freq"]["re"])

ax1.set_title("$Y_1$")
ax1.plot(f["x"][:], -f["00000_00"]["y1"][:], label="StORM")
ax1.plot([], [], label="GYRE")
ax1b = ax1.twinx()
ax1b.plot(f2["x"][:], f2["y_1"]["re"][:], c="C1", label="GYRE")
ax1.legend()

ax2.set_title("$Y_2$")
ax2.plot(f["x"][:], -f["00000_00"]["y2"][:])
ax2b = ax2.twinx()
ax2b.plot(f2["x"][:], f2["y_2"]["re"][:], c="C1")

ax3.set_title("$Y_3$")
ax3.plot(f["x"][:], -f["00000_00"]["y3"][:])
ax3b = ax3.twinx()
ax3b.plot(f2["x"][:], f2["y_3"]["re"][:], c="C1")

ax4.set_title("$Y_4$")
ax4.plot(f["x"][:], -f["00000_00"]["y4"][:])
ax4b = ax4.twinx()
ax4b.plot(f2["x"][:], f2["y_4"]["re"][:], c="C1")

f.close()
f2.close()

plt.show()
