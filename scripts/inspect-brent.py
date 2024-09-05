import h5py
import matplotlib.pyplot as plt

data = h5py.File("test-data/generated/brent-inspect.hdf5")

plt.plot(data["scan"]["freq"][:], data["scan"]["value"][:])

for k in data["sol"].keys():
    evals = data["sol"][k].attrs["evals"]
    for j in range(evals):
        j = f"eval_{j}"
        d = data["sol"][k][j]
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax1.plot(d["freq"][:], d["value"][:])
        ax1
        ax1.axvline(d.attrs["next"], label=f"next ({d.attrs['method']})")
        ax1.axvline(d.attrs["current"], c="red", ls="--", label="current")
        ax1.axvline(d.attrs["previous"], c="green", ls="-.", label="previous")
        ax1.axvline(d.attrs["counterpoint"], c="pink", ls="--", label="counterpoint")
        ax1.axhline(0, ls="--")
        ax1.text
        ax1.legend()

        lims = [d.attrs["next"], d.attrs["previous"], d.attrs["current"], d.attrs["counterpoint"]]

        delta = max(lims) - min(lims)

        ax1.set_xlim((min(lims) - delta * 0.1, max(lims) + delta * 0.1))

        ax2.plot(d["freq"][:-1], (d["value"][1:] - d["value"][:-1]) / (d["freq"][1:] - d["freq"][:-1]))

plt.show()
