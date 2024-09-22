import h5py
import matplotlib.pyplot as plt

data = h5py.File("test-data/generated/balanced-inspect.hdf5")

plt.plot(data["scan"]["freq"][:], data["scan"]["value"][:])

for k in data["sol"].keys():
    evals = data["sol"][k].attrs["evals"]
    for j in range(evals):
        j = f"eval_{j}"
        d = data["sol"][k][j]

        lims = [d.attrs["next"], d.attrs["lower"], d.attrs["upper"]]

        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        ax1.plot(d["freq"][:], d["value"][:])
        ax1.axvline(d.attrs["next"], label="next")
        ax1.axvline(d.attrs["lower"], c="red", ls="--", label="lower")
        ax1.axvline(d.attrs["upper"], c="green", ls="-.", label="upper")
        if "previous" in d.attrs:
            ax1.axvline(d.attrs["previous"], c="pink", ls="-.", label="previous")
            lims.append(d.attrs["previous"])
        ax1.axhline(0, ls="--")
        ax1.text
        ax1.legend()

        delta = max(lims) - min(lims)

        ax1.set_xlim((min(lims) - delta * 0.1, max(lims) + delta * 0.1))

        ax2.plot(d["freq"][:-1], (d["value"][1:] - d["value"][:-1]) / (d["freq"][1:] - d["freq"][:-1]))

plt.show()
