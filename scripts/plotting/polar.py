from typing import cast
from math import sqrt, ceil
import matplotlib.pyplot as plt
from matplotlib.pyplot import Figure
from matplotlib.gridspec import GridSpec, SubplotSpec
from matplotlib.projections.polar import PolarAxes
from matplotlib.colors import Normalize, Colormap, SymLogNorm
import numpy as np


def map_min_delta(r: np.ndarray, min_delta: float) -> np.ndarray:
    rmin = r[0]
    rmax = r[-1]

    new_points = np.linspace(rmin, rmax, int((rmax - rmin) / min_delta) + 1)
    idx = np.unique(np.searchsorted(r, new_points))

    return idx


# Should called before any axis specific functions are called.
# Otherwise a normal axis is created on top of the polar axis
def plot_circle_segment(
    fig: Figure,
    theta: np.ndarray,
    r: np.ndarray,
    data: np.ndarray,
    cmap: str | Colormap | None = "RdBu",
    norm: str | Normalize | None = None,
    subplot: SubplotSpec | None = None,
) -> PolarAxes:
    if subplot is None:
        ax: PolarAxes = cast(PolarAxes, fig.add_axes((0.1, 0.1, 0.8, 0.8), polar=True))
    else:
        ax = cast(PolarAxes, fig.add_subplot(subplot, polar=True))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction("clockwise")
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    if norm is None:
        data_max = np.max(np.abs(data))
        norm = SymLogNorm(linthresh=data_max * 1e-4)

    ax.pcolormesh(
        theta,
        r,
        data,
        cmap="RdBu",
        norm=norm,
    )

    return ax


def plot_mode_grid(
    modes: list[tuple[np.ndarray, float]],
    r_coord: np.ndarray,
    theta_coord: np.ndarray,
    name: str,
) -> None:
    fig = plt.figure(num=name, constrained_layout=True)
    drawn_aspect = None
    zoom_on = None

    def redraw(aspect):
        fig.clf()
        if zoom_on is not None:
            gs = GridSpec(1, 1, fig)
            data, freq = modes[zoom_on]
            plot_circle_segment(fig, theta_coord, r_coord, data, subplot=gs[0])
            fig.suptitle(f"{name}: {freq:.4}")

        else:
            nmodes = len(modes)

            nrows = ceil(sqrt(nmodes / aspect / 2))
            ncols = ceil(nmodes / nrows)

            gs = GridSpec(nrows, ncols, fig)

            fig.suptitle(name)

            for i, (data, freq) in enumerate(modes):
                ax = plot_circle_segment(fig, theta_coord, r_coord, data, subplot=gs[i])

                ax.set_title(f"{freq:.4}")

    def on_resize(event):
        nonlocal drawn_aspect

        fig = event.canvas.figure
        fig_width, fig_height = fig.get_size_inches()

        aspect = fig_width / fig_height

        if aspect != drawn_aspect and zoom_on is None:
            redraw(aspect)

        drawn_aspect = aspect

    def on_click(event):
        nonlocal zoom_on

        if zoom_on is not None:
            zoom_on = None
        elif ax := event.inaxes:
            idx = ax.figure.axes.index(ax)
            zoom_on = idx

        redraw(drawn_aspect)
        fig.canvas.draw()
        fig.canvas.flush_events()

    cid1 = fig.canvas.mpl_connect("resize_event", on_resize)
    cid2 = fig.canvas.mpl_connect("button_press_event", on_click)

    setattr(fig, "_xx_callback_holder", [cid1, cid2])
