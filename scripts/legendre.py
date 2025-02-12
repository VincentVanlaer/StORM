from scipy.special import lpmv
import numpy as np


def legendre(theta: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    res = {}

    x = np.cos(theta)

    for ell in range(0, 11):
        for m in range(-1, 2):
            if abs(m) > ell:
                continue

            res[(ell, m)] = lpmv(m, ell, x)

    return res
