import numpy as np
import matplotlib.pyplot as plt
from scipy.special import airy
from scipy.optimize import root_scalar


def inner_argument_general(c: complex, x: complex, b: complex) -> complex:
    return b * (c * x - 0.55) / ((b * c) ** (2 / 3))


def evaluate(b: complex) -> complex:
    b = b**2
    c = 0.1 + 0j
    l1 = -0.55 * b / (b * c) ** (2 / 3)

    ai0, aip0, bi0, bip0 = airy(l1)
    ai1, aip1, bi1, bip1 = airy(inner_argument_general(c, 0.5, b))

    f_eval_mid = (bi0 * ai1 - ai0 * bi1).real
    f_eval_mid_p = (l1 * c * (bi0 * aip1 - ai0 * bip1)).real
    return f_eval_mid * f_eval_mid_p


x = np.linspace(0, 2000, 10000)
plt.plot(x, evaluate(x))
plt.show()

print(root_scalar(evaluate, bracket=(4, 5), rtol=1e-15))
print(root_scalar(evaluate, bracket=(106, 111), rtol=1e-15))
print(root_scalar(evaluate, bracket=(159, 161), rtol=1e-15))
