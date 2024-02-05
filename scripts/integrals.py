from sympy import expand, symbols, Integer, init_printing, pprint

init_printing()

one_half = Integer(1) / Integer(2)
x = symbols("x")


def solution(n: int):
    return expand(
        ((-x + one_half) ** (n + 2) - (-x - one_half) ** (n + 2)) / (n + 2)
        + x * ((-x + one_half) ** (n + 1) - (-x - one_half) ** (n + 1)) / (n + 1)
    ), expand(((-x + one_half) ** (n + 1) - (-x - one_half) ** (n + 1)) / (n + 1))


r = []

for n in range(4):
    s, i = solution(n)

    r.append([s, i])

pprint(r)
