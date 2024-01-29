from sympy import (
    symbols,
    solve,
    Derivative,
    Function,
    pprint,
    sqrt,
    pi,
)


# Helpers
def ensure_equal(term1, term2):
    if (term1 - term2).simplify() != 0:
        print("The following terms should be equal but aren't:")
        pprint(term1)
        pprint(term2)
        print("The difference between the terms is")
        pprint((term2 - term1).simplify())

        assert False


def get_deriv_terms(eq, d):
    dsym = symbols(f"d{d}")
    collected = (
        (solve(eq.simplify().subs(Derivative(d, x), dsym), dsym)[0] * x)
        .expand()
        .collect([y1, y2, y3, y4])
    )

    y1_term = collected.coeff(y1, 1).simplify()
    y2_term = collected.coeff(y2, 1).simplify()
    y3_term = collected.coeff(y3, 1).simplify()
    y4_term = collected.coeff(y4, 1).simplify()

    ensure_equal(collected, y1_term * y1 + y2_term * y2 + y3_term * y3 + y4_term * y4)

    return [y1_term, y2_term, y3_term, y4_term]


# Input
(
    x,  # fractional stellar coordinate
    R,  # stellar model radius
    M,  # stellar model mass
    G,  # Newton's gravitional constant
    l,  # Spherical degree
    m,  # Azimuthal order
    omega,  # Dimensionless inertial frequency
) = symbols("x R M G l m omega")

p = Function("p")(x)  # dimensional pressure
rho = Function("rho")(x)  # dimensional density
gamma1 = Function("Gamma1")(x)  # first adiabatic exponent
Omega = Function("Omega")(x)  # dimensionless rotation frequency

# Derived properties
scale = sqrt(M * G / R**3)  # Dimensionless to dimensional frequency
sigma = scale * omega  # Dimensional frequency
sOmega = scale * Omega  # Dimensional rotation frequency
rsigma = sigma - m * sOmega  # Dimensional frequency in pseudo-corotating frame
r = x * R  # radial coordinate
a_star = 1 / gamma1 * x / p * Derivative(p, x) - x / rho * Derivative(
    rho, x
)  # dimensionless A*
dphi = -Derivative(p, x) / R / rho  # dimensional gravity (hydrostatic support)
Mr = r**2 * dphi / G  # dimensional inner mass
ddp = (
    -Derivative(rho, x) / R * dphi
    + 2 * rho / r**3 * G * Mr
    - rho * G / r**2 * 4 * pi * r**2 * rho
)  # expanded form of d^2(p)/dx^2

# Dimensionless structure coefficients
V = -x / p * Derivative(p, x)
U = 4 * pi * r**2 * rho * (r / Mr)
c1 = (r**3 / R**3) * (M / Mr)

# Solution variables
y1 = Function("y1")(x)
y2 = Function("y2")(x)
y3 = Function("y3")(x)
y4 = Function("y4")(x)

# Transformed solution variables
xi_r = y1 * r * x ** (l - 2)
p_prime = y2 * rho * dphi * r * x ** (l - 2)
phi_prime = dphi * r * x ** (l - 2) * y3
dphi_prime = dphi * x ** (l - 2) * y4

# Adiabatic approximation
rho_prime = rho / (gamma1 * p) * p_prime + xi_r * a_star * rho / r

# EoM horizontal
xi_h = Function("xi_h")(x)
xi_h = solve(
    -1 / r * (p_prime + rho * phi_prime)
    + rsigma**2 * rho * xi_h
    + 2 * rho * m * rsigma * sOmega * (xi_r + xi_h),
    xi_h,
)[0]

# Mass conservation (y1)
eq = (
    rho_prime
    + 1 / r**2 * Derivative(rho * r**2 * xi_r, x) / R
    - l * (l + 1) / r * rho * xi_h
)

y1_term, y2_term, y3_term, y4_term = get_deriv_terms(eq, y1)

ensure_equal(
    V / gamma1 - 1 - l - 2 * Omega * m * l * (l + 1) / (omega + m * Omega), y1_term
)
ensure_equal(
    l * (l + 1) / (omega**2 - m**2 * Omega**2) / c1 - V / gamma1, y2_term
)
ensure_equal(l * (l + 1) / (omega**2 - m**2 * Omega**2) / c1, y3_term)
ensure_equal(0, y4_term)

# Vertical EoM (y2)
eq = (
    rsigma**2 * rho * xi_r
    - Derivative(p_prime, x) / R
    - rho_prime * dphi
    - rho * dphi_prime
)

y1_term, y2_term, y3_term, y4_term = get_deriv_terms(eq, y2)

ensure_equal(c1 * (omega - m * Omega) ** 2 - a_star, y1_term)
ensure_equal(3 - U + a_star - l, y2_term.subs(Derivative(p, x, x), ddp * R**2))
ensure_equal(0, y3_term)
ensure_equal(-1, y4_term)

# Gravity (y3)
eq = Derivative(phi_prime, x) / R * x ** (2 - l) / dphi - y4

y1_term, y2_term, y3_term, y4_term = get_deriv_terms(eq, y3)

ensure_equal(0, y1_term)
ensure_equal(0, y2_term)
ensure_equal(3 - U - l, y3_term.subs(Derivative(p, x, x), ddp * R**2))
ensure_equal(1, y4_term)

# Gravity (y4)
eq = (
    1 / r**2 * Derivative(r**2 * dphi_prime, x) / R
    - l * (l + 1) / r**2 * phi_prime
    - 4 * pi * G * rho_prime
)

y1_term, y2_term, y3_term, y4_term = get_deriv_terms(eq, y4)

ensure_equal(V / gamma1 * U, y2_term)
ensure_equal(a_star * U, y1_term)
ensure_equal(l * (l + 1), y3_term)
ensure_equal(-U - l + 2, y4_term.subs(Derivative(p, x, x), ddp * R**2))
