import numpy as np
import matplotlib.pyplot as plt

from chua_generator import ChuaGenerator

# ======================================================
# Create generator
# ======================================================

chua = ChuaGenerator(
    alpha=9.0,
    beta=14.285714,
    m0=-8/7,
    m1=-5/7,
    dt=0.001,
    T=100
)

sigma_range = np.arange(
    0,
    0.2,
    0.01
)

# ======================================================
# Benettin LLE
# ======================================================

def compute_LLE(sigma):

    N = chua.N

    transient = 100

    d0 = 1e-8

    X1 = np.array([
        0.1,
        0,
        0
    ])

    X2 = X1 + np.array([
        d0,
        0,
        0
    ])

    S = 0
    count = 0

    sqrt_dt = np.sqrt(chua.dt)

    for n in range(N):

        noise = sigma*np.random.randn()*sqrt_dt

        X1 = chua.rk4_step(X1, noise)
        X2 = chua.rk4_step(X2, noise)

        if n > transient:

            d = np.linalg.norm(
                X2-X1
            )

            if d == 0:
                continue

            S += np.log(d/d0)

            count += 1

            direction = (X2-X1)/d

            X2 = X1 + d0*direction

    return S/(count*chua.dt)

# ======================================================
# Compute
# ======================================================

LLE = []

for sigma in sigma_range:

    print(
        f"sigma = {sigma:.3f}"
    )

    LLE.append(
        compute_LLE(sigma)
    )

# ======================================================
# Plot
# ======================================================

plt.figure(figsize=(8,6))

plt.plot(
    sigma_range,
    LLE,
    'o-'
)

plt.axhline(
    0,
    linestyle='--'
)

plt.xlabel("Noise intensity σ")

plt.ylabel("Largest Lyapunov Exponent")

plt.title("Benettin Method")

plt.grid(True)

plt.tight_layout()

plt.show()
