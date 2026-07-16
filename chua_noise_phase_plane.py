# ==========================================================
# phase_plane.py
# ==========================================================

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from chua_generator import (
    ChuaGenerator,
)

# ==========================================================
# PARAMETERS
# ==========================================================

sigma_values = [
    0,
    1e-7,
    1e-6,
    0.1,
    0.2
]

# ==========================================================
# CHUA PARAMETERS
# ==========================================================

alpha = 9.0
beta = 14.285714

m0 = -8/7
m1 = -5/7

# ==========================================================
# EQUILIBRIUM POINTS
# ==========================================================

xe = (m0 - m1) / (1 + m1)

E_plus = np.array([xe, 0, -xe])
E_minus = np.array([-xe, 0, xe])

# ==========================================================
# CREATE CHUA GENERATOR
# ==========================================================

chua = ChuaGenerator(
    alpha=alpha,
    beta=beta,
    m0=m0,
    m1=m1,
    dt=0.01,
    T=1000
)

# ==========================================================
# FIGURE
# ==========================================================

fig = plt.figure(figsize=(16,10))

# ==========================================================
# LOOP
# ==========================================================

for k, sigma in enumerate(sigma_values):

    print(f"\nRunning sigma = {sigma}")

    # -----------------------------------------
    # Generate chaotic signal
    # -----------------------------------------

    x, y, z = chua.simulate(
        sigma=sigma,
        seed=0
    )

    # -----------------------------------------
    # Remove transient
    # -----------------------------------------

    x, y, z, cut = chua.remove_transient(
        x,
        y,
        z,
        transient_time=100     # có thể thay đổi
    )

    print(f"Cut sample = {cut}")

    # -----------------------------------------
    # Plot
    # -----------------------------------------

    ax = fig.add_subplot(
        2,
        3,
        k+1,
        projection="3d"
    )

    ax.plot(
        x,
        y,
        z,
        linewidth=0.4
    )

    ax.scatter(
        E_plus[0],
        E_plus[1],
        E_plus[2],
        color='red',
        s=50,
        label='E+'
    )

    ax.scatter(
        E_minus[0],
        E_minus[1],
        E_minus[2],
        color='blue',
        s=50,
        label='E-'
    )

    ax.set_title(f"$\\sigma$ = {sigma}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

plt.tight_layout()
plt.show()
