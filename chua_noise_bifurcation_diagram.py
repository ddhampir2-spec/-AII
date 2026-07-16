import numpy as np
import matplotlib.pyplot as plt

from chua_generator import ChuaGenerator

# ==========================================================
# CREATE CHUA GENERATOR
# ==========================================================

chua = ChuaGenerator(
    alpha=9.0,
    beta=14.285714,
    m0=-8/7,
    m1=-5/7,
    dt=0.01,
    T=1000
)

# ==========================================================
# PARAMETERS
# ==========================================================

sigma_values = np.linspace(0, 0.2, 20)

transient_time = 100

SIGMA = []
RTIME = []

# ==========================================================
# SIGMA LOOP
# ==========================================================

for k, sigma in enumerate(sigma_values):

    print(f"{k+1}/{len(sigma_values)}")

    # ------------------------------------------------------
    # Generate chaotic signal
    # ------------------------------------------------------

    x, y, z = chua.simulate(
        sigma=sigma,
        seed=0
    )

    # ------------------------------------------------------
    # Remove transient
    # ------------------------------------------------------

    x, y, z, cut = chua.remove_transient(
        x,
        y,
        z,
        transient_time
    )

    # ------------------------------------------------------
    # Residence Time
    # ------------------------------------------------------

    current_side = np.sign(x[0])
    counter = 0

    RT_list = []

    for xi in x:

        side = np.sign(xi)

        if side == current_side:

            counter += 1

        else:

            RT = counter * chua.dt

            RT_list.append(RT)

            current_side = side
            counter = 1

    # ------------------------------------------------------
    # Mean Residence Time
    # ------------------------------------------------------

    if len(RT_list) > 0:

        mean_RT = np.mean(RT_list)

        SIGMA.append(sigma)
        RTIME.append(mean_RT)

# ==========================================================
# PLOT
# ==========================================================

plt.figure(figsize=(9,6))

plt.plot(
    SIGMA,
    RTIME,
    'o-',
    markersize=4
)

plt.xlabel("Noise Intensity σ")

plt.ylabel("Mean Residence Time")

plt.title("Residence Time vs Noise Intensity")

plt.grid(True)

plt.tight_layout()

plt.show()
