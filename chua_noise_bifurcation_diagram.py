import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARAMETERS
# ==========================================================
alpha = 9.0
beta  = 14.285714

m0 = -8/7
m1 = -5/7

dt = 0.1
T  = 100
N  = int(T/dt)

# ==========================================================
# SIGMA SWEEP
# ==========================================================
sigma_values = np.linspace(
    0,
    0.2,
    100
)

# ==========================================================
# CHUA NONLINEARITY
# ==========================================================
def h(x):

    return (
        m1*x
        + 0.5*(m0-m1)*(abs(x+1)-abs(x-1))
    )

# ==========================================================
# SIMULATION
# ==========================================================
def simulate_chua(sigma, seed=0):

    np.random.seed(seed)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    x[0] = 0.1
    y[0] = 0
    z[0] = 0

    sqrt_dt = np.sqrt(dt)

    for n in range(N-1):

        dx = alpha*(y[n] - x[n] - h(x[n]))
        dy = x[n] - y[n] + z[n]
        dz = -beta*y[n]

        x[n+1] = x[n] + dx*dt
        y[n+1] = y[n] + dy*dt
        z[n+1] = z[n] + dz*dt + sigma*np.random.randn()*sqrt_dt

    return x

# ==========================================================
# BUILD RESIDENCE TIME DATA
# ==========================================================
SIGMA = []
RTIME = []

for k, sigma in enumerate(sigma_values):

    print(f"{k+1}/{len(sigma_values)}")

    x = simulate_chua(
        sigma=sigma,
        seed=0
    )

    # ------------------------------------------------------
    # REMOVE TRANSIENT
    # ------------------------------------------------------
    cut = int(0.2*N)

    x = x[cut:]

   # ------------------------------------------------------
# RESIDENCE TIMES
# ------------------------------------------------------

current_side = np.sign(x[0])
counter = 0

RT_list = []

for xi in x:

    side = np.sign(xi)

    if side == current_side:

        counter += 1

    else:

        RT = counter * dt

        RT_list.append(RT)

        current_side = side
        counter = 1

# Mean Residence Time
if len(RT_list) > 0:

    mean_RT = np.mean(RT_list)

    SIGMA.append(sigma)
    RTIME.append(mean_RT)

# ==========================================================
# PLOT
# ==========================================================
plt.figure(figsize=(10,6))

plt.plot(
    SIGMA,
    RTIME,
    'o-',
    markersize=3
)

plt.xlabel(r'$\sigma$')
plt.ylabel('Mean Residence Time')

plt.title(
    'Mean Residence Time vs Noise Intensity'
)

plt.grid(True)

plt.tight_layout()
plt.show()