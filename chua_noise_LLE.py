import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# PARAMETERS
# ==========================================================
alpha = 9.0
beta  = 14.285714

m0 = -8/7
m1 = -5/7

dt = 0.001

sigma_range = np.arange(
    0.0,
    0.2,
    0.002
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
# STATE EQUATIONS
# ==========================================================
def f(X):

    x, y, z = X

    dx = alpha * (y - x - h(x))
    dy = x - y + z
    dz = -beta * y

    return np.array([dx, dy, dz])

# ==========================================================
# RK4 STEP
# ==========================================================
def rk4_step(X, noise):

    k1 = f(X)

    k2 = f(
        X + 0.5*dt*k1
    )

    k3 = f(
        X + 0.5*dt*k2
    )

    k4 = f(
        X + dt*k3
    )

    Xnew = (
        X
        + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)
    )

    # Noise only acts on z
    Xnew[2] += noise

    return Xnew

# ==========================================================
# BENETTIN LLE
# ==========================================================
def compute_LLE(sigma):

    T = 100
    N = int(T/dt)

    transient = int(0.2*N)

    d0 = 1e-8

    X1 = np.array([
        0.1,
        0.0,
        0.0
    ])

    X2 = X1 + np.array([
        d0,
        0,
        0
    ])

    S = 0.0
    count = 0

    sqrt_dt = np.sqrt(dt)

    for n in range(N):

        # ==================================================
        # SAME NOISE REALIZATION
        # ==================================================
        noise = sigma*np.random.randn()*sqrt_dt

        # ==================================================
        # RK4
        # ==================================================
        X1 = rk4_step(X1, noise)
        X2 = rk4_step(X2, noise)

        # ==================================================
        # AFTER TRANSIENT
        # ==================================================
        if n > transient:
            d = np.linalg.norm(
                X2 - X1
            )

            if d > 0:

                S += np.log(d/d0)

                count += 1

                direction = (
                    (X2 - X1)/d
                )   

                X2 = (
                    X1
                    + d0*direction
                )

    return S/(count*dt)

# ==========================================================
# COMPUTE LLE
# ==========================================================
LLE = []

for sigma in sigma_range:

    print(
        f"sigma = {sigma:.3f}"
    )

    lle = compute_LLE(sigma)

    LLE.append(lle)

# ==========================================================
# PLOT
# ==========================================================
plt.figure(figsize=(8,6))

plt.plot(
    sigma_range,
    LLE,
    'o-',
    linewidth=2
)

plt.axhline(
    0,
    linestyle='--'
)

plt.xlabel("Noise intensity σ")

plt.ylabel(
    "Largest Lyapunov Exponent"
)

plt.title(
    "Benettin LLE vs Noise Intensity"
)

plt.grid(True)

plt.tight_layout()

plt.show()