import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import eig
from mpl_toolkits.mplot3d import Axes3D

# ==========================================================
# PARAMETERS
# ==========================================================
alpha = 9.0
beta  = 14.285714

m0 = -8/7
m1 = -5/7

dt = 0.01
T  = 1000
N  = int(T/dt)

# ==========================================================
# CUSTOM SIGMA VALUES
# ==========================================================
sigma_values = [
    0,
    1e-7,
    1e-6,
    0.1,
    0.2
]

# ==========================================================
# EQUILIBRIUM POINTS
# ==========================================================
xe = (m0 - m1)/(1 + m1)

E_plus  = np.array([ xe, 0, -xe ])
E_minus = np.array([-xe, 0,  xe ])

print("\n========================================")
print("EQUILIBRIUM POINTS")
print("========================================")

print("E_plus  =", E_plus)
print("E_minus =", E_minus)

# ==========================================================
# JACOBIAN MATRIX
# ==========================================================
J = np.array([

    [-alpha*(1+m1),  alpha, 0],
    [1,               -1,   1],
    [0,              -beta, 0]

])

print("\n========================================")
print("JACOBIAN MATRIX")
print("========================================")

print(J)

# ==========================================================
# EIGENVALUES / EIGENVECTORS
# ==========================================================
eigvals, eigvecs = eig(J)

print("\n========================================")
print("EIGENVALUES / EIGENVECTORS")
print("========================================")

for i in range(len(eigvals)):

    lam = eigvals[i]

    vec = eigvecs[:,i]

    print(f"\nEigenvalue {i+1}")
    print("----------------------------------------")

    print("lambda =")
    print(lam)

    print("\nEigenvector =")
    print(vec)

    # ------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------
    if np.real(lam) > 0:

        print("\nType : UNSTABLE")

    elif np.real(lam) < 0:

        print("\nType : STABLE")

    else:

        print("\nType : CENTER")



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
def f(x, y, z):

    dx = alpha * (y - x - h(x))
    dy = x - y + z
    dz = -beta * y

    return dx, dy, dz

# ==========================================================
# RK4 SIMULATION WITH GAUSSIAN NOISE
# ==========================================================
def simulate_chua(sigma, seed=0):

    np.random.seed(seed)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    # Initial condition
    x[0] = 0.1
    y[0] = 0.0
    z[0] = 0.0

    sqrt_dt = np.sqrt(dt)

    for n in range(N-1):

        # -----------------------------
        # RK4 - k1
        # -----------------------------
        k1x, k1y, k1z = f(
            x[n],
            y[n],
            z[n]
        )

        # -----------------------------
        # RK4 - k2
        # -----------------------------
        k2x, k2y, k2z = f(
            x[n] + 0.5*dt*k1x,
            y[n] + 0.5*dt*k1y,
            z[n] + 0.5*dt*k1z
        )

        # -----------------------------
        # RK4 - k3
        # -----------------------------
        k3x, k3y, k3z = f(
            x[n] + 0.5*dt*k2x,
            y[n] + 0.5*dt*k2y,
            z[n] + 0.5*dt*k2z
        )

        # -----------------------------
        # RK4 - k4
        # -----------------------------
        k4x, k4y, k4z = f(
            x[n] + dt*k3x,
            y[n] + dt*k3y,
            z[n] + dt*k3z
        )

        # -----------------------------
        # Gaussian noise
        # -----------------------------
        noise = sigma * np.random.randn() * sqrt_dt

        # -----------------------------
        # RK4 Update
        # -----------------------------
        x[n+1] = (
            x[n]
            + dt/6.0 * (
                k1x
                + 2*k2x
                + 2*k3x
                + k4x
            )
        )

        y[n+1] = (
            y[n]
            + dt/6.0 * (
                k1y
                + 2*k2y
                + 2*k3y
                + k4y
            )
        )

        z[n+1] = (
            z[n]
            + dt/6.0 * (
                k1z
                + 2*k2z
                + 2*k3z
                + k4z
            )
            + noise
        )

    return x, y, z

# ==========================================================
# FIGURE
# ==========================================================
fig = plt.figure(figsize=(16,10))

# ==========================================================
# LOOP OVER SIGMA
# ==========================================================
for k, sigma in enumerate(sigma_values):

    print(f"\nRunning sigma = {sigma}")

    x, y, z = simulate_chua(sigma)

    # ------------------------------------------------------
    # REMOVE TRANSIENT
    # ------------------------------------------------------
    cut = int(0.2*N)

    x = x[cut:]
    y = y[cut:]
    z = z[cut:]

    # ======================================================
    # 3D AXIS
    # ======================================================
    ax = fig.add_subplot(
        2,
        3,
        k+1,
        projection='3d'
    )

    # ======================================================
    # 3D PHASE PORTRAIT
    # ======================================================
    ax.plot(
        x,
        y,
        z,
        linewidth=0.4
    )

    # ======================================================
    # EQUILIBRIUM POINTS
    # ======================================================
    ax.scatter(
        E_plus[0],
        E_plus[1],
        E_plus[2],
        s=50
    )

    ax.scatter(
        E_minus[0],
        E_minus[1],
        E_minus[2],
        s=50
    )

    # ======================================================
    # LABELS
    # ======================================================
    ax.set_title(
        f"$\\sigma$ = {sigma}"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

# ==========================================================
# FINAL LAYOUT
# ==========================================================
plt.tight_layout()

plt.show()