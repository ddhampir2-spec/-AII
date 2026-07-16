import numpy as np
import matplotlib.pyplot as plt

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

sigma = 0.1
num_runs = 5

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
# EQUILIBRIUM
# ==========================================================
xe = (m0 - m1)/(1 + m1)

E_plus  = np.array([ xe, 0, -xe ])
E_minus = np.array([-xe, 0,  xe ])

# ==========================================================
# PLANES
# ==========================================================

# ----------------------------------------------------------
# SECTION 1
#
# x - z = 0
#
# transverse transport section
# ----------------------------------------------------------
normal_cross = np.array([1.0, 0.0, -1.0])
normal_cross /= np.linalg.norm(normal_cross)

# ----------------------------------------------------------
# SECTION 2
#
# x + z = 0
#
# longitudinal section
# ----------------------------------------------------------
normal_long = np.array([1.0, 0.0, 1.0])
normal_long /= np.linalg.norm(normal_long)

# ----------------------------------------------------------
# SECTION 3
#
# through E_plus
#
# x - z = 2xe
# ----------------------------------------------------------
normal_eq = np.array([1.0, 0.0, -1.0])
normal_eq /= np.linalg.norm(normal_eq)

# ==========================================================
# STORAGE
# ==========================================================
cross_p1 = []
cross_p2 = []

long_p1 = []
long_p2 = []

eqp_p1 = []
eqp_p2 = []

eqm_p1 = []
eqm_p2 = []

# ==========================================================
# SIMULATION (RK4 + Gaussian Noise)
# ==========================================================
def simulate_chua(seed):

    np.random.seed(seed)

    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)

    x[0] = 0.1
    y[0] = 0.0
    z[0] = 0.0

    sqrt_dt = np.sqrt(dt)

    for n in range(N-1):

        # ----------------------------
        # RK4 Stage 1
        # ----------------------------
        k1x, k1y, k1z = f(
            x[n],
            y[n],
            z[n]
        )

        # ----------------------------
        # RK4 Stage 2
        # ----------------------------
        k2x, k2y, k2z = f(
            x[n] + 0.5*dt*k1x,
            y[n] + 0.5*dt*k1y,
            z[n] + 0.5*dt*k1z
        )

        # ----------------------------
        # RK4 Stage 3
        # ----------------------------
        k3x, k3y, k3z = f(
            x[n] + 0.5*dt*k2x,
            y[n] + 0.5*dt*k2y,
            z[n] + 0.5*dt*k2z
        )

        # ----------------------------
        # RK4 Stage 4
        # ----------------------------
        k4x, k4y, k4z = f(
            x[n] + dt*k3x,
            y[n] + dt*k3y,
            z[n] + dt*k3z
        )

        # ----------------------------
        # Gaussian noise
        # ----------------------------
        noise = sigma * np.random.randn() * sqrt_dt

        # ----------------------------
        # RK4 Update
        # ----------------------------
        x[n+1] = (
            x[n]
            + dt/6.0 * (
                k1x +
                2*k2x +
                2*k3x +
                k4x
            )
        )

        y[n+1] = (
            y[n]
            + dt/6.0 * (
                k1y +
                2*k2y +
                2*k3y +
                k4y
            )
        )

        z[n+1] = (
            z[n]
            + dt/6.0 * (
                k1z +
                2*k2z +
                2*k3z +
                k4z
            )
            + noise
        )

    return x, y, z

# ==========================================================
# MULTIPLE RUNS
# ==========================================================
for run in range(num_runs):

    print(f"Run {run+1}/{num_runs}")

    x, y, z = simulate_chua(seed=run)

    # ------------------------------------------------------
    # REMOVE TRANSIENT
    # ------------------------------------------------------
    cut = int(0.2*N)

    x = x[cut:]
    y = y[cut:]
    z = z[cut:]

    # ======================================================
    # LOOP
    # ======================================================
    for i in range(len(x)-1):

        X1 = np.array([x[i],   y[i],   z[i]])
        X2 = np.array([x[i+1], y[i+1], z[i+1]])

        # ==================================================
        # SECTION 1
        #
        # x-z = 0
        # ==================================================
        s1 = np.dot(normal_cross, X1)
        s2 = np.dot(normal_cross, X2)

        if s1*s2 < 0:

            theta = abs(s1)/(abs(s1)+abs(s2))

            Xc = X1 + theta*(X2-X1)

            cross_p1.append(Xc[1])
            cross_p2.append(Xc[0] + Xc[2])

        # ==================================================
        # SECTION 2
        #
        # x+z = 0
        # ==================================================
        s1 = np.dot(normal_long, X1)
        s2 = np.dot(normal_long, X2)

        if s1*s2 < 0:

            theta = abs(s1)/(abs(s1)+abs(s2))

            Xc = X1 + theta*(X2-X1)

            long_p1.append(Xc[0] - Xc[2])
            long_p2.append(Xc[1])

        # ==================================================
        # SECTION 3
        #
        # through E_plus
        #
        # x-z = 2xe
        # ==================================================
        s1 = np.dot(normal_eq, X1 - E_plus)
        s2 = np.dot(normal_eq, X2 - E_plus)

        d1 = np.linalg.norm(X1 - E_plus)
        d2 = np.linalg.norm(X2 - E_plus)

        R = 8.0

        if (s1 < 0) and (s2 > 0):

            if (d1 < R) and (d2 < R):

                theta = abs(s1)/(abs(s1)+abs(s2))

                Xc = X1 + theta*(X2-X1)

                dX = Xc - E_plus

                e1 = np.array([1.0,0.0,1.0])
                e1 = e1 / np.linalg.norm(e1)

                e2 = np.array([0.0,1.0,0.0])

                p1 = np.dot(dX, e1)
                p2 = np.dot(dX, e2)

                eqp_p1.append(p1)
                eqp_p2.append(p2)

        # ==================================================
        # SECTION 4
        #
        # through E_minus
        #
        # x-z = -2xe
        # ==================================================
        s1 = np.dot(normal_eq, X1 - E_minus)
        s2 = np.dot(normal_eq, X2 - E_minus)

        d1 = np.linalg.norm(X1 - E_minus)
        d2 = np.linalg.norm(X2 - E_minus)

        if (s1 < 0) and (s2 > 0):

            if (d1 < R) and (d2 < R):

                theta = abs(s1)/(abs(s1)+abs(s2))

                Xc = X1 + theta*(X2-X1)

                dX = Xc - E_minus

                e1 = np.array([1.0,0.0,1.0])
                e1 = e1 / np.linalg.norm(e1)

                e2 = np.array([0.0,1.0,0.0])

                p1 = np.dot(dX, e1)
                p2 = np.dot(dX, e2)

                eqm_p1.append(p1)
                eqm_p2.append(p2)


Nc_cross = len(cross_p1)
Nc_long  = len(long_p1)
Nc_eqp   = len(eqp_p1)
Nc_eqm   = len(eqm_p1)

# ==========================================================
# PLOT
# ==========================================================
fig, axes = plt.subplots(2, 2, figsize=(14,12))

# ==========================================================
# SECTION 1
# ==========================================================
axes[0,0].scatter(
    cross_p1,
    cross_p2,
    s=2,
    alpha=0.35
)

axes[0,0].set_title(
    "Section : x-z = 0"
)

axes[0,0].set_xlabel("y")
axes[0,0].set_ylabel("x+z")

axes[0,0].grid(True)

# ==========================================================
# SECTION 2
# ==========================================================
axes[0,1].scatter(
    long_p1,
    long_p2,
    s=2,
    alpha=0.35
)

axes[0,1].set_title(
    "Section : x+z = 0"
)

axes[0,1].set_xlabel("x-z")
axes[0,1].set_ylabel("y")

axes[0,1].grid(True)

# ==========================================================
# SECTION 3
# ==========================================================
axes[1,0].scatter(
    eqp_p1,
    eqp_p2,
    s=2,
    alpha=0.35
)

axes[1,0].set_title(
    "Section Through E_plus"
)

axes[1,0].set_xlabel("(x+z) local")
axes[1,0].set_ylabel("y")

axes[1,0].grid(True)

# ==========================================================
# SECTION 4
# ==========================================================
axes[1,1].scatter(
    eqm_p1,
    eqm_p2,
    s=2,
    alpha=0.35
)

axes[1,1].set_title(
    "Section Through E_minus"
)

axes[1,1].set_xlabel("(x+z) local")
axes[1,1].set_ylabel("y")

axes[1,1].grid(True)

# ==========================================================
# PRINT QUALITY REPORT
# ==========================================================
print("\n==============================================")
print("POINCARE MAP QUALITY REPORT")
print("==============================================")

print(f"T          = {T}")
print(f"dt         = {dt}")
print(f"N          = {N}")
print(f"sigma      = {sigma}")
print(f"num_runs   = {num_runs}")
print(f"R          = {R}")

print("\n----------------------------------------------")
print("NUMBER OF CROSSINGS")
print("----------------------------------------------")

print(f"Nc_cross   = {Nc_cross}")
print(f"Nc_long    = {Nc_long}")
print(f"Nc_Eplus   = {Nc_eqp}")
print(f"Nc_Eminus  = {Nc_eqm}")
plt.tight_layout()
plt.show()


