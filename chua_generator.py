# ==========================================================
# chua_generator.py
# Chua Chaotic Signal Generator
# ==========================================================

import numpy as np

# ==========================================================
# Chua Generator
# ==========================================================

class ChuaGenerator:

    def __init__(self,
                 alpha=9.0,
                 beta=14.285714,
                 m0=-8/7,
                 m1=-5/7,
                 dt=0.01,
                 T=1000):

        self.alpha = alpha
        self.beta = beta

        self.m0 = m0
        self.m1 = m1

        self.dt = dt
        self.T = T

        self.N = int(T/dt)

    # ======================================================
    # Chua Nonlinearity
    # ======================================================

    def h(self, x):

        return (
            self.m1*x
            +0.5*(self.m0-self.m1)
            *(np.abs(x+1)-np.abs(x-1))
        )

    # ======================================================
    # State Equations
    # ======================================================

    def f(self, x, y, z):

        dx = self.alpha*(y-x-self.h(x))
        dy = x-y+z
        dz = -self.beta*y

        return dx, dy, dz

    # ======================================================
    # RK4 Simulation
    # ======================================================

    def simulate(self,
                 sigma,
                 x0=0.1,
                 y0=0,
                 z0=0,
                 seed=None):

        if seed is not None:
            np.random.seed(seed)

        x = np.zeros(self.N)
        y = np.zeros(self.N)
        z = np.zeros(self.N)

        x[0] = x0
        y[0] = y0
        z[0] = z0

        sqrt_dt = np.sqrt(self.dt)

        for n in range(self.N-1):

            k1x,k1y,k1z = self.f(
                x[n],
                y[n],
                z[n]
            )

            k2x,k2y,k2z = self.f(
                x[n]+0.5*self.dt*k1x,
                y[n]+0.5*self.dt*k1y,
                z[n]+0.5*self.dt*k1z
            )

            k3x,k3y,k3z = self.f(
                x[n]+0.5*self.dt*k2x,
                y[n]+0.5*self.dt*k2y,
                z[n]+0.5*self.dt*k2z
            )

            k4x,k4y,k4z = self.f(
                x[n]+self.dt*k3x,
                y[n]+self.dt*k3y,
                z[n]+self.dt*k3z
            )

            noise = sigma*np.random.randn()*sqrt_dt

            x[n+1] = (
                x[n]
                + self.dt/6
                * (k1x+2*k2x+2*k3x+k4x)
            )

            y[n+1] = (
                y[n]
                + self.dt/6
                * (k1y+2*k2y+2*k3y+k4y)
            )

            z[n+1] = (
                z[n]
                + self.dt/6
                * (k1z+2*k2z+2*k3z+k4z)
                + noise
            )

        return x, y, z
    
    # ======================================================
    # One RK4 Step
    # ======================================================

    def rk4_step(self, X, noise=0):

        x, y, z = X

        k1 = np.array(self.f(x, y, z))

        k2 = np.array(
            self.f(
                *(X + 0.5*self.dt*k1)
            )
        )

        k3 = np.array(
            self.f(
                *(X + 0.5*self.dt*k2)
            )
        )

        k4 = np.array(
            self.f(
                *(X + self.dt*k3)
            )
        )

        Xnew = (
            X
            + self.dt/6
            * (k1 + 2*k2 + 2*k3 + k4)
        )

        Xnew[2] += noise

        return Xnew

    # ======================================================
    # Normalize
    # ======================================================

    @staticmethod
    def normalize(signal):

        return signal / np.max(np.abs(signal))

    # ======================================================
    # Remove Transient by Fixed Time
    # ======================================================

    def remove_transient(self,
                         x,
                         y,
                         z,
                         transient_time):

        cut = int(transient_time / self.dt)

        return (
            x[cut:],
            y[cut:],
            z[cut:],
            cut
        )
