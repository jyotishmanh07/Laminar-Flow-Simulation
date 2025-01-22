#!/usr/bin/env python
# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Constants
N_POINTS_X = 81
N_POINTS_Y = 41
DOMAIN_LENGTH = 2.0  # Length of the channel
DOMAIN_HEIGHT = 1.0  # Height of the channel
N_ITERATIONS = 500
TIME_STEP_LENGTH = 0.001
KINEMATIC_VISCOSITY = 0.001
DENSITY = 1.0
INLET_VELOCITY = 1.0

N_PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5

def main():
    dx = DOMAIN_LENGTH / (N_POINTS_X - 1)
    dy = DOMAIN_HEIGHT / (N_POINTS_Y - 1)
    x = np.linspace(0.0, DOMAIN_LENGTH, N_POINTS_X)
    y = np.linspace(0.0, DOMAIN_HEIGHT, N_POINTS_Y)
    X, Y = np.meshgrid(x, y)

    # Initial conditions
    u = np.zeros_like(X)
    v = np.zeros_like(X)
    p = np.zeros_like(X)

    def central_difference_x(f):
        diff = np.zeros_like(f)
        diff[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
        return diff

    def central_difference_y(f):
        diff = np.zeros_like(f)
        diff[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)
        return diff

    def laplace(f):
        diff = np.zeros_like(f)
        diff[1:-1, 1:-1] = (
            (f[1:-1, 2:] + f[1:-1, :-2] - 2 * f[1:-1, 1:-1]) / dx**2 +
            (f[2:, 1:-1] + f[:-2, 1:-1] - 2 * f[1:-1, 1:-1]) / dy**2
        )
        return diff

    max_dt = 0.5 * min(dx, dy)**2 / KINEMATIC_VISCOSITY
    if TIME_STEP_LENGTH > STABILITY_SAFETY_FACTOR * max_dt:
        raise RuntimeError("Stability is not guaranteed")

    for _ in tqdm(range(N_ITERATIONS)):
        # Tentative velocity
        lap_u = laplace(u)
        lap_v = laplace(v)
        d_u_dx = central_difference_x(u)
        d_u_dy = central_difference_y(u)
        d_v_dx = central_difference_x(v)
        d_v_dy = central_difference_y(v)

        u_tent = u + TIME_STEP_LENGTH * (
            -u * d_u_dx - v * d_u_dy + KINEMATIC_VISCOSITY * lap_u
        )
        v_tent = v + TIME_STEP_LENGTH * (
            -u * d_v_dx - v * d_v_dy + KINEMATIC_VISCOSITY * lap_v
        )

        # Apply velocity boundary conditions
        u_tent[:, 0] = INLET_VELOCITY  # Inlet
        u_tent[:, -1] = u_tent[:, -2]  # Outlet (Neumann BC)
        u_tent[0, :] = 0.0  # Bottom wall
        u_tent[-1, :] = 0.0  # Top wall
        v_tent[:, 0] = 0.0
        v_tent[:, -1] = 0.0
        v_tent[0, :] = 0.0
        v_tent[-1, :] = 0.0

        # Pressure Poisson equation
        rhs = DENSITY / TIME_STEP_LENGTH * (
            central_difference_x(u_tent) + central_difference_y(v_tent)
        )

        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_new = np.copy(p)
            p_new[1:-1, 1:-1] = 0.25 * (
                p[1:-1, :-2] + p[1:-1, 2:] +
                p[:-2, 1:-1] + p[2:, 1:-1] -
                (dx * dy)**2 * rhs[1:-1, 1:-1]
            )
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p = p_new

        # Correct velocities
        u = u_tent - TIME_STEP_LENGTH / DENSITY * central_difference_x(p)
        v = v_tent - TIME_STEP_LENGTH / DENSITY * central_difference_y(p)

        # Reapply boundary conditions
        u[:, 0] = INLET_VELOCITY
        u[:, -1] = u[:, -2]
        u[0, :] = 0.0
        u[-1, :] = 0.0
        v[:, 0] = 0.0
        v[:, -1] = 0.0
        v[0, :] = 0.0
        v[-1, :] = 0.0

    # Visualization
    plt.style.use("dark_background")
    plt.figure(figsize=(10, 5))
    plt.contourf(X, Y, p, cmap="coolwarm", levels=50)
    plt.colorbar(label="Pressure")
    plt.streamplot(X, Y, u, v, color="white", linewidth=0.7, density=2)
    plt.title("Channel Flow Simulation")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, DOMAIN_LENGTH)
    plt.ylim(0, DOMAIN_HEIGHT)
    plt.show()

if __name__ == "__main__":
    main()

    



# In[ ]:





# In[ ]:




