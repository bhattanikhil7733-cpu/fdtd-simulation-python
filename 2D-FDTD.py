import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1. Simulation Parameters
# -------------------------------------------------

nx, ny = 500, 500
dx = dy = 1.0

c0 = 299_792_458.0
eps0 = 8.854e-12
mu0 = 4 * np.pi * 1e-7

S = 0.5  # Must be <= 1/sqrt(2)
dt = S * dx / (c0 * np.sqrt(2))

nt = 800

# -------------------------------------------------
# 2. Material Definition (Free Space)
# -------------------------------------------------

eps = eps0 * np.ones((nx, ny))

# -------------------------------------------------
# 3. Field Initialization
# -------------------------------------------------

Ez = np.zeros((nx, ny))
Hx = np.zeros((nx, ny - 1))
Hy = np.zeros((nx - 1, ny))

ce = dt / eps
ch = dt / mu0

# -------------------------------------------------
# 4. Visualization Setup
# -------------------------------------------------

plt.ion()
fig, ax = plt.subplots(figsize=(7, 7))

im = ax.imshow(
    Ez.T,
    cmap="turbo",
    origin="lower",
    extent=[0, nx, 0, ny],
)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("Electric Field $E_z$")

ax.set_title("2D TMz FDTD - PEC Cavity Reflection")
ax.set_xlabel("x (grid cells)")
ax.set_ylabel("y (grid cells)")
ax.set_aspect("equal")
ax.set_facecolor("black")

plt.tight_layout()

# -------------------------------------------------
# 5. Time Stepping Loop
# -------------------------------------------------

t0, sigma = 60, 20
use_soft_source = True

# Source at exact center
sx = nx // 2
sy = ny // 2

for n in range(nt):

    # --- Update Magnetic Fields ---
    Hx -= (ch / dy) * (Ez[:, 1:] - Ez[:, :-1])
    Hy += (ch / dx) * (Ez[1:, :] - Ez[:-1, :])

    # --- Update Electric Field ---
    curl = (
        (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) / dx
        -
        (Hx[1:-1, 1:] - Hx[1:-1, :-1]) / dy
    )

    Ez[1:-1, 1:-1] += ce[1:-1, 1:-1] * curl

    # --- PEC Boundary Conditions (Perfect Reflection) ---
    Ez[0, :] = 0
    Ez[-1, :] = 0
    Ez[:, 0] = 0
    Ez[:, -1] = 0

    # --- Gaussian Source ---
    pulse = np.exp(-0.5 * ((n - t0) / sigma) ** 2)
    pulse = np.sin(2 * np.pi * 0.02 * n)
    Ez[sx, sy] += pulse
    if use_soft_source:
        Ez[sx, sy] += pulse
    else:
        Ez[sx, sy] = pulse

    # --- Plot ---
    if n % 5 == 0:
        im.set_array(Ez.T)
        max_val = np.max(np.abs(Ez))
        im.set_clim(-max_val, max_val)
        plt.draw()
        plt.pause(0.001)

plt.ioff()
plt.show()