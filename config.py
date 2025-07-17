import numpy as np

Nx, Ny = 30, 30
Lx, Ly = 1.0, 1.0
Re = 400
U_tang = 1.0
U_noSlip = 0.0
CFL = 0.3
steps = 10000
save_interval = 100

problem = "lid_standard"

omega_init = 1e-2 * (np.random.rand(Ny, Nx) - 0.5)
psi_init = np.zeros((Ny, Nx))
u_init = np.zeros((Ny, Nx))
v_init = np.zeros((Ny, Nx))

folder = f"plots/plots_CFL{CFL:.2f}_Re{Re}".replace(".", "p")
