import numpy as np
import matplotlib.pyplot as plt
from vortex_flow import Grid2D, VortexSolver, LidDrivenCavity #, TaylorGreen, KarmanVortexStreet lol

# man d
# Parameters
NX, NY = 64, 64 # Gr
N_STEPS = 500 # iterationsschritte
OUTPUT_FREQUENCY = 500 # kleiner wenn man Zwischenschritte will

# Initialisation
problem = LidDrivenCavity()
grid = Grid2D(nx=NX, ny=NY, lx=1.0, ly=1.0)
omega0 = problem.initial_conditions(grid)

solver = VortexSolver(grid=grid, nu=0.01, dt=0.001)
solver.set_initial_conditions(omega0)

# Simulation
for step in range(N_STEPS):
    solver.solve_poisson()
    solver.compute_velocity()
    solver.time_step()

    if step % OUTPUT_FREQUENCY == 0:
        plt.clf()
        plt.imshow(solver.omega, origin="lower", cmap="seismic")
        plt.title(f"Step {step}")
        plt.colorbar()
        plt.pause(0.1)

plt.show()
