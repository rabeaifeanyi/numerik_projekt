import matplotlib.pyplot as plt
from vortex_flow import Grid2D, Solver, LidDrivenCavity #, TaylorGreen, KarmanVortexStreet lol

if __name__ == "__main__":
    # Parameter
    NX, NY = 64, 64
    N_STEPS = 100  # Kleiner Wert für schnellen Test
    OUTPUT_FREQUENCY = 10  


    # Initialisation
    problem = LidDrivenCavity()
    grid = Grid2D(n=NX, m=NY, lx=1.0, ly=1.0)
    omega0 = problem.initial_conditions(grid)

    solver = Solver(grid=grid, nu=0.01, dt=0.001, problem=problem)
    solver.set_initial_conditions(omega0)

    # Simulation
    for step in range(1, N_STEPS+1):
        solver.solve_poisson()
        solver.compute_velocity()
        solver.time_step()

        if step % OUTPUT_FREQUENCY == 0:
            plt.clf()
            plt.imshow(solver.omega, origin="lower", cmap="seismic")
            plt.title(f"Step {step}")
            plt.colorbar()
            plt.pause(0.2)

    plt.show()