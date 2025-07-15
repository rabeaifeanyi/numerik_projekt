import numpy as np
# Jedes Problem hat unterschiedliche Startwerte und Bedingungen
# Das hier umfasst also nur alles worin sich die Probleme unterscheiden, also erstmal nur die Rand- und Anfangsbedingungen

class ProblemSetup:
    def __init__(self):
        pass

    def initial_conditions(self, grid):
        X, Y = np.meshgrid(grid.x, grid.y, indexing="ij")
        omega0 = np.zeros_like(X)
        return omega0

    def set_psi_boundary(self, solver):
        pass

    def set_velocity_boundary(self, solver):
        pass

    def set_vorticity_boundary(self, solver):
        pass

class LidDrivenCavity(ProblemSetup):
    def __init__(self):
        super().__init__()

    # initial conditions brauchen wir nicht neu setzten, das setzt man null am Anfang glaub ich wie in dem Basis ProblemSolver

    def set_psi_boundary(self, solver):
        solver.psi[0,:] = 0
        solver.psi[-1,:] = 0
        solver.psi[:,0] = 0
        solver.psi[:,-1] = 0

    def set_velocity_boundary(self, solver):
        solver.u[0,:] = 0
        solver.u[-1,:] = 1 # Deckel
        solver.u[:,0] = 0
        solver.u[:,-1] = 0
        solver.v[:,:] = 0

    def set_vorticity_boundary(self, solver):
        # TODO: Wirbelstärke am Rand aus psi bestimmen
        # Dummy:
        solver.omega[-1,:] = 0
        solver.omega[0,:] = 0
        solver.omega[:,0] = 0
        solver.omega[:,-1] = 0



# class TaylorGreen(ProblemSetup):
#     def __init__(self):
#         super().__init__()

#     def initial_conditions(self, grid):
#         X, Y = np.meshgrid(grid.x, grid.y)
#         omega0 = -2 * np.sin(X) * np.sin(Y)
#         return omega0

#     def boundary_conditions(self, solver):
#         # Timon:
#         # "periodische Randbedingungen sind gut, aber man braucht nicht unbedingt welche. 
#         # hilft bei Stabilität
#         # mit periodischen Randbedingungen hätte man das "Glückslos gezogen" 
#         pass
