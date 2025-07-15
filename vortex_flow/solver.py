import numpy as np

# Das hier kümmert sich um die Logik der zeitlichen Integration

class Solver:
    def __init__(self, grid, nu, dt, problem):
        self.grid = grid
        self.nu = nu
        self.dt = dt
        self.problem = problem
        
        self.omega = None
        self.psi = None
        self.u = None
        self.v = None

    def set_initial_conditions(self, omega0):
        self.omega = omega0.copy()

    def solve_poisson(self):
        # Hier wird dann die Stromfunktion aktualisiert (ψ) 

        # TODO: Laplace-Matrix mit Dirichlet-Bedingungen vorbereiten
        rhs = -self.omega.flatten()
        
        psi_flat = np.zeros_like(rhs)  # TODO: Stattdessen solven
        self.psi = psi_flat.reshape(self.grid.n, self.grid.m)

        # TODO: Randwerte der Stromfunktion setzen
        self._set_psi_boundary()

    def compute_velocity(self):
        # TODO: Geschwindigkeit aus der Stromfunktion berechnen
        self.u = np.zeros_like(self.psi)
        self.v = np.zeros_like(self.psi)

        # TODO: Randwerte der Geschwindigkeit setzen
        self._set_velocity_boundary()

    def time_step(self):
        # Timon: Man nimmt standardmäßig Runge Kutta 
        # man könnte später noch mit anderen Verfahren vergleichen
        
        # TODO: Konvektion und Diffusion der Wirbelstärke

        # TODO: Randwerte der Wirbelstärke setzen
        self._set_vorticity_boundary()

    def _set_psi_boundary(self):
        self.problem.set_psi_boundary(self)

    def _set_velocity_boundary(self):
        self.problem.set_velocity_boundary(self)

    def _set_vorticity_boundary(self):
        self.problem.set_vorticity_boundary(self)
