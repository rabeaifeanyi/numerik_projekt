import numpy as np
# Jedes Problem hat Startwerte, und unterschiedliche Bedingungen

class ProblemSetup:
    def __init__(self):
        pass

    def initial_conditions(self, grid):
        X, Y = np.meshgrid(grid.x, grid.y)
        omega0 = np.zeros_like(X)
        return omega0

    def boundary_conditions(self, solver):
        pass

class LidDrivenCavity(ProblemSetup):
    def __init__(self):
        super().__init__()

    # def initial_conditions(self, grid):
        # brauchen wir nicht, das setzt man null am Anfang glaub ich wie in dem Basis ProblemSolver
    
    def boundary_conditions(self, solver):
        pass
        # man kann mit null ohne Wirbel starten oder verwirbelt starten. ABer die Wände ändern von selbst die Wirbelstärke zwischendurch
        # 1. Stromfunktion Psi am Rand (Dirichlet-Bedingungen)
        # 2. Alle Wände Psi = 0 (kein Volumenstrom)
        # 3. Setze Geschwindigkeitsrandbedingungen für den bewegten Deckel
        # 4. Setze die Wirbelstärke am Rand
        # 5. Dirichlet-Randbedingungen an Psi ist die Wirbelstärke auf dem Rand:
        # omega = - (d^2 Psi / dn^2) ?


# class TaylorGreen(ProblemSetup):
#     def __init__(self):
#         super().__init__()

#     def initial_conditions(self, grid):
#         X, Y = np.meshgrid(grid.x, grid.y)
#         omega0 = -2 * np.sin(X) * np.sin(Y)
#         return omega0

#     def boundary_conditions(self, solver):
#         # periodische Randbedingungen, aber man braucht nicht unbedingt welche. Zusammenhang zwischen Stabilität und Randbedingungen
#         # mit periodischen Randbedingungen "Glückslos gezogen lol" 
#         pass
