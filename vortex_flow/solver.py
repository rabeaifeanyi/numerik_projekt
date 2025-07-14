class VortexSolver:
    def __init__(self, grid, nu, dt):
        self.grid = grid
        self.nu = nu
        self.dt = dt
        
        self.omega = None
        self.psi = None
        self.u = None
        self.v = None

    def set_initial_conditions(self, omega0):
        self.omega = omega0.copy()

# Timon meinte sowas wie
# 1. über poisson strom
# 2. darüber geschwindigkeiten
# 3. wirbeltransport
# Irgendwo noch die Dirichlet zwischen hauen? Dachte nur am Anfang, aber scheinbar nicht
# die schritte hier implementieren, die werden dann in jeder Iteration einmal ausgeführt

    def solve_poisson(self):
        pass

    def compute_velocity(self):
        pass

    def time_step(self):
        # Standard Runge Kutta 
        # man könnte vergleichen mit einer anderen Methode
        pass

    def dirichlet(self):
        pass



#Später braucht man das irgendwo:
    def laplace(self): # "im Inneren??"
        pass
    # Summe aller 2. partiellen Ableitungen im Raum

