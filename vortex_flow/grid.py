import numpy as np
from scipy.sparse import diags, identity, kron #sparce alles nehmen, alles was geht haha

# Siehe Thema 3
# -------------------------------------------------------------------------
# - Sparce Scaling mit scipy sparce (viel Einträge mit 0en)

class Grid2D:
    def __init__(self, n, m, lx, ly):
        self.n = n
        self.m = m
        self.lx = lx
        self.ly = ly
        self.dx = lx / (n - 1)
        self.dy = ly / (m - 1)
        
        self.x = np.linspace(0, lx, n)
        self.y = np.linspace(0, ly, m)
        
        self.Dx1, self.Dx2 = self._1d_derivative(n, self.dx)
        self.Dy1, self.Dy2 = self._1d_derivative(m, self.dy)
        
        Ix = identity(n)
        Iy = identity(m)
        self.Laplace = kron(Iy, self.Dx2) + kron(self.Dy2, Ix)
        
    def _1d_derivative(self, n, h):
        e = np.ones(n)
        Dx1 = diags([-0.5 * e, 0 * e, 0.5 * e], [-1, 0, 1]) / h # Das sind die Finite Differenzen Matrizen (laut ChatGPT!!!!!!!!!)
        Dx2 = diags([e, -2 * e, e], [-1, 0, 1]) / (h ** 2)
        return Dx1, Dx2
