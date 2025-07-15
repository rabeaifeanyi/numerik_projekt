import numpy as np
from scipy.sparse import diags, identity, kron #sparce alles nehmen, alles was geht haha

# Siehe Thema 3
# -------------------------------------------------------------------------
# - Sparce Scaling mit scipy sparce (viel Einträge mit 0en)
# - Das hier baut erstmal nur das Gitter und Ableitungsmatrizen

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
        
        self.Dx1, self.Dx2 = self._derivative(n, self.dx)
        self.Dy1, self.Dy2 = self._derivative(m, self.dy)
        
        Ix = identity(n)
        Iy = identity(m)

        # An Mittelpunkten Laplace-Operator (also in alle Richtungen)
        # Zweite Ableitung nach x und zweite Ableitung nach y ist wie eine Gesamtkrümmung
        # Laplace-Operator in 2D = Summe der zweiten Ableitungen in x und y = vier Nachbarn minus viermal der Mittelpunkt

        #         (i,j+1)
        #             ↑
        # (i-1,j) ← (i,j) → (i+1,j)
        #             ↓
        #         (i,j-1)

        # Statt das jetzt in 2D Matrizen zu verwandeln, nimmt man direkt einen 1D Vektor, weil man als "Trick" so das Kronecker Produkt verwenden kann
        # Kronecker macht aus 1D 2D, daher wollen wir vorher flach machen
        # Statt das Dx1D =
                    # [ [-1, +1, 0],
                    #   [0, -1, +1],
                    #         ... ]
        # in alle Zeilen zu kopieren, macht das Kronecker Produkt genau das Dx2D = I_y ⊗ Dx1D, Dy2D = Dy1D ⊗ I_x

        # Laplace-Operator
        self.Laplace = kron(Iy, self.Dx2) + kron(self.Dy2, Ix)
    
    def _derivative(self, n, h):
        # ChatGPT meinte man kann die Standardableitungen nehmen, aber ich glaube man muss alpha schon berechnen

        main = np.zeros(n)
        upper = +0.5 * np.ones(n-1)
        lower = -0.5 * np.ones(n-1)
        D1 = diags([lower, main, upper], [-1, 0, 1]) / h

        main2 = -2.0 * np.ones(n)
        upper2 = np.ones(n-1)
        lower2 = np.ones(n-1)
        D2 = diags([lower2, main2, upper2], [-1, 0, 1]) / (h ** 2)

        return D1, D2
    
    
        if alpha1 is None:
            alpha1 = [-0.5, 0.0, 0.5]
        if alpha2 is None:
            alpha2 = [1.0, -2.0, 1.0]

