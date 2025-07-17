import numpy as np
from .fdm import derivative_matrix_2d

class Grid:
    def __init__(self, Nx, Ny, Lx, Ly):
        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.x = np.linspace(0, Lx, Nx)
        self.y = np.linspace(0, Ly, Ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.Dx = derivative_matrix_2d(Nx, Ny, self.dx, self.dy, 'x', order=1)
        self.Dy = derivative_matrix_2d(Nx, Ny, self.dx, self.dy, 'y', order=1)
        self.Dxx = derivative_matrix_2d(Nx, Ny, self.dx, self.dy, 'x', order=2)
        self.Dyy = derivative_matrix_2d(Nx, Ny, self.dx, self.dy, 'y', order=2)
