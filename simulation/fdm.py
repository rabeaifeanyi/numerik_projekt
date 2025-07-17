import numpy as np
import scipy.sparse as sp

def derivative_matrix_1d(N, h, order=1, scheme='central'):
    """
    Generates a sparse 1D finite difference matrix for the derivative of given order.

    Parameters
    ----------
    N : int
        Number of grid points in corresponding direction
    h : float
        Grid spacing
    order : int
        Derivative order
    scheme : str
        'central', 'forward' or 'backward'

    Returns
    -------
    D : scipy.sparse.csr_matrix
        Derivative matrix
    """
    if order == 1:
        diags = [-1, 0, 1]
        data = [-0.5, 0, 0.5]
    elif order == 2:
        diags = [-1, 0, 1]
        data = [1, -2, 1]
    else:
        raise NotImplementedError("Only 1st and 2nd derivatives supported")

    offsets = diags
    # erstellt ein Array mit shape(N,1) was nur den Wert val enhält (ein Wert aus data)
    data = np.array([np.full(N, val) for val in data])
    D = sp.diags(data, offsets, shape=(N, N), format='csr') / h**order

    # Boundary handling (naive, can be improved)
    # convert D to lil (list in list) for row wise editing
    D = D.tolil()
    if order == 1:
        D[0, 0:3] = np.array([-1.5, 2, -0.5]) / h**order # forward
        D[-1, -3:] = np.array([0.5, -2, 1.5]) / h**order # backward
    elif order == 2:
        D[0, 0:3] = np.array([1, -2, 1]) / h**order
        D[-1, -3:] = np.array([1, -2, 1]) / h**order
    # convert D back to csr for memory efficient computation
    D = D.tocsr()

    return D

# Diese Funktion funktioniert auch!!!
def derivative_matrix_2d(Nx, Ny, dx, dy, axis, order=1):
    """
    Returns sparse 2D derivative matrix for axis x or y.
    """
    if axis == 'x':
        Dx_1d = derivative_matrix_1d(Nx, dx, order=order)
        # Identity matrix with length of domain in y-axis
        I_ny = sp.eye(Ny)
        return sp.kron(I_ny, Dx_1d)
    elif axis == 'y':
        Dy_1d = derivative_matrix_1d(Ny, dy, order=order)
        # Identity matrix with length of domain in x-axis
        I_nx = sp.eye(Nx)
        return sp.kron(Dy_1d, I_nx)
    
def main():
    Nx = 9   # columns
    Ny = 7   # rows
    
    DIM_X = 1.0
    DIM_Y = 1.0

    # Data points in fluid domain
    x = np.linspace(start = 0, stop = DIM_X, num = Nx)
    y = np.linspace(start = 0, stop = DIM_Y, num = Ny)

    # Calculate step width from grid
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    print(dx, dy)

    Dx_1d = derivative_matrix_1d(Nx, h = dx, order=1)
    Dy_1d = derivative_matrix_1d(Ny, h = dy, order=1)

    print(f'Dx_1d:\n{Dx_1d.toarray()}')
    print(f'Dy_1d:\n{Dy_1d.toarray()}')

    Dx = derivative_matrix_2d(Nx, Ny, dx, dy, axis='x',order=1)
    Dy = derivative_matrix_2d(Nx, Ny, dx, dy, axis='y',order=1)
    Dxx = derivative_matrix_2d(Nx, Ny, dx, dy, axis='x',order=2)
    Dyy = derivative_matrix_2d(Nx, Ny, dx, dy, axis='y',order=2)

    
if __name__ == '__main__':
    main()
