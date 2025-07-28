######################################### IMPORTE ############################################

import numpy as np
import scipy.sparse as sp


######################################### FUNKTIONEN ############################################

# -------------------------------------------------------------------------------------- #
#                               1D-ABLEITUNGSMATRIX                                     #
# -------------------------------------------------------------------------------------- #
def derivative_matrix_1d(N, h, order=1, scheme='central'):
    """
    Erzeugt eine spärlich besetzte 1D-Ableitungsmatrix für Finite-Differenzen.

    Parameter
    ----------
    N : int
        Anzahl der Gitterpunkte
    h : float
        Gitterabstand (Δx oder Δy)
    order : int
        Ableitungsordnung (1 oder 2)
    scheme : str
        Nicht verwendet – vorgesehen für future: 'central', 'forward', 'backward'

    Rückgabe
    -------
    D : scipy.sparse.csr_matrix
        Spärlich besetzte Ableitungsmatrix (N x N)
    """

    # Standard-Zentraldifferenzen für 1. oder 2. Ableitung
    if order == 1:
        diags = [-1, 0, 1]
        data = [-0.5, 0, 0.5]
    elif order == 2:
        diags = [-1, 0, 1]
        data = [1, -2, 1]
    else:
        raise NotImplementedError("Nur 1. und 2. Ableitung sind implementiert.")

    # Erstellt diagonale Sparse-Matrix
    data = np.array([np.full(N, val) for val in data])
    D = sp.diags(data, diags, shape=(N, N), format='csr') / h**order

    # Ränder mit vorwärts-/rückwärts-Differenzen korrigieren
    D = D.tolil()  # effizienter für Zeilenänderungen

    if order == 1:
        D[0, 0:3] = np.array([-1.5, 2.0, -0.5]) / h
        D[-1, -3:] = np.array([0.5, -2.0, 1.5]) / h
    elif order == 2:
        D[0, 0:3] = np.array([1, -2, 1]) / h**2
        D[-1, -3:] = np.array([1, -2, 1]) / h**2

    return D.tocsr()  # zurück in effizientes Format für Matrixoperationen


# -------------------------------------------------------------------------------------- #
#                               2D-ABLEITUNGSMATRIX                                     #
# -------------------------------------------------------------------------------------- #
def derivative_matrix_2d(Nx, Ny, dx, dy, axis, order=1):
    """
    Erzeugt eine 2D-Finite-Differenzen-Ableitungsmatrix als Kroneckerprodukt.

    Parameter
    ----------
    Nx, Ny : int
        Anzahl der Gitterpunkte in x- und y-Richtung
    dx, dy : float
        Gitterabstände
    axis : 'x' oder 'y'
        Richtung der Ableitung
    order : int
        Ableitungsordnung (1 oder 2)

    Rückgabe
    -------
    D_2d : scipy.sparse.csr_matrix
        Spärlich besetzte Matrix zur Anwendung auf Flattened 2D-Felder
    """

    if axis == 'x':
        Dx_1d = derivative_matrix_1d(Nx, dx, order=order)
        I_ny = sp.eye(Ny)  # Identität für y-Achse
        return sp.kron(I_ny, Dx_1d)  # 2D-Matrix = I ⊗ Dx
    elif axis == 'y':
        Dy_1d = derivative_matrix_1d(Ny, dy, order=order)
        I_nx = sp.eye(Nx)  # Identität für x-Achse
        return sp.kron(Dy_1d, I_nx)  # 2D-Matrix = Dy ⊗ I
    else:
        raise ValueError("Axis muss 'x' oder 'y' sein.")


######################################### DEMO / TEST ############################################

def main():
    """
    Beispielhafte Anwendung zur Ausgabe der Ableitungsmatrizen.
    """
    Nx = 9   # Anzahl Spalten (x-Richtung)
    Ny = 7   # Anzahl Zeilen (y-Richtung)

    DIM_X = 1.0
    DIM_Y = 1.0

    # Gleichverteilte Gitterpunkte
    x = np.linspace(0, DIM_X, Nx)
    y = np.linspace(0, DIM_Y, Ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    print("dx =", dx, ", dy =", dy)

    # 1D-Ableitungsmatrizen anzeigen
    Dx_1d = derivative_matrix_1d(Nx, h=dx, order=1)
    Dy_1d = derivative_matrix_1d(Ny, h=dy, order=1)

    print(f'Dx_1d:\n{Dx_1d.toarray()}')
    print(f'Dy_1d:\n{Dy_1d.toarray()}')

    # 2D-Ableitungsmatrizen (flach auf Felder der Form (Ny, Nx) anwendbar)
    Dx = derivative_matrix_2d(Nx, Ny, dx, dy, axis='x', order=1)
    Dy = derivative_matrix_2d(Nx, Ny, dx, dy, axis='y', order=1)
    Dxx = derivative_matrix_2d(Nx, Ny, dx, dy, axis='x', order=2)
    Dyy = derivative_matrix_2d(Nx, Ny, dx, dy, axis='y', order=2)

    print(f"Dx shape: {Dx.shape}, Dxx shape: {Dxx.shape}")

# Nur ausführen, wenn direkt gestartet
if __name__ == "__main__":
    main()
