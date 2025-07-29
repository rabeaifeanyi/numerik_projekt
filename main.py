######################################### IMPORTE ############################################

# -------------------------------------------------------------------------------------- #
#                                         MODULE                                         #
# -------------------------------------------------------------------------------------- #
import os
import numpy as np
import scipy.sparse as sp
import argparse
import sys
from datetime import datetime

# -------------------------------------------------------------------------------------- #
#                                     EIGNE SKRIPTE                                      #
# -------------------------------------------------------------------------------------- #
# Eigenes Modul mit Ableitungsmatrizen importieren aus fdm.py
from fdm import derivative_matrix_2d

# Funktionen zum Plotten und Erstellen von GIFs aus plotting.py importieren
from plotting import plot_velocity_field, create_gif_from_folder


####################################### FUNKTIONEN ############################################

# -------------------------------------------------------------------------------------- #
#                              PHYSIKALISCHE HILFSFUNKTIONEN                             #
# -------------------------------------------------------------------------------------- #
def solve_poisson(omega_flat, Dxx, Dyy, Nx, Ny, mask_boundary, psi_bc):
    """
    Löst die Poisson-Gleichung Δψ = -ω mit Dirichlet-Randbedingungen
    auf einem 2D-Gitter mit Hilfe der finiten Differenzen Methode.
    """
    Lap = Dxx + Dyy
 
    # Matrizen vektorisieren
    B = np.reshape(mask_boundary, (Ny * Nx))
    psi_bc_flat = np.reshape(psi_bc, (Ny * Nx))

    notB = 1.0 - B

    A = sp.diags(notB) @ Lap + sp.diags(B)  # apply identity on boundary
    b = notB * - omega_flat + B * psi_bc_flat

    psi_flat = sp.linalg.spsolve(A, b)
    return psi_flat

def compute_velocity(psi_flat, Dx, Dy,
                     U_top, U_bottom, U_left, U_right,
                     mask_top, mask_bottom, mask_left, mask_right):
    """
    Berechnet u und v-Geschwindigkeiten aus ψ (Stromfunktion).
    Berücksichtigt individuell einstellbare Randgeschwindigkeiten für:
    - obere, untere, linke und rechte Wand.
    """

    # Wände vektorisieren
    W_top = mask_top.ravel()
    W_bottom = mask_bottom.ravel()
    W_left = mask_left.ravel()
    W_right = mask_right.ravel()

    # Punkte, die nicht an einer Wand liegen
    not_boundary = 1.0 - (W_top + W_bottom + W_left + W_right)

    # Berechne u = ∂ψ/∂y + Randgeschwindigkeit
    u_flat = (Dy @ psi_flat) * not_boundary \
             + U_top * W_top + U_bottom * W_bottom

    # Berechne v = -∂ψ/∂x + Randgeschwindigkeit
    v_flat = (-Dx @ psi_flat) * not_boundary \
             + U_left * W_left + U_right * W_right

    return u_flat, v_flat

def vorticity_rhs(omega_flat, u_flat, v_flat, Dx, Dy, Dxx, Dyy, Nx, Ny, nu, mask_boundary):
    """
    Berechnet die rechte Seite der Vorticity-Gleichung:
    ∂ω/∂t = -u ∂ω/∂x - v ∂ω/∂y + ν Δω
    """
    B = np.reshape(mask_boundary, (Ny * Nx))
    notB = 1.0 - B

    diffusion = nu * (Dxx + Dyy)
    convection = sp.diags(u_flat) @ Dx + sp.diags(v_flat) @ Dy

    rhs_flat = (diffusion - convection) @ (
        notB * omega_flat +
        B * (Dx @ v_flat - Dy @ u_flat) 
    )

    return rhs_flat

def compute_rhs(omega_flat, Dx, Dy, Dxx, Dyy, Nx, Ny, mask_boundary, psi_bc, U_top, U_bottom, U_left, U_right, mask_top,  mask_bottom, mask_left, mask_right, nu):
    """
    Wrapper für vollständige Berechnung der RHS der Vorticity-Gleichung.
    """
    psi_flat = solve_poisson(omega_flat, Dxx, Dyy, Nx, Ny, mask_boundary, psi_bc)
    u_flat, v_flat = compute_velocity(psi_flat, Dx, Dy, U_top, U_bottom, U_left, U_right, mask_top, mask_bottom, mask_left, mask_right)
    rhs_flat = vorticity_rhs(omega_flat, u_flat, v_flat, Dx, Dy, Dxx, Dyy, Nx, Ny, nu, mask_boundary)
    
    return rhs_flat

# -------------------------------------------------------------------------------------- #
#                                    RANDBEDINGUNGEN                                     #
# -------------------------------------------------------------------------------------- #
def init_velocity(type="constant"):
    """
    Gibt je eine Funktion zurück, die die Wandgeschwindigkeit in Abhängigkeit von der Zeit liefert.
    """
    if type == "constant":
        def U_top(t): return 1.0                                # konstant
        def U_bottom(t): return 0.0                             # ruht
        def U_left(t): return 0.0                               # ruht
        def U_right(t): return 0.0                              # ruht

    elif type == "positive-sine":
        def U_top(t): return 1 + np.sin(2 * np.pi * t / 2000)   # sinusförmig
        def U_bottom(t): return 0.0                             # ruht
        def U_left(t): return 0.0                               # ruht
        def U_right(t): return 0.0                              # ruht
    
    elif type == "sine":
        def U_top(t):
            if t == 0: return 1.0
            else: return np.sin(2 * np.pi * t / 2000)           # sinusförmig
        def U_bottom(t): return 0.0                             # ruht
        def U_left(t): return 0.0                               # ruht
        def U_right(t): return 0.0                              # ruht

    elif type == "top-bottom":
        def U_top(t): return 1.0                                # konstant
        def U_bottom(t): return 1.0                             # konstant
        def U_left(t): return 0.0                               # ruht
        def U_right(t): return 0.0                              # ruht

    elif type == "test":
        def U_top(t): return 1.0                                # konstant
        def U_bottom(t): return 0.0                             # ruht
        def U_left(t): return 1.0                               # konstant
        def U_right(t): return 0.0                              # ruht
    
    return U_top, U_bottom, U_left, U_right

# -------------------------------------------------------------------------------------- #
#                                 ZEITSCHRITT-FUNKTIONEN                                 #
# -------------------------------------------------------------------------------------- #
def euler_step(f, y, dt):
    """
    Einfacher expliziter Euler-Zeitintegrator.
    """
    y_new = y + dt * f(y) 
    
    return y_new

def rk4_step(f, y, dt):
    """
    Klassisches Runge-Kutta Verfahren.
    """
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    y_new = y + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return y_new

# -------------------------------------------------------------------------------------- #
#                                    HILFSFUNKTIONEN                                     #
# -------------------------------------------------------------------------------------- #
def parse_args():
    """
    Kommandozeilenargumente: steuern GIF, Matrixspeicherung, Plot usw.
    """
    parser = argparse.ArgumentParser(description="Simulation")

    parser.add_argument("--live", action="store_true", help="Show live plot")
    parser.add_argument("--gif", action="store_true", help="Create GIF at end")
    parser.add_argument("--save", action="store_true", help="Save result matrices")
    parser.add_argument("--result_print", action="store_true", help="Print result matrices")
    
    return parser.parse_args()

def progress_bar(t, steps, bar_length=40):
    """
    Fortschrittsbalken für CLI-Ausgabe.
    """
    percent = t / steps
    arrow = '=' * int(percent * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    sys.stdout.write(f'\r[{arrow}{spaces}] {int(percent * 100)}%  (t={t})')
    sys.stdout.flush()

######################################### HAUPTPROGRAMM #####################################################
def main():
    # Initialisierung vom Parser
    args = parse_args()
    
    ##################################### Einstellungen #################################################
    # Du kannst hier einstellen, was genau simuliert werden soll!

    # --- Simulationsbereich ---
    DIM_X, DIM_Y = 1.0, 1.0           # Größe des Simulationsfelds (Breite x Höhe)

    # --- Auflösung des Gitters ---
    NX, NY = 80, 80                   # Gitterauflösung 

    # --- Simulationsdauer ---
    N_INTER = 10000                   # Anzahl Zeitschritte

    # --- Speicherintervall ---
    SAVE_INTERVAL = 100               # Alle wie viele Zeitschritte soll ein Snapshot gespeichert werden?

    # --- CFL-Zahl ---
    CFL = 0.3                         # Stabilitätsbedingung → bei Werten über 1.0 wird es instabil (so kann man einen "crash" verursachen)

    # --- Reynolds-Zahl ---
    RE = 400                          # Beeinflusst die Turbulenz/Stabilität der Strömung

    # --- Randbedingungen ("Bewegung der Wände") ---
    # Beispiele:
    #   - "constant"        → Obere Wand konstant (z. B. u = 1), Rest ruht → klassischer Fall
    #   - "sine"            → Obere Wand sinusförmig 
    #   - "sine-positive"   → Nur positive Sinusbewegung, bzw. sinus plus 1, also kein Richtungswechsel
    #   - "top-bottom"      → Oben + unten bewegen sich, hier von links nach rechts
    #   - "test"            → Zum testen
    # Wenn du was testen willst einfach bei init_velocity() hinzufügen
    RANDBEDINGUNG = "sine"

    # --- Zeitintegrationsmethode ---
    #   - "euler"           → Einfach, aber ungenau und instabil bei großen Zeitschritten
    #   - "runge-kutta"     → Stabiler und genauer, noch nicht getestet #TODO
    TIMESTEP_METHOD = "euler"
    #####################################################################################################
    
    # Zeitstempel für eindeutige Ordnerstruktur
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = f"plots/plots_CFL{CFL:.2f}_Re{RE}_{TIMESTEP_METHOD}_{RANDBEDINGUNG}_{timestamp}".replace(".", "p")
    os.makedirs(folder)

    # --- Gitter erstellen ---
    x = np.linspace(start = 0, stop = DIM_X, num = NX)
    y = np.linspace(start = 0, stop = DIM_Y, num = NY)
    X, Y = np.meshgrid(x, y)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # --- Randbedingungen und Parameter ---
    psi_bc = np.zeros((NY, NX))     # Stromfunktion an den Rändern

    U_top_func, U_bottom_func, U_left_func, U_right_func = init_velocity(RANDBEDINGUNG)

    nu = abs(U_top_func(0)) * DIM_X / RE   # Kinematische Viskosität
    dt = CFL * dx / abs(U_top_func(0))     # Zeitschritt aus CFL-Bedingung

    # --- Initialisierung ---
    np.random.seed(1234)
    omega_init = 1e-2 * (np.random.rand(NY, NX) - 0.5)
    psi_init = np.zeros((NY, NX))
    u_init = np.zeros((NY, NX))
    v_init = np.zeros((NY, NX))

    # --- Masken für Randbedingungen ---
    mask_boundary = np.zeros((NY, NX), dtype = int)
    mask_boundary[0,:] = 1
    mask_boundary[-1,:] = 1
    mask_boundary[:,0] = 1
    mask_boundary[:,-1] = 1

    mask_top = np.zeros((NY, NX), dtype = int)
    mask_top[0, :] = 1

    mask_bottom = np.zeros((NY, NX), dtype=int)
    mask_bottom[-1, :] = 1

    mask_left = np.zeros((NY, NX), dtype=int)
    mask_left[:, 0] = 1

    mask_right = np.zeros((NY, NX), dtype=int)
    mask_right[:, -1] = 1

    # --- Ableitungsmatrizen ---
    Dx = derivative_matrix_2d(NX, NY, dx, dy, axis='x',order=1)
    Dy = derivative_matrix_2d(NX, NY, dx, dy, axis='y',order=1)
    Dxx = derivative_matrix_2d(NX, NY, dx, dy, axis='x',order=2)
    Dyy = derivative_matrix_2d(NX, NY, dx, dy, axis='y',order=2)

    # --- Flattened Felder für Vektoroperationen ---
    omega_flat = np.reshape(omega_init, (NY * NX))
    psi_flat = np.reshape(psi_init, (NY * NX))
    u_flat = np.reshape(u_init, (NY * NX))
    v_flat = np.reshape(v_init, (NY * NX))

    # --- Ergebnislisten (für GIF oder Speicherung) ---
    omega_list, psi_list, u_list, v_list = [], [], [], []   

    # --- Protokollierung ---
    param_text = (
        f"Simulation Parameters:\n"
        f"  - Grid Size: {NX} x {NY}\n"
        f"  - DIM_X: {DIM_X}\n"
        f"  - DIM_Y: {DIM_Y}\n"
        f"  - dx: {dx}\n"
        f"  - dy: {dy}\n"
        f"  - CFL: {CFL}\n"
        f"  - Re: {RE}\n"
        f"  - Nu: {nu}\n"
        f"  - Time Step: {dt}\n"
        f"  - Total Iterations: {N_INTER}\n"
    )
    print(param_text)
    with open(f"{folder}/simulation_parameters.txt", "w") as f:
        f.write(param_text)

    # --- Haupt-Zeitschleife ---
    for t in range(N_INTER):
        def rhs_func(omega):
            return compute_rhs(
                omega_flat=omega,
                Dx=Dx,
                Dy=Dy,
                Dxx=Dxx,
                Dyy=Dyy,
                Nx=NX,
                Ny=NY,
                mask_boundary=mask_boundary,
                mask_top=mask_top,
                mask_bottom=mask_bottom, 
                mask_left=mask_left, 
                mask_right=mask_right,
                psi_bc=psi_bc,
                U_top=U_top_func(t),
                U_bottom=U_bottom_func(t),
                U_left=U_left_func(t),
                U_right=U_right_func(t),
                nu=nu
            )

        if TIMESTEP_METHOD == "euler":
            # --- Euler ---
            omega_flat[:] = euler_step(rhs_func, omega_flat, dt)
        elif TIMESTEP_METHOD == "runge-kutta":
            # --- RK4 ---
            omega_flat = rk4_step(rhs_func, omega_flat, dt)

        if t % SAVE_INTERVAL == 0:
            progress_bar(t, N_INTER)

            # Update psi, u, v
            psi_flat = solve_poisson(omega_flat, Dxx, Dyy, NX, NY, mask_boundary, psi_bc)
            u_flat, v_flat = compute_velocity(psi_flat, Dx, Dy, U_top_func(t), U_bottom_func(t), U_left_func(t), U_right_func(t), mask_top, mask_bottom, mask_left, mask_right)

            # Speichern für spätere Nutzung
            psi_list.append(psi_flat.reshape((NY, NX)))
            u_list.append(u_flat.reshape((NY, NX)))
            v_list.append(v_flat.reshape((NY, NX)))
            omega_list.append(omega_flat.reshape((NY, NX)))

            # Geschwindigkeitsbetrag
            u_mag = np.sqrt(u_list[-1] ** 2 + v_list[-1] ** 2)
            
            if args.gif:
                plot_velocity_field(X, Y, u_list[-1], v_list[-1], u_mag, t, folder)

    # --- GIF-Erstellung am Ende ---
    if args.gif:
        create_gif_from_folder(folder, time=f"{timestamp}")

    # --- Ergebnisse speichern ---
    if args.save:
        np.savez_compressed(
            f"{folder}/final_state.npz",
            psi=psi_list[-1],
            u=u_list[-1],
            v=v_list[-1],
            omega=omega_list[-1]
        )

    # --- Ergebnisse ausgeben ---
    if args.result_print:
        print(f'Final psi:\n{psi_list[-1]}')
        print(f'Final u:\n{u_list[-1]}')
        print(f'Final v:\n{v_list[-1]}')
        print(f'Final omega:\n{omega_list[-1]}')


if __name__ == '__main__':
    main()