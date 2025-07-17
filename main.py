import argparse
import os
from simulation import Grid, compute_rhs, euler_step, make_masks
from config import *
from plotting import plot_velocity_field

def parse_args():
    parser = argparse.ArgumentParser(description="Simulation")

    parser.add_argument("--log", action="store_true", help="Enable logging")
    parser.add_argument("--live", action="store_true", help="Show live plot")
    parser.add_argument("--gif", action="store_true", help="Create GIF at end")
    parser.add_argument("--mp3", action="store_true", help="Create MP3 at end")
    
    return parser.parse_args()

def main():
    args = parse_args()

    grid = Grid(Nx, Ny, Lx, Ly)
    dt = CFL * grid.dx / np.abs(U_tang)
    NU = np.abs(U_tang) * Lx / Re

    omega_flat = np.reshape(omega_init, (Ny * Nx))
    psi_flat = np.reshape(psi_init, (Ny * Nx))
    u_flat = np.reshape(u_init, (Ny * Nx))
    v_flat = np.reshape(v_init, (Ny * Nx))
    masks = make_masks(Nx, Ny, problem)
    psi_bc = np.zeros((Ny, Nx))

    os.makedirs(folder, exist_ok=True)

    for t in range(steps):
        U_sin = U_tang * np.sin(2 * np.pi * t / 500)

        def rhs_func(omega):
            return compute_rhs(
                omega_flat=omega,
                Dx=grid.Dx,
                Dy=grid.Dy,
                Dxx=grid.Dxx,
                Dyy=grid.Dyy,
                Nx=Nx,
                Ny=Ny,
                mask_boundary= masks['boundary'],
                mask_top=masks['top'],
                mask_walls=masks['walls'], #TODO um zu generalisieren könnten wir das splitten oder noch cooler machen
                psi_bc=psi_bc,
                U_tang=U_sin,
                U_noSlip=U_noSlip,
                NU=NU
            )

        omega_flat = euler_step(rhs_func, omega_flat, dt)

        if t % save_interval == 0:
            if args.log:
                print(f"[t = {t}]")
            from simulation import solve_poisson, compute_velocity
            psi_flat = solve_poisson(omega_flat, grid.Dxx, grid.Dyy, Nx, Ny, masks['boundary'], psi_bc)
            u_flat, v_flat = compute_velocity(psi_flat, grid.Dx, grid.Dy, Nx, Ny, U_tang, U_noSlip, masks['top'], masks['walls'])
            u = u_flat.reshape((Ny, Nx))
            v = v_flat.reshape((Ny, Nx))
            u_mag = np.sqrt(u ** 2 + v ** 2)
            plot_velocity_field(grid.X, grid.Y, u, v, u_mag, t, folder)
            
            if args.live:
                import matplotlib.pyplot as plt
                plt.clf()
                plt.contourf(grid.X, grid.Y, u_mag)
                plt.pause(0.01)
    
    # if args.gif:
    #     from plotting.create_gif import create_gif_from_folder
    #     create_gif_from_folder(folder)

    # if args.mp3:
    #     from plotting.create_mp3 import create_mp3_from_data
    #     create_mp3_from_data(folder)

if __name__ == "__main__":
    main()
