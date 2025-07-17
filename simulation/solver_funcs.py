import numpy as np
import scipy.sparse as sp

def solve_poisson(omega_flat, Dxx, Dyy, Nx, Ny, mask_boundary, psi_bc):
    Lap = Dxx + Dyy
    B = np.reshape(mask_boundary, (Ny * Nx))
    psi_bc_flat = np.reshape(psi_bc, (Ny * Nx))
    notB = 1.0 - B
    A = sp.diags(notB) @ Lap + sp.diags(B) 
    b = notB * - omega_flat + B * psi_bc_flat
    psi_flat = sp.linalg.spsolve(A, b)
    return psi_flat

def compute_velocity(psi_flat, Dx, Dy, Nx, Ny, u_tang, u_noSlip, mask_top, mask_walls):
    W_top = np.reshape(mask_top, (Ny * Nx))
    notW_top = 1.0 - W_top
    W_walls = np.reshape(mask_walls, (Ny * Nx))
    notW_walls = 1.0 - W_walls
    u_flat = Dy @ psi_flat  * (notW_top * notW_walls) + u_tang * W_top
    v_flat = - Dx @ psi_flat * (notW_top * notW_walls) + u_noSlip * (W_top * W_walls)
    return u_flat, v_flat

def vorticity_rhs(omega_flat, u_flat, v_flat, Dx, Dy, Dxx, Dyy, Nx, Ny, NU, mask_boundary):
    B = np.reshape(mask_boundary, (Ny * Nx))
    notB = 1.0 - B
    diffusion = NU * (Dxx + Dyy)
    convection = sp.diags(u_flat) @ Dx + sp.diags(v_flat) @ Dy
    rhs_flat = (diffusion - convection) @ (
        notB * omega_flat +
        B * (Dx @ v_flat - Dy @ u_flat) 
    )
    return rhs_flat

def compute_rhs(omega_flat, Dx, Dy, Dxx, Dyy, Nx, Ny, mask_boundary, psi_bc, U_tang, U_noSlip, mask_top, mask_walls, NU):
    psi_flat = solve_poisson(omega_flat, Dxx, Dyy, Nx, Ny, mask_boundary, psi_bc)
    u_flat, v_flat = compute_velocity(psi_flat, Dx, Dy, Nx, Ny, U_tang, U_noSlip, mask_top, mask_walls)
    rhs_flat = vorticity_rhs(omega_flat, u_flat, v_flat, Dx, Dy, Dxx, Dyy, Nx, Ny, NU, mask_boundary)
    return rhs_flat

def euler_step(f, y, dt):
    return y + dt * f(y)

def rk4_step(f, y, dt):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
