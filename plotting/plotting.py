import matplotlib.pyplot as plt
import os

#TODO cooler machen

def plot_velocity_field(X, Y, u, v, u_mag, t, folder):
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.contourf(X, Y, u_mag)
    ax.streamplot(X, Y, u, v, density=1.5, color="tab:cyan")
    ax.set_title(f't = {t}')
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/t{t}_contour.png")
    plt.close()
