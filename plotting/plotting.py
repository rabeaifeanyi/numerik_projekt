import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib import animation as animation
from pathlib import Path

#TODO cooler machen
CONTOUR_LEVELS = np.linspace(0, 1, 40)
STREAM_COLOR = "tab:cyan"

def plot_velocity_field(X, Y, u, v, u_mag, t, folder):
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    ax.contourf(X, Y, u_mag,levels=CONTOUR_LEVELS,cmap="viridis", vmax=.7)
    #ax.streamplot(X, Y, u, v, density=1.5, color="tab:cyan")
    #ax.set_title(f't = {t}')
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/t{t:05}_contour.png")
    plt.close()



FIGSIZE = (12,12)
VECTOR_STEP = 3
CONTOUR_LEVELS = np.linspace(0, 1, 21)
STREAM_COLOR = "tab:cyan"

class VelocityRenderer:
    def __init__(self, folder, X, Y):
        self.X, self.Y = X, Y
        self.frames = []
        self.folder = Path(folder)
        self.folder.mkdir(exist_ok=True, parents=True)

        self.fig = plt.figure()
        self.grid_specs = self.fig.add_gridspec(6, 1)
        self.ax = self.fig.add_subplot(self.grid_specs[0, 0])
        self.ax.set_aspect("equal")
        self.ax.set_xlim(X.min(), X.max())
        self.ax.set_ylim(Y.min(), Y.max())
        self.ax.invert_yaxis()

    def capture_frame(self, u, v, t):
        self.ax.clear()

        u_mag = np.sqrt(u ** 2 + v ** 2)
        contour_plot = self.ax.contourf(
            self.X, self.Y, u_mag,
            levels=CONTOUR_LEVELS,
            cmap="viridis"
        )

        stream = self.ax.streamplot(
            self.X, self.Y, u, v,
            density=1.5,
            color=STREAM_COLOR
        )
        plt.savefig(f"{self.folder}/t{t}_contour.png")

    #     self.ax.set_title(f"Velocity Field at t = {t:.2f} s")
    #     self.ax.axis("off")

    #     self.frames.append([contour_plot])


    # def finalize(self):
    #     print(len(self.frames), "\n\n")
    #     filename = self.folder / f"t_velocity.gif"
    #     ani = animation.ArtistAnimation(self.fig, self.frames, blit=False)
    #     ani.save(filename, writer=animation.PillowWriter())
    #     plt.close(self.fig)
    #     print(f"Animation saved at {filename}")