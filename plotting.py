######################################### IMPORTE ############################################

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib import animation as animation
from matplotlib.ticker import FormatStrFormatter
import imageio.v2 as imageio


####################################### KONSTANTEN ############################################

# Farbkonturen für die Geschwindigkeitsvisualisierung
CONTOUR_LEVELS = np.linspace(0, 1, 30)

# Standardfarbe für Stromlinienplots
STREAM_COLOR = "tab:cyan"

# Schrittweite beim Darstellen des Vektorfelds
VECTOR_FIELD_STEP = 3


####################################### FUNKTIONEN ############################################

# -------------------------------------------------------------------------------------- #
#                            VISUALISIERUNG DES GESCHWINDIGKEITSFELDS                    #
# -------------------------------------------------------------------------------------- #
def plot_velocity_field(X, Y, u, v, u_mag, t, folder,
                        vector_field_step=6, plot_quiver=True,
                        plot_stream=False, plot_title=None):
    """
    Erstellt ein Plot des Geschwindigkeitsfelds zu einem gegebenen Zeitpunkt.
    
    Parameter:
    - X, Y: Gitterkoordinaten
    - u, v: Geschwindigkeit in x- und y-Richtung (2D)
    - u_mag: Betrag der Geschwindigkeit (für Farbkontur)
    - t: Zeitschrittindex (für Dateinamen)
    - folder: Speicherpfad für den Plot
    - vector_field_step: Ausdünnung des Vektorfelds (größer = weniger Pfeile)
    - plot_quiver: True = Vektorfeld anzeigen
    - plot_stream: True = Stromlinien anzeigen
    - plot_title: Optionaler Titel für den Plot
    """

    fig, ax = plt.subplots()
    ax.invert_yaxis()

    # Konturplot der Geschwindigkeit (mit PowerNorm "verstärkt" kleine Werte)
    norm = PowerNorm(gamma=0.8, vmin=0) # TODO herausfinden, warum Maximalbereiche weiß werden
    ax.contourf(X, Y, u_mag, levels=CONTOUR_LEVELS, cmap="turbo", norm=norm)

    # Optional: Stromlinien darstellen
    if plot_stream:
        ax.streamplot(X, Y, u, v, density=1, color="magenta")

    # Optional: Vektorfeld darstellen
    if plot_quiver:
        mag_max = 0.6  # Maximale Pfeillänge begrenzen

        # Faktor zur Begrenzung extrem langer Pfeile
        factor = np.minimum(1, mag_max / (u_mag + 1e-8))
        U_clipped = u * factor
        V_clipped = v * factor

        amplify = 2  
        U_scaled = U_clipped * amplify
        V_scaled = V_clipped * amplify

        # Vektorfeld (nur an ausgewählten Gitterpunkten zur Übersichtlichkeit)
        ax.quiver(
            X[1:-1:vector_field_step, 1:-1:vector_field_step],
            Y[1:-1:vector_field_step, 1:-1:vector_field_step],
            U_scaled[1:-1:vector_field_step, 1:-1:vector_field_step],
            -V_scaled[1:-1:vector_field_step, 1:-1:vector_field_step],  # invertiere y-Richtung
            color='white',
            scale=9, # hier kann man dran schrauben, wenn man mit den Längen der Pfeilen unzufrieden ist (höher macht kürzer)
        )

    # Optionaler Plot-Titel
    if plot_title:
        ax.set_title(f'{plot_title}')

    # Speicherort erzeugen (falls nicht vorhanden)
    os.makedirs(folder, exist_ok=True)

    # Bild abspeichern
    plt.savefig(f"{folder}/t{t:05}_contour.png")
    plt.close()

def plot_velocity_field_fancy(X, Y, u, v, u_mag, t, folder,
                              vector_field_step=6,
                              title_base="Geschwindigkeitsfeld",
                              filename_base="velocity_t"):
    """
    Erzeugt zwei PDF-Plots des Geschwindigkeitsfelds für die Abgabe
    """

    plt.rcParams["font.family"] = "Times New Roman"
    norm = PowerNorm(gamma=0.8, vmin=0)
    os.makedirs(folder, exist_ok=True)

    # ----------- Plot 1: Quiver (Vektorfeld) ----------- #
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.invert_yaxis()
    contour1 = ax1.contourf(X, Y, u_mag, levels=CONTOUR_LEVELS, cmap="turbo", norm=norm)
    cbar1 = fig1.colorbar(contour1, ax=ax1, fraction=0.046, pad=0.04, format=FormatStrFormatter('%.2f'))
    cbar1.set_label("Geschwindigkeitsbetrag $|\\vec{u}|$")
    ax1.set_xlabel("x-Richtung")
    ax1.set_ylabel("y-Richtung")
    ax1.set_title(f"{title_base} (Vektorfeld)")

    mag_max = 0.6
    factor = np.minimum(1, mag_max / (u_mag + 1e-8))
    U_scaled = u * factor * 2
    V_scaled = v * factor * 2

    ax1.quiver(
        X[1:-1:vector_field_step, 1:-1:vector_field_step],
        Y[1:-1:vector_field_step, 1:-1:vector_field_step],
        U_scaled[1:-1:vector_field_step, 1:-1:vector_field_step],
        -V_scaled[1:-1:vector_field_step, 1:-1:vector_field_step],
        color='white',
        scale=9
    )

    fig1.tight_layout()
    fig1.savefig(os.path.join(folder, f"{filename_base}{t:05}_quiver.pdf"), format="pdf", dpi=300)
    plt.close(fig1)

    # ----------- Plot 2: Streamlines ----------- #
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.invert_yaxis()
    contour2 = ax2.contourf(X, Y, u_mag, levels=CONTOUR_LEVELS, cmap="turbo", norm=norm)
    cbar2 = fig2.colorbar(contour2, ax=ax2, fraction=0.046, pad=0.04, format=FormatStrFormatter('%.2f'))
    cbar2.set_label("Geschwindigkeitsbetrag $|\\vec{u}|$")
    ax2.set_xlabel("x-Richtung")
    ax2.set_ylabel("y-Richtung")
    ax2.set_title(f"{title_base} (Stromlinien)")

    ax2.streamplot(X, Y, u, v, density=1.2, color='white', linewidth=0.8)

    fig2.tight_layout()
    fig2.savefig(os.path.join(folder, f"{filename_base}{t:05}_stream.pdf"), format="pdf", dpi=300)
    plt.close(fig2)


# -------------------------------------------------------------------------------------- #
#                                GIF AUS BILDERN ERSTELLEN                               #
# -------------------------------------------------------------------------------------- #
def create_gif_from_folder(image_folder, output_folder="gifs"):
    """
    Erstellt ein animiertes GIF aus allen gespeicherten Einzelbildern (PNG) in einem Ordner.

    Parameter:
    - image_folder: Ordner mit Bildern
    - output_folder: Zielordner für das GIF
    - time: Optionaler Zeitstempel für den Dateinamen
    """

    # Alle Bilddateien (sortiert nach Zeit)
    image_files = sorted(glob.glob(os.path.join(image_folder, 't*_contour.png')))

    # Zielpfad des GIFs
    output_gif = os.path.join(output_folder, f'{os.path.basename(image_folder)}.gif')

    # GIF schreiben 
    os.makedirs(output_folder, exist_ok=True)
    with imageio.get_writer(output_gif, mode='I', duration=0.05, loop=0) as writer:
        for filename in image_files:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"GIF gespeichert unter: {output_gif}")
