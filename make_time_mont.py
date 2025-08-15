import numpy as np
import matplotlib.pyplot as plt
import os

def make_time_montage_stream(
    npz_path,
    t_start=0,
    t_end=2450,
    t_step=50,
    take_every=3,       # <-- jeden dritten Snapshot verwenden
    rows=3,
    cols=4,
    density=1.5,
    linewidth=0.7,
    caption_font="Times New Roman",
    caption_size=8,
    out_path=None
):
    data = np.load(npz_path)
    u_list = data["u_list"]
    v_list = data["v_list"]
    NX = int(data["NX"]); NY = int(data["NY"])
    DIM_X = float(data["DIM_X"]); DIM_Y = float(data["DIM_Y"])
    SAVE_INTERVAL = int(data["SAVE_INTERVAL"])

    x = np.linspace(0, DIM_X, NX)
    y = np.linspace(0, DIM_Y, NY)
    X, Y = np.meshgrid(x, y)

    # gewünschte Zeitpunkte (subsample: jeden dritten)
    desired_ts = list(range(t_start, t_end + 1, t_step))[::take_every]
    frame_idx = [(t, t // SAVE_INTERVAL) for t in desired_ts if 0 <= t // SAVE_INTERVAL < len(u_list)]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8), squeeze=False)
    plt.subplots_adjust(wspace=0.08, hspace=0.20)  # etwas Platz nach unten für Untertitel

    for k, (t, idx) in enumerate(frame_idx[: rows * cols]):
        r = k // cols
        c = k % cols
        ax = axes[r, c]

        # Streamplot ohne Pfeile
        ax.streamplot(X, Y, u_list[idx], v_list[idx],
                      density=density, color='black', linewidth=linewidth, arrowstyle="-")
        ax.invert_yaxis()
        ax.set_aspect('equal', adjustable='box')
        # keine Achsen zeichnen, aber wir fügen den Untertitel manuell unterhalb ein
        ax.set_xticks([]); ax.set_yticks([]); ax.set_axis_off()

        # Untertitel UNTER dem Plot, zentriert, Times New Roman, klein, nah dran
        ax.text(0.5, -0.04, f"t = {t}", transform=ax.transAxes,
                ha='center', va='top', fontname=caption_font, fontsize=caption_size)

    # übrige Felder leeren
    total_slots = rows * cols
    for k in range(len(frame_idx), total_slots):
        r = k // cols
        c = k % cols
        axes[r, c].axis("off")

    if out_path is None:
        base = os.path.splitext(os.path.basename(npz_path))[0]
        out_path = f"{base}_montage_stream_noarrows_grid{rows}x{cols}_every{take_every}.png"

    # kein suptitle
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Montage gespeichert unter: {out_path}")


make_time_montage_stream(
    "/Users/reschenhagen/Documents/Uni/numerik_projekt/plots/plots_CFL0p50_Re400_euler_sine_2025-08-15_20-04-27/final_state.npz",
    t_start=0, t_end=2450, t_step=50, take_every=4, rows=3, cols=4,
    out_path="montage_sine_stream_noarrows_0_2450_grid3x4.png"
)

make_time_montage_stream(
    "/Users/reschenhagen/Documents/Uni/numerik_projekt/plots/plots_CFL0p50_Re400_euler_positive-sine_2025-08-15_20-05-19/final_state.npz",
    t_start=0, t_end=2450, t_step=50, take_every=4, rows=3, cols=4,
    out_path="montage_positive_sine_stream_noarrows_0_2450_grid3x4.png"
)

make_time_montage_stream(
    "/Users/reschenhagen/Documents/Uni/numerik_projekt/plots/plots_CFL0p50_Re400_euler_pulse_2025-08-15_19-55-06/final_state.npz",
    t_start=0, t_end=2450, t_step=50, take_every=4, rows=3, cols=4,
    out_path="montage_pulse_stream_noarrows_0_2450_grid3x4.png"
)

