import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as pe

# ============================================================
# LOAD DATA
# ============================================================

raw_t              = np.load("losses_t.npy")
raw_target         = np.load("losses_target.npy")
best_losses_t      = np.load("best_losses_t.npy")

N_STEPS, N_ITER         = raw_t.shape
_,       N_TARGET_SAVES = raw_target.shape

print(f"N_STEPS={N_STEPS}, N_ITER={N_ITER}, N_TARGET_SAVES={N_TARGET_SAVES}")
print(f"best_losses_t shape:      {best_losses_t.shape}")

assert best_losses_t.ndim == 1 and len(best_losses_t) == N_STEPS

# ============================================================
# PALETA TFG
# ============================================================

navy     = "#132D5E"
gold     = "#B68C32"
ink      = "#161A26"
steel    = "#6C7A91"
feather  = "#E8ECF4"
hairline = "#D2DAE6"

COL_MAIN = navy
COL_RAW  = steel
COL_BEST = gold
COL_BAND = feather
COL_TEXT = ink

# ============================================================
# RC PARAMS
# ============================================================

mpl.rcParams.update({
    "font.family"         : "serif",
    "font.serif"          : ["Computer Modern Roman", "Palatino", "DejaVu Serif"],
    "mathtext.fontset"    : "cm",
    "text.usetex"         : False,
    "axes.linewidth"      : 0.9,
    "axes.spines.top"     : False,
    "axes.spines.right"   : False,
    "xtick.direction"     : "out",
    "ytick.direction"     : "in",
    "xtick.major.size"    : 0,
    "xtick.minor.size"    : 0,
    "ytick.major.size"    : 5,
    "ytick.minor.size"    : 3,
    "ytick.major.width"   : .8,
    "ytick.minor.width"   : .5,
    "ytick.minor.visible" : True,
    "axes.grid"           : False,
    "figure.dpi"          : 150,
    "savefig.dpi"         : 300,
    "savefig.bbox"        : "tight",
})

# ============================================================
# SMOOTHER
# ============================================================

W = 30

def rolling_median(arr, w):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - w)
        out[i] = np.median(arr[lo:i+1])
    return out

smoothed_t      = np.array([rolling_median(raw_t[s],      W) for s in range(N_STEPS)])
smoothed_target = np.array([rolling_median(raw_target[s], W) for s in range(N_STEPS)])

# ============================================================
# SHARED HELPERS
# ============================================================

def style_ax(ax):
    ax.set_facecolor("white")
    ax.spines["left"].set_color(steel)
    ax.spines["bottom"].set_color(steel)
    ax.spines["left"].set_linewidth(.9)
    ax.spines["bottom"].set_linewidth(.9)
    ax.tick_params(axis='y', colors=steel)

def add_step_bands(ax, n_steps, block_size):
    for s in range(n_steps):
        x0, x1 = s * block_size, (s+1) * block_size
        fc = COL_BAND if s % 2 == 0 else "white"
        ax.axvspan(x0, x1, facecolor=fc, alpha=.35, linewidth=0, zorder=0)

def add_step_labels(ax, n_steps, block_size):
    ax.set_xticks([])
    for s in range(n_steps):
        ax.text((s + .5) * block_size, -0.05, rf"$t_{{{s+1}}}$",
                fontsize=8, color=steel, ha="center",
                transform=ax.get_xaxis_transform())

def set_log_yaxis(ax, data_min, data_max):
    ax.set_yscale("log")
    ax.set_ylim(data_min * .4, data_max * 2.2)
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10) * .1, numticks=30))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

def add_best_markers(ax, best_y, block_size):
    T_best = len(best_y)
    best_x = np.array([s * block_size + (block_size - 1)
                       for s in range(T_best)], dtype=float)
    ax.plot(best_x, best_y,
            color=COL_BEST, linewidth=1.4, linestyle="--", alpha=.85, zorder=6)
    ax.scatter(best_x, best_y,
               s=70, color=COL_BEST, zorder=8,
               edgecolors=ink, linewidths=0.8, marker="D")

def make_legend(ax, extra_handles=None):
    h_sc = mpl.lines.Line2D([], [], marker="o", color="white",
                             markerfacecolor=COL_RAW, markersize=5,
                             alpha=.45, linewidth=0, label="Loss value")
    h_sm = mpl.lines.Line2D([], [], color=COL_MAIN, linewidth=2,
                             label=rf"Rolling median ($w={W}$)")
    handles = [h_sc, h_sm] + (extra_handles or [])
    leg = ax.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(handles),
        fontsize=8.5,
        frameon=True,
        borderpad=.6,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    leg.get_frame().set_edgecolor(hairline)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(.95)

h_best_handle = mpl.lines.Line2D([], [], marker="D", color=COL_BEST,
                                  markeredgecolor=ink, markeredgewidth=0.8,
                                  markersize=6, linewidth=1.4, linestyle="--",
                                  label="Best loss per step")

# ============================================================
# PLOT 1 — loss t → t-1
# ============================================================

fig1, ax_t = plt.subplots(figsize=(10, 4.8))
fig1.patch.set_facecolor("white")
style_ax(ax_t)
add_step_bands(ax_t, N_STEPS, N_ITER)

for s in range(N_STEPS):
    xg   = s * N_ITER + np.arange(N_ITER, dtype=float)
    mask = np.arange(0, N_ITER, 4)
    ax_t.scatter(xg[mask], raw_t[s, mask],
                 s=2, alpha=.18, color=COL_RAW, linewidths=0, rasterized=True, zorder=2)
    ln, = ax_t.plot(xg, smoothed_t[s],
                    color=COL_MAIN, linewidth=2.1, zorder=5, solid_capstyle="round")
    ln.set_path_effects([
        pe.Stroke(linewidth=4, foreground="white", alpha=.65),
        pe.Normal()
    ])

add_best_markers(ax_t, best_losses_t, N_ITER)
add_step_labels(ax_t, N_STEPS, N_ITER)
set_log_yaxis(ax_t, raw_t[:, -200:].min(), raw_t.max())
ax_t.set_xlim(-N_ITER * .008, N_STEPS * N_ITER * 1.008)
ax_t.set_ylabel(r"$\mathcal{L}(t \to t-1)$", fontsize=12, color=COL_TEXT, labelpad=10)
ax_t.set_xlabel("Denoising step",              fontsize=12, color=COL_TEXT, labelpad=22)
make_legend(ax_t, extra_handles=[h_best_handle])

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("losses_t.pdf")
plt.savefig("losses_t.png", dpi=300)
plt.close(fig1)
print("Saved → losses_t.pdf / losses_t.png")

# ============================================================
# PLOT 2 — loss t → 0  +  best loss markers
# ============================================================

fig2, ax_f = plt.subplots(figsize=(10, 4.8))
fig2.patch.set_facecolor("white")
style_ax(ax_f)
add_step_bands(ax_f, N_STEPS, N_TARGET_SAVES)

for s in range(N_STEPS):
    xg = s * N_TARGET_SAVES + np.arange(N_TARGET_SAVES, dtype=float)
    ax_f.scatter(xg, raw_target[s],
                 s=6, alpha=.35, color=COL_RAW, linewidths=0, rasterized=True, zorder=2)
    ln, = ax_f.plot(xg, smoothed_target[s],
                    color=COL_MAIN, linewidth=2.1, zorder=5, solid_capstyle="round")
    ln.set_path_effects([
        pe.Stroke(linewidth=4, foreground="white", alpha=.65),
        pe.Normal()
    ])

#add_best_markers(ax_f, best_losses_target, N_TARGET_SAVES)
add_step_labels(ax_f, N_STEPS, N_TARGET_SAVES)
set_log_yaxis(ax_f, raw_target.min(), raw_target.max())
ax_f.set_xlim(-N_TARGET_SAVES * .008, N_STEPS * N_TARGET_SAVES * 1.008)
ax_f.set_ylabel(r"$\mathcal{L}(t \to 0)$", fontsize=12, color=COL_TEXT, labelpad=10)
ax_f.set_xlabel("Denoising step",           fontsize=12, color=COL_TEXT, labelpad=22)
make_legend(ax_f, extra_handles=[])

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("losses_target.pdf")
plt.savefig("losses_target.png", dpi=300)
plt.close(fig2)
print("Saved → losses_target.pdf / losses_target.png")