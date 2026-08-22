"""Figure for the applications chapter.

Three panels. The first is the synthesis: for each task, how much of the gap
between the native low-resolution observation and the high-resolution reference
each reconstruction recovers. Expressing every task on that common scale is
what makes four different metrics comparable. The second panel resolves the
small-object result by object size, which is where super-resolution is expected
to matter most. The third gives the downlink budget.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

THESIS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(THESIS, "results")
IMAGES = os.path.join(THESIS, "Bright_Wiredu_Nuakoh_Thesis_2023-24", "images")

plt.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight",
})

INK = "#1b2a33"
ACCENT = "#0e6a6e"
GREY = "#96a3aa"


def gap_closed(lr, method, hr, lower_is_better=False):
    """Fraction of the native-LR to HR gap that ``method`` recovers."""
    if lower_is_better:
        lr, method, hr = -lr, -method, -hr
    denom = hr - lr
    return np.nan if abs(denom) < 1e-9 else (method - lr) / denom


def collect_tasks():
    """One (task, scale, bicubic, lsman) record per task and scale."""
    recs = []

    def add(task, df, col, lower=False):
        hr = float(df[df.condition == "HR reference"][col].iloc[0])
        for s in (2, 4):
            try:
                lr = float(df[df.condition == f"Native LR x{s}"][col].iloc[0])
                bi = float(df[df.condition == f"Bicubic x{s}"][col].iloc[0])
                ls = float(df[df.condition == f"LSMAN x{s}"][col].iloc[0])
            except IndexError:
                continue
            recs.append({"task": task, "scale": s,
                         "Bicubic": gap_closed(lr, bi, hr, lower),
                         "LSMAN": gap_closed(lr, ls, hr, lower)})

    p = os.path.join(RESULTS, "results_downstream.csv")
    if os.path.exists(p):
        add("Land use", pd.read_csv(p), "accuracy")
    p = os.path.join(RESULTS, "results_smallobjects.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        add("Small objects", d[d.band == "all"], "accuracy")
    p = os.path.join(RESULTS, "results_changedetection.csv")
    if os.path.exists(p):
        add("Change", pd.read_csv(p), "f1")
    p = os.path.join(RESULTS, "results_agriculture.csv")
    if os.path.exists(p):
        add("Canopy count", pd.read_csv(p), "count_error", lower=True)
    return pd.DataFrame(recs)


def panel_synthesis(ax, tasks):
    if tasks.empty:
        ax.axis("off")
        return
    order = [t for t in ["Land use", "Small objects", "Change", "Canopy count"]
             if t in set(tasks.task)]
    x = np.arange(len(order))
    w = 0.18
    for k, (scale, alpha) in enumerate(((2, 1.0), (4, 0.55))):
        sub = tasks[tasks.scale == scale].set_index("task")
        bi = [sub.Bicubic.get(t, np.nan) * 100 for t in order]
        ls = [sub.LSMAN.get(t, np.nan) * 100 for t in order]
        ax.bar(x + (k * 2 - 1.5) * w, bi, w, color=GREY, alpha=alpha,
               label=f"Bicubic $\\times{scale}$")
        ax.bar(x + (k * 2 - 0.5) * w, ls, w, color=ACCENT, alpha=alpha,
               label=f"LSMAN $\\times{scale}$")
    ax.axhline(0, color=INK, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=7.5, rotation=12)
    ax.set_ylabel("\\% of LR$\\to$HR gap recovered" if plt.rcParams["text.usetex"]
                  else "% of LR-to-HR gap recovered")
    ax.legend(fontsize=6.5, frameon=False, ncol=2)
    ax.set_title("(a)  recovery by task", fontsize=9, loc="left", color=INK)


def panel_smallobjects(ax):
    p = os.path.join(RESULTS, "results_smallobjects.csv")
    if not os.path.exists(p):
        ax.axis("off")
        return
    df = pd.read_csv(p)
    bands = [b for b in ["small", "medium", "large"] if b in set(df.band)]
    x = np.arange(len(bands))
    styles = [("Native LR x4", GREY, "o", "Native LR $\\times4$"),
              ("Bicubic x4", "#5d7480", "d", "Bicubic $\\times4$"),
              ("LSMAN x4", ACCENT, "s", "LSMAN $\\times4$"),
              ("HR reference", INK, "^", "HR reference")]
    for cond, colour, marker, label in styles:
        sub = df[df.condition == cond].set_index("band")
        y = np.array([sub.accuracy.get(b, np.nan) * 100 for b in bands])
        lo = np.array([sub.ci_lo.get(b, np.nan) * 100 for b in bands])
        hi = np.array([sub.ci_hi.get(b, np.nan) * 100 for b in bands])
        # The bands hold only a couple of hundred test objects each, so the
        # intervals are wide relative to the differences; drawing them keeps
        # the panel from implying more separation than the data support.
        ax.fill_between(x, lo, hi, color=colour, alpha=0.12, lw=0)
        ax.plot(x, y, marker=marker, color=colour, lw=1.4, ms=5, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n" for b in bands], fontsize=8)
    ax.set_ylabel("object accuracy (%)")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("(b)  small objects", fontsize=9, loc="left", color=INK)


def panel_downlink(ax):
    bw = pd.read_csv(os.path.join(RESULTS, "results_bandwidth.csv")).sort_values("scale")
    vol = bw.scene_MB_png.values
    lab = ["full resolution", "transmit at 1/2", "transmit at 1/4"]
    bars = ax.barh(np.arange(len(vol)), vol, height=0.55, color=[INK, ACCENT, ACCENT])
    bars[0].set_alpha(0.55)
    for i, r in enumerate(bw.itertuples()):
        note = f"{r.scene_MB_png:.0f} MB   {r.downlink_s:.1f} s"
        if not np.isnan(r.psnr_db):
            note += f"   {r.psnr_db:.1f} dB"
        ax.text(r.scene_MB_png + 5, i, note, va="center", fontsize=7, color=INK)
    ax.set_yticks(np.arange(len(vol)))
    ax.set_yticklabels(lab, fontsize=8)
    ax.set_xlabel("MB per $10{,}000^2$ scene")
    ax.set_xlim(0, max(vol) * 1.5)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_title("(c)  downlink budget", fontsize=9, loc="left", color=INK)


def main():
    tasks = collect_tasks()
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.0))
    panel_synthesis(axes[0], tasks)
    panel_smallobjects(axes[1])
    panel_downlink(axes[2])
    fig.tight_layout()
    out = os.path.join(IMAGES, "lsman_applications.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote", out)
    if not tasks.empty:
        print(tasks.to_string(index=False))


if __name__ == "__main__":
    main()
