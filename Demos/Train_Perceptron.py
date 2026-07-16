#!/usr/bin/env python3
"""
Code written by Nicholas J. Cooper with help from Claude Code (VSCode frontend only) w/ backend model Qwen3.6-27B-Q6_K (available from here: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/blob/main/Qwen3.6-27B-Q6_K.gguf)

Interactive Perceptron Training Visualizer.

Demonstrates the perceptron learning algorithm step-by-step with an
interactive matplotlib window.  Supports 2-D and 3-D randomly generated
binary-class datasets with configurable separability.

Two-phase step workflow:
    Press 1: highlight candidate point (yellow ring), show prediction.
    Press 2: update weights if misclassified (green flash), advance.

Usage:
    python Train_Perceptron.py
"""

import argparse

import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider


# ─────────────────────────────────────────────
# Dataset Generator
# ─────────────────────────────────────────────
class DatasetGenerator:
    """Generate random binary-class datasets (two Gaussian clusters)."""

    @staticmethod
    def generate(dim: int = 2,
                 n_samples: int = 80,
                 separation: float = 0.8,
                 noise: float = 1.0,
                 seed: int | None = None):
        """Return X (2*n_samples × dim), y (2*n_samples,) with labels ±1.

        Parameters
        ----------
        dim : Dimensionality (2 or 3).
        n_samples : Points *per class*.
        separation : Distance factor between cluster centres.
            0.0 → overlapping (inseparable);  ≥2.0 → well separated.
        noise : Standard-deviation of the Gaussian spread.
        seed : RNG seed for reproducibility.
        """
        rng = np.random.default_rng(seed)

        offset = separation * noise
        mean_pos = np.zeros(dim);  mean_pos[0] = offset
        mean_neg = np.zeros(dim);  mean_neg[0] = -offset

        X_pos = rng.normal(loc=mean_pos, scale=noise, size=(n_samples, dim))
        X_neg = rng.normal(loc=mean_neg, scale=noise, size=(n_samples, dim))

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate([np.ones(n_samples), -np.ones(n_samples)])

        # Shuffle together
        idx = rng.permutation(2 * n_samples)
        return X[idx], y[idx]


# ─────────────────────────────────────────────
# Perceptron
# ─────────────────────────────────────────────
class Perceptron:
    """Single-sample perceptron with step activation."""

    def __init__(self, dim: int, lr: float = 0.1):
        self.dim = dim
        self.learning_rate = lr
        self.weights = np.zeros(dim)
        self.bias = 0.0
        self.step_count = 0

    # -- public API ---------------------------------------------------------
    def predict_one(self, x: np.ndarray) -> int:
        """Predict ±1 for a single sample."""
        return int(np.sign(self.weights @ x + self.bias))

    def predict_all(self, X: np.ndarray) -> np.ndarray:
        """Predict ±1 for every row of X."""
        decisions = X @ self.weights + self.bias
        preds = np.sign(decisions)
        preds[preds == 0] = 1  # convention: 0 → +1
        return preds.astype(int)

    def update(self, x: np.ndarray, y_true: int) -> None:
        """One-step perceptron update (called when point is misclassified).

        w ← w + lr · y · x
        b ← b + lr · y
        """
        self.weights += self.learning_rate * y_true * x
        self.bias += self.learning_rate * y_true
        self.step_count += 1

    def reset(self) -> None:
        self.weights = np.zeros(self.dim)
        self.bias = 0.0
        self.step_count = 0

    def error_count(self, X: np.ndarray, y: np.ndarray) -> int:
        preds = self.predict_all(X)
        return int(np.sum(preds != y))


# ─────────────────────────────────────────────
# Visualizer
# ─────────────────────────────────────────────
class PerceptronVisualizer:
    """Matplotlib interactive window with training controls.

    Two-phase step workflow:
        Phase 1 (Preview): highlight the candidate point with a yellow ring.
        Phase 2 (Commit):  if misclassified, update weights (green flash);
                           then advance to the next point.
    """

    # --- layout fractions (left, bottom, width, height) -------------------
    _MAIN = [0.08, 0.30, 0.84, 0.65]
    _BUT_ROW = [0.08, 0.23, 0.84, 0.045]
    _SLR_LR  = [0.10, 0.15, 0.28, 0.04]
    _SLR_SEP = [0.10, 0.08, 0.28, 0.04]
    _SLR_N   = [0.50, 0.01, 0.40, 0.04]

    # --- constructor -------------------------------------------------------
    def __init__(self, X: np.ndarray, y: np.ndarray, perc: Perceptron):
        self.X = X
        self.y = y
        self.perc = perc
        self.dim = perc.dim

        # training state
        self._idx = 0          # current sample index (0..n-1)
        self._epoch = 0
        self._phase = "idle"   # "idle" | "preview"
        self._misclassified = False

        # GUI
        self.fig = plt.figure(figsize=(10, 7))
        self.fig.canvas.manager.set_window_title("Perceptron Training Visualizer")
        self._widgets: list = []
        self._build_ui()

    # -----------------------------------------------------------------------
    # Widget lifecycle
    # -----------------------------------------------------------------------
    def _disconnect_widgets(self):
        for w in self._widgets:
            try:
                w.disconnect_events()
            except Exception:
                pass
        self._widgets.clear()

    # --- UI construction ---------------------------------------------------
    def _build_ui(self, new_figure: bool = False):
        """Build or rebuild the full UI."""
        mpl.rcParams.update({"font.size": 14, "axes.titlesize": 17})

        if new_figure:
            self._disconnect_widgets()
            for ax in self.fig.axes:
                ax.remove()

        # ---- main plot axes ----
        if self.dim == 3:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            self.ax = self.fig.add_axes(self._MAIN, projection="3d")
        else:
            self.ax = self.fig.add_axes(self._MAIN)
        self._make_axes(self.ax, self.X, self.y)

        # ---- buttons (Step, Reset, Regenerate, Dim) ----
        bw = 0.085
        blefts = [0.08, 0.18, 0.28, 0.38]
        btn_configs = [
            ("Step",       self._on_step,   "#7bf1a8"),
            ("Reset",      self._on_reset,  "#fcb44b"),
            ("Regenerate", self._on_regen,  "#7bf1a8"),
            (f"{self.dim}D", self._on_dim_toggle, "#fcb44b"),
        ]
        for i, (label, cb, hover) in enumerate(btn_configs):
            ax_btn = self.fig.add_axes([blefts[i], self._BUT_ROW[1],
                                        bw, self._BUT_ROW[3]])
            btn = Button(ax_btn, label, hovercolor=hover)
            btn.on_clicked(cb)
            self._widgets.append(btn)
            if label.endswith("D"):
                self.btn_dim = btn
            elif label == "Step":
                self.btn_step = btn

        # ---- sliders ----
        ax_lr = self.fig.add_axes(self._SLR_LR)
        self.sl_lr = Slider(ax_lr, "LR", 0.01, 2.0,
                            valinit=self.perc.learning_rate, valstep=0.01,
                            valfmt="%.2f")
        self.sl_lr.on_changed(self._on_lr_change)
        self._widgets.append(self.sl_lr)

        ax_sep = self.fig.add_axes(self._SLR_SEP)
        self.sl_sep = Slider(ax_sep, "Separation", 0.0, 4.0,
                             valinit=0.8, valstep=0.1, valfmt="%.1f")
        self.sl_sep.on_changed(self._on_sep_change)
        self._widgets.append(self.sl_sep)

        ax_n = self.fig.add_axes(self._SLR_N)
        current_n = len(self.X) // 2
        self.sl_n = Slider(ax_n, "Samples/class", 20, 200,
                           valinit=current_n, valstep=10, valfmt="%d")
        self.sl_n.on_changed(self._on_n_change)
        self._widgets.append(self.sl_n)

        self._draw_plot()

    # --- axes population ---------------------------------------------------
    def _make_axes(self, ax, X, y):
        if self.dim == 2:
            self._make_axes_2d(ax, X, y)
        else:
            self._make_axes_3d(ax, X, y)

    def _make_axes_2d(self, ax, X, y):
        ax.set_aspect("equal")
        x0lo, x0hi = X[:, 0].min() - 1.5, X[:, 0].max() + 1.5
        x1lo, x1hi = X[:, 1].min() - 1.5, X[:, 1].max() + 1.5
        ax.set_xlim(x0lo, x0hi)
        ax.set_ylim(x1lo, x1hi)
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")

        pos = y == 1
        neg = y == -1
        self.ln_pos, = ax.plot(X[pos, 0], X[pos, 1], "bo",
                               markersize=10, alpha=0.7, label="+1")
        self.ln_neg, = ax.plot(X[neg, 0], X[neg, 1], "rx",
                               markersize=11, alpha=0.7, label="-1")
        ax.legend(loc="lower right", fontsize=15)

        self._boundary_lines: list = []
        # normal-vector arrow
        self._normal_arrow = ax.quiver(0, 0, 0, 0, color="purple",
                                       width=0.006, scale_units="xy",
                                       angles="xy", scale=1, alpha=0)
        # yellow preview ring
        self._hl_preview, = ax.plot([], [], "o", markerfacecolor="none",
                                     markeredgecolor="black", markersize=20,
                                     markeredgewidth=3, alpha=0,
                                     label="_nolegend_")
        # green update flash
        self._hl_update, = ax.plot([], [], "go", markersize=16, alpha=0,
                                   label="_nolegend_")

        self._txt_status = ax.text(0.02, 0.98, "", transform=ax.transAxes,
                                   fontsize=17, verticalalignment="top",
                                   bbox=dict(boxstyle="round", facecolor="wheat",
                                             alpha=0.5))

    def _make_axes_3d(self, ax, X, y):
        ax.set_xlabel(r"$x_0$")
        ax.set_ylabel(r"$x_1$")
        ax.set_zlabel(r"$x_2$")
        ax.view_init(elev=25, azim=130)

        # axis limits proportional to data ranges → normal arrow looks correct
        margin = 1.5
        ax.set_xlim(X[:, 0].min() - margin, X[:, 0].max() + margin)
        ax.set_ylim(X[:, 1].min() - margin, X[:, 1].max() + margin)
        ax.set_zlim(X[:, 2].min() - margin, X[:, 2].max() + margin)

        pos = y == 1
        neg = y == -1
        self.ln_pos = ax.scatter(X[pos, 0], X[pos, 1], X[pos, 2],
                                 c="blue", marker="o", s=80, alpha=0.7,
                                 label="+1")
        self.ln_neg = ax.scatter(X[neg, 0], X[neg, 1], X[neg, 2],
                                 c="red", marker="x", s=100, alpha=0.7,
                                 label="-1")
        ax.legend(loc="lower right", fontsize=15, ncol=2)

        # hyperplane mesh (tight around data)
        x0r = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 16)
        x1r = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 16)
        self._mesh_X0, self._mesh_X1 = np.meshgrid(x0r, x1r)
        self._boundary_surf = None

        # 3-D normal-vector arrow (recreated each frame in _draw_normal_3d)
        self._normal_arrow = None

        # highlight artists (3D scatter)
        self._hl_preview = ax.scatter([], [], [], c="none", s=350,
                                       edgecolors="black", linewidths=3,
                                       alpha=0)
        self._hl_update = ax.scatter([], [], [], c="green", s=250, alpha=0)

        self._txt_status = ax.text2D(0.02, 0.98, "", transform=ax.transAxes,
                                     fontsize=17, verticalalignment="top",
                                     bbox=dict(boxstyle="round", facecolor="wheat",
                                               alpha=0.5))

    # --- full redraw -------------------------------------------------------
    def _draw_plot(self):
        """Redraw all dynamic elements and issue one draw_idle."""
        # Guard: figure may have been closed by the user (self.fig -> None)
        # or the canvas may have been destroyed during a rebuild.
        if self.fig is None or not self.fig.canvas or not self.fig.get_axes():
            return

        errors = self.perc.error_count(self.X, self.y)

        phase_msg = ""
        if self._phase == "preview":
            if self._misclassified:
                phase_msg = "  ← misclassified (will update)"
            else:
                phase_msg = "  ← correct (will skip)"

        status = (f"Epoch: {self._epoch}  |  Sample: {self._idx}/{len(self.X)}  "
                  f"|  Updates: {self.perc.step_count}  |  "
                  f"Errors: {errors}/{len(self.y)}  |  "
                  f"LR: {self.perc.learning_rate:.2f}"
                  f"{phase_msg}")
        self._txt_status.set_text(status)

        # decision boundary + normal vector
        if self.dim == 2:
            self._draw_boundary_2d()
            self._draw_normal_2d()
        else:
            self._draw_boundary_3d()
            self._draw_normal_3d()

        # highlights
        self._draw_highlights()

        # weights in title
        w = self.perc.weights
        if np.linalg.norm(w) < 1e-8:
            self.ax.set_title("w = 0 (no boundary yet)")
        else:
            labels = "  ".join(f"$w_{i}={w[i]:.2f}$" for i in range(self.dim))
            self.ax.set_title(f"{labels}  $b={self.perc.bias:.2f}$")

        self.fig.canvas.draw_idle()

    # --- boundary drawing --------------------------------------------------
    def _draw_boundary_2d(self):
        for ln in self._boundary_lines:
            ln.remove()
        self._boundary_lines.clear()

        w = self.perc.weights
        ax = self.ax
        xlo, xhi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()

        if abs(w[1]) > 1e-8:
            xs = np.linspace(xlo, xhi, 100)
            ys = -(w[0] * xs + self.perc.bias) / w[1]
            ln, = ax.plot(xs, ys, "k--", linewidth=2.5)
            self._boundary_lines.append(ln)
        elif abs(w[0]) > 1e-8:
            xs = np.full(2, -self.perc.bias / w[0])
            ys = [ylo, yhi]
            ln, = ax.plot(xs, ys, "k--", linewidth=2.5)
            self._boundary_lines.append(ln)

    def _draw_normal_2d(self):
        """Draw the weight vector as an arrow normal to the decision boundary."""
        w = self.perc.weights
        wnorm = np.linalg.norm(w)
        if wnorm < 1e-8:
            self._normal_arrow.set_alpha(0)
            return

        # anchor at the boundary point closest to origin
        if abs(w[1]) > 1e-8:
            anchor_x0 = 0.0
            anchor_x1 = -self.perc.bias / w[1]
        elif abs(w[0]) > 1e-8:
            anchor_x0 = -self.perc.bias / w[0]
            anchor_x1 = 0.0
        else:
            anchor_x0 = anchor_x1 = 0.0

        # scale to ~1/3 of the plot width
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        plot_w = max(xlim[1] - xlim[0], ylim[1] - ylim[0])
        scale = plot_w / (3.5 * wnorm)

        self._normal_arrow.set_offsets([anchor_x0, anchor_x1])
        self._normal_arrow.set_UVC(w[0] * scale, w[1] * scale)
        self._normal_arrow.set_alpha(0.8)

    def _draw_boundary_3d(self):
        if self._boundary_surf is not None:
            self._boundary_surf.remove()
            self._boundary_surf = None

        w = self.perc.weights
        b = self.perc.bias

        if abs(w[2]) > 1e-8:
            Z = -(w[0] * self._mesh_X0 + w[1] * self._mesh_X1 + b) / w[2]
            self._boundary_surf = self.ax.plot_surface(
                self._mesh_X0, self._mesh_X1, Z,
                alpha=0.25, color="gray", edgecolor="none")
        elif abs(w[0]) > 1e-8 or abs(w[1]) > 1e-8:
            Xlim = self.ax.get_xlim()
            Ylim = self.ax.get_ylim()
            Zlim = self.ax.get_zlim()

            if abs(w[0]) > abs(w[1]):
                x0c = -b / w[0]
                xs = np.full((2, 2), x0c)
                ys = np.array([[Ylim[0], Ylim[0]], [Ylim[1], Ylim[1]]])
                zs = np.array([[Zlim[0], Zlim[1]], [Zlim[0], Zlim[1]]])
            else:
                x1c = -b / w[1]
                xs = np.array([[Xlim[0], Xlim[1]], [Xlim[0], Xlim[1]]])
                ys = np.full((2, 2), x1c)
                zs = np.array([[Zlim[0], Zlim[1]], [Zlim[0], Zlim[1]]])
            self._boundary_surf = self.ax.plot_surface(xs, ys, zs,
                                                       alpha=0.25, color="gray")

    def _draw_normal_3d(self):
        """Draw the weight vector as an arrow normal to the decision plane.

        3D quiver (Line3DCollection) cannot be updated in-place, so we
        remove and recreate the arrow each frame.
        """
        w = self.perc.weights
        wnorm = np.linalg.norm(w)

        # remove previous arrow
        if self._normal_arrow is not None:
            self._normal_arrow.remove()

        if wnorm < 1e-8:
            self._normal_arrow = None
            return

        b = self.perc.bias

        # anchor: closest point on the plane w·x + b = 0 to the origin
        #   npt = -b * w / ||w||^2  (guaranteed: w·npt + b = 0)
        npt = -b * w / (w @ w)

        # direction: use the weight vector, normalised to unit data length,
        # then scaled to a fraction of the data extent.
        # Using the data extent (not axis limits) avoids distortion from
        # the boundary plane inflating the axis ranges.
        data_extent = np.max(np.ptp(self.X, axis=0))  # max range across dims
        w_unit = w / wnorm
        arrow_len = data_extent / 2.0

        self._normal_arrow = self.ax.quiver(
            npt[0], npt[1], npt[2],
            w_unit[0] * arrow_len, w_unit[1] * arrow_len, w_unit[2] * arrow_len,
            color="purple", arrow_length_ratio=0.15, alpha=0.8)

    # --- highlights --------------------------------------------------------
    def _clear_highlights(self):
        """Hide all highlight artists."""
        self._hl_preview.set_alpha(0)
        self._hl_update.set_alpha(0)

    def _show_preview(self, pt):
        """Show yellow preview ring on *pt*."""
        self._clear_highlights()
        self._hl_preview.set_alpha(1)
        if self.dim == 2:
            self._hl_preview.set_data([pt[0]], [pt[1]])
        else:
            self._hl_preview._offsets3d = ([pt[0]], [pt[1]], [pt[2]])

    def _show_update(self, pt):
        """Show green flash on *pt* (used after a weight update)."""
        self._clear_highlights()
        self._hl_update.set_alpha(1)
        if self.dim == 2:
            self._hl_update.set_data([pt[0]], [pt[1]])
        else:
            self._hl_update._offsets3d = ([pt[0]], [pt[1]], [pt[2]])

    def _draw_highlights(self):
        if self._phase == "idle":
            self._clear_highlights()
            return
        # In preview phase, highlight is already set by _on_step;
        # just redraw.
        if self.fig is not None and self.fig.canvas:
            self.fig.canvas.draw_idle()

    # --- two-phase step logic -----------------------------------------------
    def _on_step(self, _):
        """Button callback: one press advances one phase."""
        if self._phase == "idle":
            # Phase 1: preview the next candidate
            self._phase = "preview"
            idx = self._idx % len(self.X)
            pt = self.X[idx]
            self._misclassified = (self.perc.predict_one(pt) != self.y[idx])
            self._show_preview(pt)
            self._draw_plot()

        elif self._phase == "preview":
            # Phase 2: commit (update if misclassified) then advance
            idx = self._idx % len(self.X)
            pt = self.X[idx]
            if self._misclassified:
                self.perc.update(pt, int(self.y[idx]))
                self._show_update(pt)

            self._idx += 1
            if self._idx % len(self.X) == 0:
                self._epoch += 1

            self._phase = "idle"
            self._draw_plot()

    # --- other callbacks ---------------------------------------------------
    def _on_reset(self, _):
        self.perc.reset()
        self._idx = 0
        self._epoch = 0
        self._phase = "idle"
        self._draw_plot()

    def _on_regen(self, _):
        self.regenerate()

    def _on_dim_toggle(self, _):
        new_dim = 3 if self.dim == 2 else 2
        self.perc.dim = new_dim
        self.perc.weights = np.zeros(new_dim)
        self.perc.bias = 0.0
        self.perc.step_count = 0
        self.dim = new_dim
        self.X, self.y = DatasetGenerator.generate(
            dim=self.dim,
            n_samples=int(self.sl_n.val),
            separation=self.sl_sep.val,
        )
        self._idx = 0
        self._epoch = 0
        self._phase = "idle"
        self.btn_dim.label.set_text(f"{self.dim}D")
        self._build_ui(new_figure=True)

    def _on_lr_change(self, val):
        self.perc.learning_rate = val
        self._draw_plot()

    def _on_sep_change(self, _):
        pass  # value shown by slider itself

    def _on_n_change(self, _):
        pass  # value shown by slider itself

    # --- regenerate --------------------------------------------------------
    def regenerate(self):
        """Regenerate data and rebuild UI in-place."""
        self.X, self.y = DatasetGenerator.generate(
            dim=self.dim,
            n_samples=int(self.sl_n.val),
            separation=self.sl_sep.val,
        )
        self.perc.reset()
        self._idx = 0
        self._epoch = 0
        self._phase = "idle"
        self._build_ui(new_figure=True)

    # --- entry point -------------------------------------------------------
    def run(self):
        plt.show()


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Interactive Perceptron Training Visualizer")
    parser.add_argument("--dim", type=int, default=2, choices=[2, 3],
                        help="Dataset dimensionality (default: 2)")
    parser.add_argument("--samples", type=int, default=80,
                        help="Samples per class (default: 80)")
    parser.add_argument("--separation", type=float, default=0.8,
                        help="Cluster separation factor (default: 0.8)")
    parser.add_argument("--lr", type=float, default=0.5,
                        help="Initial learning rate (default: 0.5)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: None)")
    args = parser.parse_args()

    X, y = DatasetGenerator.generate(dim=args.dim,
                                     n_samples=args.samples,
                                     separation=args.separation,
                                     seed=args.seed)
    perc = Perceptron(dim=args.dim, lr=args.lr)
    vis = PerceptronVisualizer(X, y, perc)

    # sync sliders to CLI defaults
    vis.sl_sep.set_val(args.separation)
    vis.sl_n.set_val(args.samples)
    vis.sl_lr.set_val(args.lr)

    vis.run()


if __name__ == "__main__":
    main()
