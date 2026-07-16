#!/usr/bin/env python3
"""
Code written by Nicholas J. Cooper with help from Claude Code (VSCode frontend only) w/ backend model Qwen3.6-27B-Q6_K (available from here: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/blob/main/Qwen3.6-27B-Q6_K.gguf)

Interactive Self-Attention Computational Flow Visualizer.

Demonstrates how a single attention weight A[i, j] is computed from raw
token embeddings through projection, dot product, and softmax — without
any Transformer library. The student clicks a cell in the attention map,
then steps through the full computational provenance of that value.

Usage:
    python Visualize_Attn.py
"""

import argparse

import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NUM_TOKENS: int = 4       # N — number of input tokens
TOKEN_DIM: int = 3        # D — embedding / projection dimension

# Colors (dataviz reference palette)
C_ROW_HL: str = "#eda100"       # active row highlight (yellow)
C_COL_HL: str = "#b7d3f6"       # active column highlight (light blue)
C_TARGET_HL: str = "#c98500"    # target element highlight (deep amber)
C_RESULT: str = "#1baf7a"       # result cell (aqua)
C_RESULT_FADE: str = "#d4f5e9"  # completed cell (faded aqua)
C_GRID: str = "#c3c2b7"         # grid lines
C_TEXT: str = "#0b0b0b"         # primary ink
C_MUTED: str = "#52514e"        # secondary ink
C_SURFACE: str = "#fcfcfb"      # figure background
C_PLACEHOLDER: str = "#e8e8e6"  # empty placeholder cell
C_SELECTED: str = "#4a3aa7"     # selected cell highlight (violet)

# Cell rendering — default (entry screen, vectors)
CELL_W: float = 0.050           # figure fraction per cell
CELL_H: float = 0.050           # figure fraction per cell
CELL_GAP: float = 0.004         # gap between cells
CELL_TEXT_SIZE: int = 13      # font size for cell values
LABEL_SIZE: int = 11            # font size for row/col labels

# Smaller cells for weight matrices in projection view
W_CELL_W: float = 0.050
W_CELL_H: float = 0.05
W_GAP: float = 0.004
W_TEXT_SIZE: int = 13
W_LABEL_SIZE: int = 11

# Attention map cells (larger, more readable)
A_CELL_W: float = 0.060
A_CELL_H: float = 0.060
A_GAP: float = 0.005
A_TEXT_SIZE: int = 14

# Phase names for display
PHASE_NAMES: list[str] = [
    "Project Query",
    "Project Key",
    "Dot Product",
    "Softmax",
]

PHASE_FORMULAS: list[str] = [
    r"$q_i = x_i \cdot W_Q$",
    r"$k_j = x_j \cdot W_K$",
    r"$S_{i,j} = q_i \cdot k_j$",
    r"$A_{i,j} = \frac{e^{S_{i,j}}}{\sum_k e^{S_{i,k}}}$",
]


# ─────────────────────────────────────────────
# MatrixPatch — renders a matrix as a grid of patches + text
# ─────────────────────────────────────────────
class MatrixPatch:
    """Render a 2-D array as a grid of coloured rectangles and text labels
    on a matplotlib figure.  Supports per-cell highlight overrides and
    optional clickability."""

    def __init__(
        self,
        fig: plt.Figure,
        data: np.ndarray,
        pos: tuple[float, float],
        cell_w: float = CELL_W,
        cell_h: float = CELL_H,
        gap: float = CELL_GAP,
        row_labels: list[str] | None = None,
        col_labels: list[str] | None = None,
        title: str = "",
        clickable: bool = False,
        text_size: int = CELL_TEXT_SIZE,
        label_size: int = LABEL_SIZE,
    ) -> None:
        self.fig = fig
        self.data = data
        self.rows, self.cols = data.shape
        self.pos = pos
        self.cell_w = cell_w
        self.cell_h = cell_h
        self.gap = gap
        self.row_labels = row_labels
        self.col_labels = col_labels
        self.title = title
        self.clickable = clickable
        self.text_size = text_size
        self.label_size = label_size

        # Per-cell highlight colour overrides
        self._hl: dict[tuple[int, int], str] = {}
        # Per-cell border overrides: (edgecolor, linewidth)
        self._border: dict[tuple[int, int], tuple[str, float]] = {}
        # Per-cell text overrides
        self._text_ov: dict[tuple[int, int], str] = {}

        # Artist collections (so we can redraw / remove)
        self._rects: list[Rectangle] = []
        self._texts: list[mpl.text.Text] = []
        self._label_texts: list[mpl.text.Text] = []
        self._title_text: mpl.text.Text | None = None

    # ------------------------------------------------------------------
    # Highlight API
    # ------------------------------------------------------------------
    def set_cell_color(self, row: int, col: int, color: str) -> None:
        self._hl[(row, col)] = color

    def set_cell_border(self, row: int, col: int,
                        color: str = "#000000", width: float = 3.0) -> None:
        self._border[(row, col)] = (color, width)

    def clear_highlights(self) -> None:
        self._hl.clear()
        self._border.clear()
        self._text_ov.clear()

    def clear_colors(self) -> None:
        """Clear color and border overrides, preserve text overrides."""
        self._hl.clear()
        self._border.clear()

    def set_cell_text(self, row: int, col: int, text: str) -> None:
        self._text_ov[(row, col)] = text

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _cell_pos(self, row: int, col: int) -> tuple[float, float]:
        """Return (x, y) figure-fraction for bottom-left of cell (row, col)."""
        x = self.pos[0] + col * (self.cell_w + self.gap)
        y = self.pos[1] + (self.rows - 1 - row) * (self.cell_h + self.gap)
        return (x, y)

    @property
    def width(self) -> float:
        """Total rendered width of the grid."""
        return self.cols * self.cell_w + (self.cols - 1) * self.gap

    @property
    def height(self) -> float:
        """Total rendered height of the grid."""
        return self.rows * self.cell_h + (self.rows - 1) * self.gap

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return (x, y, w, h) bounding box of the entire grid."""
        return (self.pos[0], self.pos[1], self.width, self.height)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render(self) -> None:
        """Create or update all patch and text artists."""
        self._remove_old()
        self._draw_cells()
        self._draw_labels()
        self._draw_title()

    def _remove_old(self) -> None:
        for artist in self._rects + self._texts + self._label_texts:
            artist.remove()
        if self._title_text is not None:
            self._title_text.remove()
        self._rects.clear()
        self._texts.clear()
        self._label_texts.clear()
        self._title_text = None

    def _draw_cells(self) -> None:
        for r in range(self.rows):
            for c in range(self.cols):
                cx, cy = self._cell_pos(r, c)
                facecolor = self._hl.get((r, c), C_SURFACE)
                if (r, c) in self._border:
                    edgecolor, linewidth = self._border[(r, c)]
                else:
                    edgecolor, linewidth = C_GRID, 0.8

                rect = Rectangle(
                    (cx, cy), self.cell_w, self.cell_h,
                    facecolor=facecolor, edgecolor=edgecolor,
                    linewidth=linewidth,
                    transform=self.fig.transFigure,
                    picker=True if self.clickable else False,
                )
                self.fig.add_artist(rect)
                self._rects.append(rect)

                # Text centre of cell
                tx = cx + self.cell_w / 2
                ty = cy + self.cell_h / 2
                text = self._text_ov.get((r, c), self._format_val(self.data[r, c]))
                color = self._contrast_text(facecolor)
                t = self.fig.text(
                    tx, ty, text,
                    fontsize=self.text_size, fontfamily="monospace",
                    ha="center", va="center", color=color,
                    transform=self.fig.transFigure,
                )
                self._texts.append(t)

    def _draw_labels(self) -> None:
        if self.col_labels is not None:
            for c, label in enumerate(self.col_labels):
                cx, cy = self._cell_pos(0, c)
                tx = cx + self.cell_w / 2
                ty = cy + self.cell_h + self.cell_h * 0.3
                t = self.fig.text(
                    tx, ty, label,
                    fontsize=self.label_size, ha="center", va="bottom",
                    color=C_MUTED, transform=self.fig.transFigure,
                )
                self._label_texts.append(t)

        if self.row_labels is not None:
            for r, label in enumerate(self.row_labels):
                cx, cy = self._cell_pos(r, 0)
                tx = cx - self.cell_w * 0.15
                ty = cy + self.cell_h / 2
                t = self.fig.text(
                    tx, ty, label,
                    fontsize=self.label_size, ha="right", va="center",
                    color=C_MUTED, transform=self.fig.transFigure,
                )
                self._label_texts.append(t)

    def _draw_title(self) -> None:
        if not self.title:
            return
        tx = self.pos[0] + self.width / 2
        ty = self.pos[1] + self.height + self.cell_h * 0.6
        self._title_text = self.fig.text(
            tx, ty, self.title,
            fontsize=self.label_size + 1, ha="center", va="bottom",
            fontweight="bold", color=C_TEXT,
            transform=self.fig.transFigure,
        )

    @staticmethod
    def _format_val(v: float) -> str:
        return f"{v:+.2f}"

    @staticmethod
    def _contrast_text(bg: str) -> str:
        """Return white or black text colour for maximum contrast on *bg* hex."""
        bg = bg.lstrip("#")
        if len(bg) != 6:
            return C_TEXT
        r, g, b = int(bg[0:2], 16), int(bg[2:4], 16), int(bg[4:6], 16)
        lum = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
        return "#ffffff" if lum < 0.45 else C_TEXT


# ─────────────────────────────────────────────
# AttnVisualizer
# ─────────────────────────────────────────────
class AttnVisualizer:
    """Interactive matplotlib window that traces the computation of a
    single attention weight A[i, j] from input tokens through projection,
    dot product, and softmax."""

    # Layout positions (figure fractions)
    _TITLE_Y: float = 0.95
    _STATUS_Y: float = 0.11
    _BUT_STEP: list[float] = [0.04, 0.02, 0.12, 0.045]
    _BUT_RESET: list[float] = [0.18, 0.02, 0.12, 0.045]

    # Fixed anchor positions for all matrices (used across all phases)
    WQK_X = 0.25
    QK_X = 0.45
    Q_Y = 0.45
    K_Y = 0.725
    _X_POS: tuple[float, float] = (0.1, 0.175)
    _WQ_POS: tuple[float, float] = (WQK_X, Q_Y + 0.025)
    _Q_POS: tuple[float, float] = (QK_X, Q_Y)
    _WK_POS: tuple[float, float] = (WQK_X, K_Y)
    _KT_POS: tuple[float, float] = (QK_X + 0.2, K_Y)
    _ATTN_POS: tuple[float, float] = (0.6275, 0.15)

    # Entry-screen positions (for the initial select view)
    _ENTRY_X_POS: tuple[float, float] = _X_POS
    _ENTRY_ATTN_POS: tuple[float, float] = _ATTN_POS

    def __init__(self, seed: int = 42) -> None:
        # --- Data generation ---
        rng = np.random.default_rng(seed)
        self.X: np.ndarray = np.round(rng.uniform(-1, 1, (NUM_TOKENS, TOKEN_DIM)), 2)
        self.W_Q: np.ndarray = np.round(rng.uniform(-1, 1, (TOKEN_DIM, TOKEN_DIM)), 2)
        self.W_K: np.ndarray = np.round(rng.uniform(-1, 1, (TOKEN_DIM, TOKEN_DIM)), 2)
        self.W_V: np.ndarray = np.round(rng.uniform(-1, 1, (TOKEN_DIM, TOKEN_DIM)), 2)

        # --- Pre-computed intermediates ---
        self.Q: np.ndarray = np.round(self.X @ self.W_Q, 2)
        self.K: np.ndarray = np.round(self.X @ self.W_K, 2)
        self.V: np.ndarray = np.round(self.X @ self.W_V, 2)
        self.Scores: np.ndarray = np.round(self.Q @ self.K.T, 2)
        self.Attn: np.ndarray = np.round(self._softmax_rows(self.Scores), 2)

        # --- State ---
        self._mode: str = "select"
        self._target_i: int = 0
        self._target_j: int = 0
        self._phase: int = 0
        self._sub_step: int = 0

        # --- GUI ---
        self.fig: plt.Figure | None = None
        # Persistent matrix patches — created once, reused across phases
        self._mat_x: MatrixPatch | None = None
        self._mat_attn: MatrixPatch | None = None
        # Persistent projection patches (W_Q, W_K, Q, K^T, arrows)
        self._mat_wq: MatrixPatch | None = None
        self._mat_q: MatrixPatch | None = None
        self._mat_wk: MatrixPatch | None = None
        self._mat_kt: MatrixPatch | None = None
        self._arrow_q: mpl.text.Text | None = None
        self._arrow_k: mpl.text.Text | None = None
        # Phase-specific dynamic patches (cleared each render)
        self._dynamic_patches: list[MatrixPatch] = []
        self._dynamic_artists: list[mpl.artist.Artist] = []
        # Persistent X row highlights (carried across phases)
        self._x_row_i_hl: str = ""
        self._x_row_j_hl: str = ""
        self._txt_title: mpl.text.Text | None = None
        self._txt_status: mpl.text.Text | None = None
        self._btn_step: Button | None = None
        self._btn_reset: Button | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Softmax helper
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax_rows(mat: np.ndarray) -> np.ndarray:
        """Row-wise softmax."""
        e = np.exp(mat)
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _format_value(v: float) -> str:
        return f"{v:+.2f}"

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _clear_dynamic(self) -> None:
        """Remove all phase-specific dynamic artists (not persistent patches)."""
        for p in self._dynamic_patches:
            p._remove_old()
        self._dynamic_patches.clear()
        for a in self._dynamic_artists:
            a.remove()
        self._dynamic_artists.clear()

    def _hide_projection(self) -> None:
        """Remove projection patch artists so they don't render on entry screen."""
        if self._mat_wq:
            self._mat_wq._remove_old()
        if self._mat_q:
            self._mat_q._remove_old()
        if self._mat_wk:
            self._mat_wk._remove_old()
        if self._mat_kt:
            self._mat_kt._remove_old()
        if self._arrow_q:
            self._arrow_q.set_visible(False)
        if self._arrow_k:
            self._arrow_k.set_visible(False)

    def _show_projection(self) -> None:
        """Show projection patches and arrows (used during trace)."""
        if self._arrow_q:
            self._arrow_q.set_visible(True)
        if self._arrow_k:
            self._arrow_k.set_visible(True)

    def _build_ui(self) -> None:
        mpl.rcParams.update({"font.size": 13})

        self.fig = plt.figure(figsize=(16, 12))
        self.fig.patch.set_facecolor(C_SURFACE)
        self.fig.canvas.manager.set_window_title("Self-Attention Visualizer")

        # Title area
        self._txt_title = self.fig.text(
            0.5, self._TITLE_Y, "",
            fontsize=16, ha="center", va="center",
            fontweight="bold", color=C_TEXT,
            transform=self.fig.transFigure,
        )

        # Status area
        self._txt_status = self.fig.text(
            0.5, self._STATUS_Y, "",
            fontsize=13, ha="center", va="center",
            color=C_MUTED,
            transform=self.fig.transFigure,
        )

        # Buttons
        ax_step = self.fig.add_axes(self._BUT_STEP)
        self._btn_step = Button(ax_step, "Step", hovercolor="#7bf1a8")
        self._btn_step.on_clicked(self._on_step)

        ax_reset = self.fig.add_axes(self._BUT_RESET)
        self._btn_reset = Button(ax_reset, "Reset", hovercolor="#fcb44b")
        self._btn_reset.on_clicked(self._on_reset)

        # Persistent X matrix — created once, reused across phases
        self._mat_x = MatrixPatch(
            self.fig, self.X, pos=self._X_POS,
            cell_w=W_CELL_W, cell_h=W_CELL_H, gap=W_GAP,
            row_labels=[f"t{k}" for k in range(NUM_TOKENS)],
            col_labels=[f"d{k}" for k in range(TOKEN_DIM)],
            title="Input X",
            text_size=W_TEXT_SIZE, label_size=W_LABEL_SIZE,
        )

        # Persistent attention map — created once, reused across phases
        self._mat_attn = MatrixPatch(
            self.fig, self.Attn, pos=self._ATTN_POS,
            cell_w=A_CELL_W, cell_h=A_CELL_H, gap=A_GAP,
            row_labels=[f"t{k}" for k in range(NUM_TOKENS)],
            col_labels=[f"t{k}" for k in range(NUM_TOKENS)],
            title="Attention Map A",
            clickable=True,
            text_size=A_TEXT_SIZE,
            label_size=LABEL_SIZE,
        )
        self._colour_heatmap(self._mat_attn, self.Attn)

        # --- Persistent projection patches (created once, hidden until trace) ---
        kt_zeros: np.ndarray = np.zeros_like(self.K.T)

        self._mat_wq = MatrixPatch(
            self.fig, self.W_Q, pos=self._WQ_POS,
            cell_w=W_CELL_W, cell_h=W_CELL_H, gap=W_GAP,
            row_labels=[f"d{k}" for k in range(TOKEN_DIM)],
            col_labels=[f"d'{k}" for k in range(TOKEN_DIM)],
            title="W_Q",
            text_size=W_TEXT_SIZE, label_size=W_LABEL_SIZE,
        )

        # Q starts empty (placeholders), populated at phase transition
        q_zeros = np.zeros_like(self.Q)
        self._mat_q = MatrixPatch(
            self.fig, q_zeros, pos=self._Q_POS,
            cell_w=W_CELL_W, cell_h=W_CELL_H, gap=W_GAP,
            row_labels=[f"t{k}" for k in range(NUM_TOKENS)],
            col_labels=[f"d'{k}" for k in range(TOKEN_DIM)],
            title="Q = X · W_Q",
            text_size=W_TEXT_SIZE, label_size=W_LABEL_SIZE,
        )
        for r in range(NUM_TOKENS):
            for c in range(TOKEN_DIM):
                self._mat_q.set_cell_color(r, c, C_PLACEHOLDER)
                self._mat_q.set_cell_text(r, c, "·")

        self._mat_wk = MatrixPatch(
            self.fig, self.W_K, pos=self._WK_POS,
            cell_w=W_CELL_W, cell_h=W_CELL_H, gap=W_GAP,
            row_labels=[f"d{k}" for k in range(TOKEN_DIM)],
            col_labels=[f"d'{k}" for k in range(TOKEN_DIM)],
            title="W_K",
            text_size=W_TEXT_SIZE, label_size=W_LABEL_SIZE,
        )

        # K^T starts empty (placeholders), populated at phase transition
        self._mat_kt = MatrixPatch(
            self.fig, kt_zeros, pos=self._KT_POS,
            cell_w=W_CELL_W, cell_h=W_CELL_H, gap=W_GAP,
            row_labels=[f"d'{k}" for k in range(TOKEN_DIM)],
            col_labels=[f"t{k}" for k in range(NUM_TOKENS)],
            title="K^T = (X · W_K)^T",
            text_size=W_TEXT_SIZE, label_size=W_LABEL_SIZE,
        )
        for r in range(TOKEN_DIM):
            for c in range(NUM_TOKENS):
                self._mat_kt.set_cell_color(r, c, C_PLACEHOLDER)
                self._mat_kt.set_cell_text(r, c, "·")

        # Arrows (persistent across trace phases)
        self._arrow_q = self.fig.text(
            self._WQ_POS[0] + self._mat_wq.width + 0.008,
            self._WQ_POS[1] + self._mat_wq.height / 2, "→",
            fontsize=32, ha="center", va="center", color=C_MUTED,
            transform=self.fig.transFigure,
        )
        self._arrow_k = self.fig.text(
            self._WK_POS[0] + self._mat_wk.width + 0.008,
            self._WK_POS[1] + self._mat_wk.height / 2, "→",
            fontsize=32, ha="center", va="center", color=C_MUTED,
            transform=self.fig.transFigure,
        )

        # Hide projection elements on entry screen (no data yet)
        self._hide_projection()

        # Connect pick event for clickable attention map cells
        self.fig.canvas.mpl_connect("pick_event", self._on_cell_pick)

        # Draw entry screen
        self._render_select()

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Entry screen (selection mode)
    # ------------------------------------------------------------------
    def _render_select(self) -> None:
        """Position X and A for the selection view, then render them."""
        self._clear_dynamic()
        self._hide_projection()

        # Reset persistent X row highlight tracking
        self._x_row_i_hl = ""
        self._x_row_j_hl = ""

        self._txt_title.set_text(
            "Self-Attention Computational Flow  —  Click a cell in the Attention Map"
        )
        self._txt_status.set_text(
            f"Input tokens X: ({NUM_TOKENS}, {TOKEN_DIM})  |  "
            f"Attention map A: ({NUM_TOKENS}, {NUM_TOKENS})  —  "
            f"Pick any A[i, j] to trace how it was computed"
        )

        # Move X and A to entry-screen positions
        self._mat_x.pos = self._ENTRY_X_POS
        self._mat_attn.pos = self._ENTRY_ATTN_POS
        self._mat_x.title = "Input Tokens X"
        self._mat_x.clear_highlights()
        self._mat_attn.clear_highlights()
        self._colour_heatmap(self._mat_attn, self.Attn)

        # Render X and A at entry positions
        self._mat_x.render()
        self._mat_attn.render()

        self.fig.canvas.draw_idle()

    @staticmethod
    def _colour_heatmap(patch: MatrixPatch, data: np.ndarray) -> None:
        """Colour cells of a MatrixPatch using the sequential blue ramp."""
        lo = data.min()
        hi = data.max()
        rng = hi - lo
        if rng < 1e-6:
            rng = 1.0
        ramp = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef",
                "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
                "#256abf", "#1c5cab", "#184f95", "#104281"]
        for r in range(patch.rows):
            for c in range(patch.cols):
                t = (data[r, c] - lo) / rng
                idx = int(t * (len(ramp) - 1))
                patch.set_cell_color(r, c, ramp[idx])

    # ------------------------------------------------------------------
    # (Removed — attention map is now a persistent patch)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Pick event → start trace
    # ------------------------------------------------------------------
    def _on_cell_pick(self, event: object) -> None:
        """Handle click on an attention map cell."""
        if self._mode != "select" or self._mat_attn is None:
            return

        artist = event.artist
        rect_idx = None
        for idx, rect in enumerate(self._mat_attn._rects):
            if rect is artist:
                rect_idx = idx
                break
        if rect_idx is None:
            return

        row = rect_idx // self._mat_attn.cols
        col = rect_idx % self._mat_attn.cols

        self._mode = "tracing"
        self._target_i = row
        self._target_j = col
        self._phase = 0
        self._sub_step = 0

        # Reset Q and K^T to empty (placeholders) for the new trace
        self._mat_q.clear_highlights()
        self._mat_q.data = np.zeros_like(self.Q)
        for r in range(NUM_TOKENS):
            for c in range(TOKEN_DIM):
                self._mat_q.set_cell_color(r, c, C_PLACEHOLDER)
                self._mat_q.set_cell_text(r, c, "·")

        self._mat_kt.clear_highlights()
        self._mat_kt.data = np.zeros_like(self.K.T)
        for r in range(TOKEN_DIM):
            for c in range(NUM_TOKENS):
                self._mat_kt.set_cell_color(r, c, C_PLACEHOLDER)
                self._mat_kt.set_cell_text(r, c, "·")

        self._btn_step.label.set_text("Step")
        self._render_trace()

    # ------------------------------------------------------------------
    # Trace rendering
    # ------------------------------------------------------------------
    def _render_trace(self) -> None:
        """Render the current state of the trace."""
        if self.fig is None:
            return

        self._clear_dynamic()

        i, j = self._target_i, self._target_j
        phase = self._phase
        step = self._sub_step

        if phase in (0, 1):
            self._render_projection(i, j, phase, step)
        elif phase == 2:
            self._render_dot_product(i, j, step)
        elif phase == 3:
            self._render_softmax(i, j, step)

        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Phases 0 & 1: Unified Projection View
    # Uses persistent X, W_Q, Q, W_K, K^T, arrows, and A patches.
    # ------------------------------------------------------------------
    def _render_projection(self, i: int, j: int,
                           phase: int, step: int) -> None:
        """Render projection phase using persistent patches."""
        x_i = self.X[i]
        x_j = self.X[j]
        q_row = self.Q[i]
        k_row = self.K[j]
        kt = self.K.T

        active_label = PHASE_NAMES[phase]
        phase_num = phase + 1

        self._txt_title.set_text(
            f"Step {phase_num}/4: {active_label}  |  "
            f"Computing A[{i}, {j}]"
        )

        # --- Show persistent projection elements ---
        self._show_projection()

        # --- Phase transition: populate Q (phase 0→1) or K^T (already done) ---
        if phase == 42:
            # Q is still empty (placeholders) — will populate as we step
            # K^T is still empty
            pass
        else:
            # Phase 1: Populate Q with real data
            self._mat_q.clear_highlights()
            self._mat_q.data = self.Q
            self._mat_q.title = "Q = X · W_Q"

            # Populate K^T with real data
            self._mat_kt.clear_highlights()
            self._mat_kt.data = kt
            self._mat_kt.title = "K^T = (X · W_K)^T"

        # --- Persistent X matrix: position, highlights, render ---
        self._mat_x.pos = self._X_POS
        self._mat_x.clear_highlights()
        for c in range(TOKEN_DIM):
            self._mat_x.set_cell_color(i, c, C_ROW_HL)
            self._mat_x.set_cell_color(j, c, C_COL_HL)

        # --- Clear weight matrix highlights on every render (prevent lingering) ---
        self._mat_wq.clear_highlights()
        self._mat_wk.clear_highlights()

        # --- Phase transition: swap in real data at phase boundary ---
        if phase == 0:
            # Q stays empty (placeholders), K^T stays empty
            pass
        else:
            # Phase 1: Populate Q with real data (mimics K^T behavior)
            self._mat_q.clear_highlights()
            self._mat_q.data = self.Q

            # Populate K^T with real data
            self._mat_kt.clear_highlights()
            self._mat_kt.data = kt

        # --- Highlights on Q/K^T ---
        if phase == 0:
            if step > 0:
                col_k = step - 1
                for r in range(TOKEN_DIM):
                    self._mat_wq.set_cell_color(r, col_k, C_ROW_HL)

                # Q row i: color highlights, non-row cells stay as placeholders
                self._mat_q.clear_colors()
                for r in range(NUM_TOKENS):
                    for c in range(TOKEN_DIM):
                        if r != i:
                            self._mat_q.set_cell_color(r, c, C_PLACEHOLDER)
                for c in range(TOKEN_DIM):
                    if c < step:
                        self._mat_q.set_cell_color(i, c, C_RESULT_FADE)
                    if c == col_k:
                        self._mat_q.set_cell_color(i, c, C_RESULT)

                terms = " + ".join(
                    f"{self._format_value(x_i[k])}×{self._format_value(self.W_Q[k, col_k])}"
                    for k in range(TOKEN_DIM)
                )
                self._txt_status.set_text(
                    f"q_{i}[{col_k}] = {terms} = {self._format_value(q_row[col_k])}  "
                    f"(step {step}/{TOKEN_DIM})"
                )
            else:
                self._mat_q.clear_colors()
                # Set all Q cells to gray placeholder (like K^T)
                for r in range(NUM_TOKENS):
                    for c in range(TOKEN_DIM):
                        self._mat_q.set_cell_color(r, c, C_PLACEHOLDER)
                self._txt_status.set_text(
                    f"Press Step to compute row {i} of Q  (step 0/{TOKEN_DIM})"
                )
        else:
            # Phase 1: Q and K^T now have real data, highlight active elements
            self._mat_q.clear_highlights()
            for c in range(TOKEN_DIM):
                self._mat_q.set_cell_color(i, c, C_RESULT_FADE)

            if step > 0:
                dim_k = step - 1
                for r in range(TOKEN_DIM):
                    self._mat_wk.set_cell_color(r, dim_k, C_COL_HL)
                self._mat_kt.clear_highlights()
                for s in range(step):
                    self._mat_kt.set_cell_color(s, j, C_RESULT_FADE)
                self._mat_kt.set_cell_color(dim_k, j, C_RESULT)

                terms = " + ".join(
                    f"{self._format_value(x_j[k])}×{self._format_value(self.W_K[k, dim_k])}"
                    for k in range(TOKEN_DIM)
                )
                self._txt_status.set_text(
                    f"k_{j}[{dim_k}] = {terms} = {self._format_value(k_row[dim_k])}  "
                    f"(step {step}/{TOKEN_DIM})"
                )
            else:
                self._mat_kt.clear_highlights()
                self._txt_status.set_text(
                    f"Press Step to compute column {j} of K^T  (step 0/{TOKEN_DIM})"
                )

        # --- Persistent attention map: position, border, render ---
        self._mat_attn.pos = self._ATTN_POS
        self._mat_attn.clear_highlights()
        self._colour_heatmap(self._mat_attn, self.Attn)
        self._mat_attn.set_cell_border(i, j)

        # Render persistent patches
        self._mat_x.render()
        self._mat_attn.render()
        self._mat_wq.render()
        self._mat_q.render()
        self._mat_wk.render()
        self._mat_kt.render()

    # ------------------------------------------------------------------
    # Phase 2: Dot Product  S[i,j] = Q[i,:] · K^T[:,j]
    # Uses persistent X, W_Q, Q, W_K, K^T, arrows, and A patches.
    # Shows the full pre-softmax score row S[i, :] with cell j updating.
    # ------------------------------------------------------------------
    def _render_dot_product(self, i: int, j: int, step: int) -> None:
        q_row = self.Q[i]
        k_row = self.K[j]
        score = self.Scores[i, j]
        s_row = self.Scores[i]  # full pre-softmax row

        self._txt_title.set_text(
            f"Step 3/4: {PHASE_NAMES[2]}  |  {PHASE_FORMULAS[2]}  "
            f"|  Computing A[{i}, {j}]"
        )

        # --- Show persistent projection elements (arrows + W_Q + W_K) ---
        self._show_projection()

        # Clear stale column highlights from projection phases
        self._mat_wq.clear_highlights()
        self._mat_wk.clear_highlights()

        # --- Persistent X: keep row highlights from projection phases ---
        self._mat_x.pos = self._X_POS
        self._mat_x.clear_highlights()
        for c in range(TOKEN_DIM):
            self._mat_x.set_cell_color(i, c, C_ROW_HL)
            self._mat_x.set_cell_color(j, c, C_COL_HL)

        # --- Q and K^T highlights: progressive element-by-element ---
        self._mat_q.clear_highlights()
        self._mat_kt.clear_highlights()

        # Mark completed dot-product element pairs in result-faded green
        for k in range(step):
            self._mat_q.set_cell_color(i, k, C_RESULT_FADE)
            self._mat_kt.set_cell_color(k, j, C_RESULT_FADE)

        # Highlight current element pair
        if step > 0:
            self._mat_q.set_cell_color(i, step - 1, C_RESULT)
            self._mat_kt.set_cell_color(step - 1, j, C_RESULT)

        # --- Full pre-softmax score row S[i, :] (dynamic, appears at phase start) ---
        score_x = self._KT_POS[0]
        score_y = self._Q_POS[1] + 0.15

        # Build row data: all scores, but j-th cell is partial during computation
        score_row_data = np.copy(s_row)
        if step > 0 and step < TOKEN_DIM:
            score_row_data[j] = round(sum(q_row[k] * k_row[k] for k in range(step)), 2)

        mat_s = MatrixPatch(
            self.fig, score_row_data.reshape(1, -1),
            pos=(score_x, score_y),
            cell_w=CELL_W, cell_h=CELL_H, gap=CELL_GAP,
            row_labels=[f"t{i}"],
            col_labels=[f"t{k}" for k in range(NUM_TOKENS)],
            title=f"S[{i}, :]  (before softmax)",
            text_size=CELL_TEXT_SIZE, label_size=LABEL_SIZE,
        )

        # Colour the row
        mat_s.clear_highlights()
        for c in range(NUM_TOKENS):
            if c == j and step > 0:
                # Cell j: active computation
                if step == TOKEN_DIM:
                    mat_s.set_cell_color(0, c, C_RESULT)
                else:
                    mat_s.set_cell_color(0, c, C_TARGET_HL)
            else:
                # Other cells: normal display
                pass

        if step > 0:
            terms = " + ".join(
                f"{self._format_value(q_row[k])}×{self._format_value(k_row[k])}"
                for k in range(step)
            )
            if step < TOKEN_DIM:
                terms += " + ···"

            if step == TOKEN_DIM:
                self._txt_status.set_text(
                    f"S[{i},{j}] = {terms} = {self._format_value(score)}  "
                    f"(step {step}/{TOKEN_DIM})"
                )
            else:
                partial = round(sum(q_row[k] * k_row[k] for k in range(step)), 2)
                self._txt_status.set_text(
                    f"S[{i},{j}] = {terms} = {self._format_value(partial)}  "
                    f"(step {step}/{TOKEN_DIM})"
                )
        else:
            self._txt_status.set_text(
                f"Press Step to compute S[{i},{j}] = row {i} of Q · col {j} of K^T"
                f"  (step 0/{TOKEN_DIM})"
            )

        # --- Persistent attention map ---
        self._mat_attn.pos = self._ATTN_POS
        self._mat_attn.clear_highlights()
        self._colour_heatmap(self._mat_attn, self.Attn)
        self._mat_attn.set_cell_border(i, j)

        # Render persistent patches
        self._mat_x.render()
        self._mat_attn.render()
        self._mat_wq.render()
        self._mat_q.render()
        self._mat_wk.render()
        self._mat_kt.render()

        # Render dynamic patches
        mat_s.render()
        self._dynamic_patches.append(mat_s)

    # ------------------------------------------------------------------
    # Phase 3: Softmax  A[i,:] = softmax(Scores[i,:])
    # ------------------------------------------------------------------
    def _render_softmax(self, i: int, j: int, step: int) -> None:
        s_row = self.Scores[i]
        a_row = self.Attn[i]

        self._txt_title.set_text(
            f"Step 4/4: {PHASE_NAMES[3]}  |  {PHASE_FORMULAS[3]}  "
            f"|  Computing A[{i}, {j}]"
        )

        # --- Show persistent projection elements (they're still visible in background) ---
        self._show_projection()

        # Clear stale highlights from prior phases
        self._mat_wq.clear_highlights()
        self._mat_wk.clear_highlights()
        self._mat_q.clear_highlights()
        self._mat_kt.clear_highlights()

        # Score row and attention row side by side
        mat_s = MatrixPatch(
            self.fig, s_row.reshape(1, -1),
            pos=(0.65, 0.6),
            row_labels=[f"t{i}"],
            col_labels=[f"t{k}" for k in range(NUM_TOKENS)],
            title=f"S[{i}, :]  (before softmax)",
        )

        mat_a = MatrixPatch(
            self.fig, a_row.reshape(1, -1),
            pos=(0.65, 0.475),
            row_labels=[f"t{i}"],
            col_labels=[f"t{k}" for k in range(NUM_TOKENS)],
            title=f"A[{i}, :]  (after softmax)",
        )

        if step > 0:
            mat_s.set_cell_color(0, j, C_RESULT_FADE)

            # Colour attention row
            for c in range(NUM_TOKENS):
                mat_a.set_cell_color(0, c, C_RESULT_FADE)
            mat_a.set_cell_color(0, j, C_RESULT)

            exp_vals = [f"exp({self._format_value(s_row[k])})"
                        for k in range(NUM_TOKENS)]
            numerator = f"exp({self._format_value(s_row[j])})"
            denom = " + ".join(exp_vals)
            exp_nums = [f"{np.exp(s_row[k] - s_row.max()):.4f}"
                        for k in range(NUM_TOKENS)]
            self._txt_status.set_text(
                f"A[{i},{j}] = {numerator} / ({denom})"
                f"  = {exp_nums[j]} / ({' + '.join(exp_nums)})"
                f"  = {a_row[j]:.4f}   (step {step}/1)"
            )
        else:
            self._txt_status.set_text(
                f"Press Step to apply softmax to row {i}  (step 0/1)"
            )

        # --- Persistent X: keep row highlights from earlier phases ---
        self._mat_x.pos = self._X_POS
        self._mat_x.clear_highlights()
        for c in range(TOKEN_DIM):
            self._mat_x.set_cell_color(i, c, C_ROW_HL)
            self._mat_x.set_cell_color(j, c, C_COL_HL)

        # --- Persistent attention map ---
        self._mat_attn.pos = self._ATTN_POS
        self._mat_attn.clear_highlights()
        self._colour_heatmap(self._mat_attn, self.Attn)
        self._mat_attn.set_cell_border(i, j)

        # Render persistent patches
        self._mat_x.render()
        self._mat_attn.render()
        self._mat_wq.render()
        self._mat_q.render()
        self._mat_wk.render()
        self._mat_kt.render()

        # Render dynamic patches
        mat_s.render()
        self._dynamic_patches.append(mat_s)
        mat_a.render()
        self._dynamic_patches.append(mat_a)

    # ------------------------------------------------------------------
    # Step / Reset callbacks
    # ------------------------------------------------------------------
    def _on_step(self, _: object) -> None:
        if self._mode == "select":
            return

        self._sub_step += 1

        phase_steps = TOKEN_DIM if self._phase < 3 else 1
        if self._sub_step > phase_steps:
            self._phase += 1
            self._sub_step = 1
            if self._phase > 3:
                self._mode = "select"
                self._render_select()
                return

        self._render_trace()

    def _on_reset(self, _: object) -> None:
        self._mode = "select"
        self._phase = 0
        self._sub_step = 0
        if self.fig is not None:
            self._render_select()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self) -> None:
        print("Self-Attention Visualizer started.", flush=True)
        print("Click a cell in the Attention Map, then press Step.",
              flush=True)
        plt.show()


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive Self-Attention Computational Flow Visualizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for matrix generation (default: 42)")
    args = parser.parse_args()

    vis = AttnVisualizer(seed=args.seed)
    vis.run()


if __name__ == "__main__":
    main()
