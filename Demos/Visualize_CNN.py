#!/usr/bin/env python3
"""
Code written by Nicholas J. Cooper with help from Claude Code (VSCode frontend only) w/ backend model Qwen3.6-27B-Q6_K (available from here: https://huggingface.co/unsloth/Qwen3.6-27B-MTP-GGUF/blob/main/Qwen3.6-27B-Q6_K.gguf)

Interactive CNN Feature Map Visualizer.

Demonstrates how a pretrained ShuffleNet v2 (ImageNet) extracts features
at multiple spatial resolutions from live webcam input.  Shows the top-5
classification predictions alongside feature maps from four intermediate
stages of the network.

Usage:
    python Run_CNN.py
"""

import argparse
import time

import cv2
import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import torch
import torchvision.models as models


# ─────────────────────────────────────────────
# Layout constants [left, bottom, width, height]
# ─────────────────────────────────────────────
#  Figure: 18 × 15 inches  (1728 × 1440 px at 96dpi)
#  Left column: webcam → banner → predictions  (x=0.02–0.30)
#  Right side: 4 stage blocks  (x=0.33–0.84)
#  Each stage: 2-row × 8-col grid  (16 feature maps)
#  Cell: 0.062 (~106 px) — ~100% larger than original ~52 px

_AX_WEBCAM  = [0.02, 0.44, 0.28, 0.52]
_AX_BANNER  = [0.02, 0.39, 0.28, 0.05]
_AX_CLASS   = [0.02, 0.18, 0.28, 0.19]

_STAGE_ROWS = 2
_STAGE_COLS = 8
_CELL       = 0.100
_CELL_GAP   = 0.0075          # within a stage block
_STAGE_GAP  = 0.025          # between stages

CELL_HORIZ_SPACING_MULT = 0.7

_STAGE_BLOCK_H = _STAGE_ROWS * _CELL + (_STAGE_ROWS - 1) * _CELL_GAP  # 0.127
_STAGE_BLOCK_W = _STAGE_COLS * _CELL - 0.25 # + (_STAGE_COLS - 1) * _CELL_GAP  # 0.511

# Position stages vertically filling 0.08 – 0.78
_STAGE1_TOP = 0.78
_STAGE2_TOP = _STAGE1_TOP - _STAGE_BLOCK_H - _STAGE_GAP
_STAGE3_TOP = _STAGE2_TOP - _STAGE_BLOCK_H - _STAGE_GAP
_STAGE4_TOP = _STAGE3_TOP - _STAGE_BLOCK_H - _STAGE_GAP

STAGE_LEFT = 0.385
_AX_STAGE4 = [STAGE_LEFT, _STAGE4_TOP, _STAGE_BLOCK_W, _STAGE_BLOCK_H]
_AX_STAGE3 = [STAGE_LEFT, _STAGE3_TOP, _STAGE_BLOCK_W, _STAGE_BLOCK_H]
_AX_STAGE2 = [STAGE_LEFT, _STAGE2_TOP, _STAGE_BLOCK_W, _STAGE_BLOCK_H]
_AX_STAGE1 = [STAGE_LEFT, _STAGE1_TOP, _STAGE_BLOCK_W, _STAGE_BLOCK_H]

STAGE_LABEL_LEFT = 0.355
_LABEL_Y = lambda top, h: top + h * 0.55
_LABEL_S1 = [STAGE_LABEL_LEFT, _LABEL_Y(_AX_STAGE1[1], _STAGE_BLOCK_H), 0.08, 0.08]
_LABEL_S2 = [STAGE_LABEL_LEFT, _LABEL_Y(_AX_STAGE2[1], _STAGE_BLOCK_H), 0.08, 0.08]
_LABEL_S3 = [STAGE_LABEL_LEFT, _LABEL_Y(_AX_STAGE3[1], _STAGE_BLOCK_H), 0.08, 0.08]
_LABEL_S4 = [STAGE_LABEL_LEFT, _LABEL_Y(_AX_STAGE4[1], _STAGE_BLOCK_H), 0.08, 0.08]

_BUT_PAUSE  = [0.02, 0.02, 0.12, 0.045]
_BUT_CLOSE  = [0.86, 0.02, 0.12, 0.045]
_SLR_FPS    = [0.20, 0.02, 0.25, 0.04]
_TXT_DEVICE = [0.50, 0.02, 0.34, 0.045]

NUM_CHANNELS = 16  # feature maps shown per stage (2 rows × 8 cols)
STAGES = ("stage1", "stage2", "stage3", "stage4")


# ─────────────────────────────────────────────
# CNN Visualizer
# ─────────────────────────────────────────────
class CNNVisualizer:
    """Matplotlib interactive window that streams webcam frames through
    a pretrained ShuffleNet v2 and visualizes intermediate feature maps.

    Layout
    ------
    Top-left   – live webcam preview (mirrored, 224×224) + prediction banner.
    Top-right  – top-5 ImageNet classification predictions with bars.
    Lower half – four stage blocks, each a 2-row × 8-col grid of 16 feature maps:
                  stage1 (56×56), stage2 (28×28),
                  stage3 (14×14), stage4 (7×7).
    Bottom     – Pause/Resume button, FPS slider, Close button.
    """

    # -----------------------------------------------------------------------
    def __init__(self, device: torch.device, webcam_idx: int = 0):
        self.device = device
        self.webcam_idx = webcam_idx
        self._paused = False
        self._running = True

        # --- Load model ---------------------------------------------------
        print(f"Loading ShuffleNet v2 x1_0 on {device} ...", flush=True)
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT
        self.model = models.shufflenet_v2_x1_0(weights=weights)
        self.model.eval().to(device)
        self.categories = weights.meta["categories"]
        if device.type == "cuda":
            mem_mb = torch.cuda.memory_allocated(device) / 1e6
            print(f"  Model loaded.  CUDA memory: {mem_mb:.1f} MB", flush=True)
        else:
            print("  Model loaded.", flush=True)

        # --- Feature storage ----------------------------------------------
        self.features: dict[str, torch.Tensor] = {}

        # --- Webcam -------------------------------------------------------
        self.cap = cv2.VideoCapture(webcam_idx)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            print(f"  Webcam {webcam_idx} opened.", flush=True)
        else:
            print(f"  WARNING: Webcam {webcam_idx} not available, "
                  "using placeholder.", flush=True)

        # --- Preprocessing constants (match ImageNet) ---------------------
        self.IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
        self.IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

        # --- Prediction smoothing ----------------------------------------
        self._pred_history: list[tuple[float, str]] = []
        self._last_banner_update = 0.0

        # --- Build UI -----------------------------------------------------
        self._build_ui()

    # -----------------------------------------------------------------------
    # Manual forward pass that captures intermediate features
    # -----------------------------------------------------------------------
    def _forward_with_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference while capturing intermediate stage outputs."""
        self.features.clear()

        feat = x
        feat = self.model.conv1(feat)
        feat = self.model.maxpool(feat)
        self.features["stage1"] = feat  # 56×56 × 24

        for i in (2, 3, 4):
            stage = getattr(self.model, f"stage{i}")
            feat = stage(feat)
            self.features[f"stage{i}"] = feat

        feat = self.model.conv5(feat)
        logits = self.model.fc(feat.mean([2, 3]))
        return logits

    # -----------------------------------------------------------------------
    # Webcam capture
    # -----------------------------------------------------------------------
    def _capture_frame(self) -> np.ndarray | None:
        """Read one frame from webcam, return RGB numpy array or None."""
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, 1)
        return frame

    @staticmethod
    def _make_placeholder() -> np.ndarray:
        """Return a gray placeholder image when no webcam is available."""
        return np.full((480, 640, 3), 128, dtype=np.uint8)

    # -----------------------------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------------------------
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert an RGB frame to a normalized ImageNet tensor [1,3,224,224]."""
        h, w = frame.shape[:2]
        scale = 256 / min(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)
        cy, cx = new_h // 2, new_w // 2
        cropped = resized[cy-112:cy+112, cx-112:cx+112]
        tensor = torch.from_numpy(cropped).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(self.IMAGENET_MEAN).view(3, 1, 1)
        std  = torch.tensor(self.IMAGENET_STD).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor.unsqueeze(0)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------
    def _infer(self, tensor: torch.Tensor) -> list[tuple[str, float]]:
        """Run a single forward pass. Returns top-5 (label, probability)."""
        tensor = tensor.to(self.device)
        with torch.no_grad():
            logits = self._forward_with_features(tensor)

        probs = torch.softmax(logits, dim=1)[0]
        top5_probs, top5_idx = torch.topk(probs, 5)
        results = []
        for p, idx in zip(top5_probs.cpu().numpy(), top5_idx.cpu().numpy()):
            results.append((self.categories[int(idx)], float(p)))
        return results

    # -----------------------------------------------------------------------
    # Channel selection
    # -----------------------------------------------------------------------
    @staticmethod
    def _select_channels(feat: np.ndarray, num: int = NUM_CHANNELS) -> list[int]:
        """Return indices of the *num* most active channels (by mean abs).

        feat shape: (C, H, W) on CPU numpy.
        """
        c_mean = np.abs(feat).mean(axis=(1, 2))
        return np.argsort(c_mean)[::-1][:num].tolist()

    @staticmethod
    def _normalize_channel(channel: np.ndarray) -> np.ndarray:
        """Normalize a 2-D channel to [0, 1]."""
        lo, hi = channel.min(), channel.max()
        if hi - lo < 1e-6:
            return np.zeros_like(channel, dtype=np.float32)
        return ((channel - lo) / (hi - lo)).astype(np.float32)

    # -----------------------------------------------------------------------
    # Prediction smoothing
    # -----------------------------------------------------------------------
    def _update_banner(self, top1_label: str) -> None:
        """Maintain a rolling 5-second window of top-1 labels.

        Update the banner text once per second regardless of framerate.
        """
        now = time.time()
        if now - self._last_banner_update < 1.0:
            return

        self._pred_history.append((now, top1_label))
        cutoff = now - 5.0
        self._pred_history = [(t, l) for t, l in self._pred_history
                              if t > cutoff]

        if self._pred_history:
            labels = [l for _, l in self._pred_history]
            stable = max(set(labels), key=labels.count)
        else:
            stable = top1_label

        self._txt_banner.set_text(f"  {stable}")
        self._last_banner_update = now

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self):
        mpl.rcParams.update({"font.size": 19, "axes.titlesize": 23})

        self.fig = plt.figure(figsize=(18, 15))
        self.fig.canvas.manager.set_window_title(
            "CNN Feature Visualizer – ShuffleNet v2")

        # -- Webcam axis --
        self.ax_webcam = self.fig.add_axes(_AX_WEBCAM)
        self.ax_webcam.set_aspect("equal")
        self._im_webcam = self.ax_webcam.imshow(
            np.zeros((224, 224, 3), dtype=np.uint8))
        self.ax_webcam.axis("off")
        self.ax_webcam.set_title("Webcam Input", pad=8)

        # -- Prediction banner (below webcam) --
        self.ax_banner = self.fig.add_axes(_AX_BANNER)
        self.ax_banner.axis("off")
        self._txt_banner = self.ax_banner.text(
            0.5, 0.55, "", transform=self.ax_banner.transAxes,
            fontsize=31, verticalalignment="center",
            horizontalalignment="center", fontweight="bold",
            color="#E0E0E0",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#2B2B2B",
                      edgecolor="#555555", alpha=0.95))

        # -- Classification panel --
        self.ax_class = self.fig.add_axes(_AX_CLASS)
        self.ax_class.axis("off")
        self.ax_class.set_title("Top-5 Predictions", pad=8)

        self._class_lines: list = []
        self._class_bars: list = []
        for i in range(5):
            y = 0.88 - i * 0.15
            line = self.ax_class.text(
                0.05, y + 0.05, "  " * 20,
                transform=self.ax_class.transAxes, fontsize=17,
                verticalalignment="top", family="monospace")
            self._class_lines.append(line)
            bar = plt.Rectangle((0.05, y), 0.0, 0.04,
                                transform=self.ax_class.transAxes,
                                facecolor="#4C72B0", edgecolor="none",
                                alpha=0.7)
            self.ax_class.add_patch(bar)
            self._class_bars.append(bar)

        # -- Feature map stages  (2 rows × 8 cols per stage) --
        self.stage_axes: dict[str, list] = {}
        self.stage_imags: dict[str, list] = {}

        stage_rects = {
            "stage1": _AX_STAGE1,
            "stage2": _AX_STAGE2,
            "stage3": _AX_STAGE3,
            "stage4": _AX_STAGE4,
        }
        for stage_name in STAGES:
            rect = stage_rects[stage_name]
            rx, ry, rw, rh = rect
            axes_list, imags_list = [], []

            for idx in range(NUM_CHANNELS):
                row = idx // _STAGE_COLS
                col = idx % _STAGE_COLS
                ax_x = rx + col * (CELL_HORIZ_SPACING_MULT * (_CELL + _CELL_GAP))
                ax_y = ry + row * (_CELL + _CELL_GAP)
                ax = self.fig.add_axes([ax_x, ax_y, _CELL, _CELL])
                ax.axis("off")
                axes_list.append(ax)
                imags_list.append(None)

            self.stage_axes[stage_name] = axes_list
            self.stage_imags[stage_name] = imags_list

        # -- Stage labels --
        stage_info = {
            "stage1": ("Stage 1", "56×56", _LABEL_S1),
            "stage2": ("Stage 2", "28×28", _LABEL_S2),
            "stage3": ("Stage 3", "14×14", _LABEL_S3),
            "stage4": ("Stage 4", "7×7", _LABEL_S4),
        }
        for _sname, (title, res, rect) in stage_info.items():
            self.fig.text(rect[0], rect[1],
                          f"{title}\n({res})", fontsize=17,
                          verticalalignment="center",
                          horizontalalignment="center",
                          fontweight="bold")

        # -- Controls --
        ax_pause = self.fig.add_axes(_BUT_PAUSE)
        self.btn_pause = Button(ax_pause, "Pause", hovercolor="#7bf1a8")
        self.btn_pause.on_clicked(self._on_pause)

        ax_close = self.fig.add_axes(_BUT_CLOSE)
        self.btn_close = Button(ax_close, "Close", hovercolor="#fcb44b")
        self.btn_close.on_clicked(self._on_close)

        ax_fps = self.fig.add_axes(_SLR_FPS)
        self.sl_fps = Slider(ax_fps, "Target FPS", 1, 30, valinit=10,
                             valstep=1, valfmt="%d")
        self.sl_fps.on_changed(self._on_fps_change)

        dev_text = f"Device: {self.device}"
        self.fig.text(_TXT_DEVICE[0], _TXT_DEVICE[1] + 0.02,
                      dev_text, fontsize=17,
                      verticalalignment="center",
                      horizontalalignment="center",
                      color="gray")

        # -- Timer --
        self._timer_interval = 1000.0 / self.sl_fps.val
        self.timer = self.fig.canvas.new_timer(
            interval=int(self._timer_interval))
        self.timer.add_callback(self._on_timer, ())

        self.fig.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Timer callback (main loop)
    # -----------------------------------------------------------------------
    def _on_timer(self, *_: any):
        if not self._running:
            return

        frame = self._capture_frame()
        if frame is None:
            frame = self._make_placeholder()

        display = cv2.resize(frame, (224, 224),
                             interpolation=cv2.INTER_LINEAR)
        self._im_webcam.set_data(display)

        tensor = self._preprocess(frame).to(torch.float32)
        top5 = self._infer(tensor)

        for i, (label, prob) in enumerate(top5):
            pct = prob * 100
            self._class_lines[i].set_text(
                f"{i+1}. {label:<30s} {pct:5.1f}%")
            self._class_bars[i].set_width(prob * 0.9)

        if top5:
            self._update_banner(top5[0][0])

        for stage_name in STAGES:
            feat = self.features.get(stage_name)
            if feat is None:
                continue
            feat_np = feat[0].cpu().numpy()  # [C, H, W]
            channels = self._select_channels(feat_np, NUM_CHANNELS)

            for i, ax in enumerate(self.stage_axes[stage_name]):
                ch_data = self._normalize_channel(feat_np[channels[i]])
                im = self.stage_imags[stage_name][i]
                if im is None:
                    im = ax.imshow(ch_data, cmap="viridis", aspect="equal")
                    self.stage_imags[stage_name][i] = im
                else:
                    im.set_data(ch_data)

        self.fig.canvas.draw_idle()

    # -----------------------------------------------------------------------
    # Button/slider callbacks
    # -----------------------------------------------------------------------
    def _on_pause(self, _: any):
        self._paused = not self._paused
        self.btn_pause.label.set_text("Resume" if self._paused else "Pause")
        if self._paused:
            self.timer.stop()
        else:
            self.timer.start()

    def _on_close(self, _: any):
        self._running = False
        self.timer.stop()
        self.cleanup()
        plt.close(self.fig)

    def _on_fps_change(self, val: float):
        self._timer_interval = 1000.0 / val
        was_paused = self._paused
        self.timer.stop()
        self.timer.interval = int(self._timer_interval)
        if not was_paused:
            self.timer.start()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------
    def cleanup(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def run(self):
        print("Starting visualization.  Use Pause/Close buttons.",
              flush=True)
        self.timer.start()
        plt.show()
        self.cleanup()


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Interactive CNN Feature Map Visualizer")
    parser.add_argument("--webcam", type=int, default=0,
                        help="Webcam device index (default: 0)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU inference (default: use CUDA if available)")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        print("WARNING: CUDA not available, falling back to CPU.",
              flush=True)
        device = torch.device("cpu")

    vis = CNNVisualizer(device=device, webcam_idx=args.webcam)
    vis.run()


if __name__ == "__main__":
    main()
