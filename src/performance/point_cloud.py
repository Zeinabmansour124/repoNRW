# src/performance/point_cloud.py
"""
Sortie 3D — Nuage de points en perspective, repère FIXE et STABLE.
Améliorations : plus dense, z-ordering (points proches devant), points
proportionnels à la profondeur, ombrage par distance, lissage temporel.
"""

import time
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


DOWNSAMPLE = 5             # equilibre densite/vitesse
MIN_DEPTH = 0.0
MAX_DEPTH = 1.0
OUTPUT_SIZE = (480, 640)   # (h, w)

# --- Vue 3D FIXE (angle 3/4) ---
VIEW_ANGLE_Y = 0.0     # 0 = vue de FACE (objet droit, pas incline)
VIEW_ANGLE_X = 0.0     # pas de plongee -> objet bien droit

# --- Lissage temporel de la depth (anti-jitter) ---
TEMPORAL_ALPHA = 0.4


class PointCloudBuilder:
    def __init__(self, downsample=DOWNSAMPLE,
                 min_depth=MIN_DEPTH, max_depth=MAX_DEPTH,
                 output_size=OUTPUT_SIZE):
        self.downsample = downsample
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.output_size = output_size
        self._prev_depth = None
        self.n_points = 0
        self.total_time_ms = 0.0
        self.n_calls = 0

    def build(self, depth, rgb=None, K=None):
        t0 = time.perf_counter()
        if depth is None or depth.size == 0:
            return np.zeros((0, 3), dtype=np.float32), None

        depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)

        if self._prev_depth is not None and self._prev_depth.shape == depth.shape:
            depth = TEMPORAL_ALPHA * depth + (1.0 - TEMPORAL_ALPHA) * self._prev_depth
        self._prev_depth = depth.copy()

        h, w = depth.shape
        ds = self.downsample

        d = depth[::ds, ::ds]
        hh, ww = d.shape
        u, v = np.meshgrid(np.arange(ww), np.arange(hh))

        X = (u / ww - 0.5)
        Y = (v / hh - 0.5)
        Z = d

        mask = (Z >= self.min_depth) & (Z <= self.max_depth)
        points = np.stack([X[mask], Y[mask], Z[mask]], axis=-1).astype(np.float32)

        colors = None
        if rgb is not None and rgb.shape[:2] == depth.shape:
            rgb_down = rgb[::ds, ::ds]
            colors = rgb_down[mask].astype(np.uint8)

        self.n_points = len(points)
        self.total_time_ms += (time.perf_counter() - t0) * 1000
        self.n_calls += 1
        return points, colors

    def render_2d(self, points, colors=None):
        """Projection perspective, repère fixe, avec z-ordering et ombrage."""
        h, w = self.output_size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        if points is None or len(points) == 0:
            return img

        ay, ax = VIEW_ANGLE_Y, VIEW_ANGLE_X
        cay, say = np.cos(ay), np.sin(ay)
        cax, sax = np.cos(ax), np.sin(ax)

        X = points[:, 0]
        Y = points[:, 1]
        Zc = points[:, 2] - 0.5

        # Rotations fixes
        Xr = X * cay + Zc * say
        Zr = -X * say + Zc * cay
        Yr = Y * cax - Zr * sax
        Zr = Y * sax + Zr * cax

        # Projection perspective
        f = 1.3
        denom = Zr + 2.0
        denom[np.abs(denom) < 1e-3] = 1e-3
        px = (Xr * f / denom + 0.5) * (w - 1)
        py = (Yr * f / denom + 0.5) * (h - 1)

        px = np.clip(px, 0, w - 1).astype(np.int32)
        py = np.clip(py, 0, h - 1).astype(np.int32)

        # Couleur de base
        if colors is None:
            zc = (points[:, 2] - points[:, 2].min()) / (np.ptp(points[:, 2]) + 1e-6)
            col = cv2.applyColorMap((zc * 255).astype(np.uint8),
                                    cv2.COLORMAP_TURBO).reshape(-1, 3).astype(np.float32)
        else:
            col = colors[:, ::-1].astype(np.float32)   # RGB -> BGR

        # --- Ombrage par distance : les points lointains sont assombris ---
        prox = 1.0 - (points[:, 2] - points[:, 2].min()) / (np.ptp(points[:, 2]) + 1e-6)
        shade = (0.45 + 0.55 * prox)[:, None]      # 0.45 (loin) .. 1.0 (proche)
        col = np.clip(col * shade, 0, 255).astype(np.uint8)

        # --- Z-ORDERING : dessiner du plus loin au plus proche ---
        order = np.argsort(-points[:, 2])          # loin d'abord
        px, py, col = px[order], py[order], col[order]
        prox_sorted = prox[order]

        # --- Taille de point proportionnelle à la proximité (proche = plus gros) ---
        sizes = (1 + (prox_sorted * 2)).astype(np.int32)   # 1..3

        # Dessin par blocs (vectorisé par taille)
        for s in (1, 2, 3):
            m = sizes == s
            if not np.any(m):
                continue
            xs, ys, cs = px[m], py[m], col[m]
            for dy in range(s):
                for dx in range(s):
                    yy = np.clip(ys + dy, 0, h - 1)
                    xx = np.clip(xs + dx, 0, w - 1)
                    img[yy, xx] = cs

        return img

    def get_stats(self):
        return {
            "downsample": self.downsample,
            "n_calls": self.n_calls,
            "last_n_points": self.n_points,
            "avg_time_ms": (self.total_time_ms / self.n_calls if self.n_calls > 0 else 0),
        }


def save_ply(points, colors, filepath):
    if points is None or len(points) == 0:
        return False
    has_colors = colors is not None and len(colors) == len(points)
    with open(filepath, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_colors:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            x, y, z = points[i]
            if has_colors:
                r, g, b = colors[i]
                f.write(f"{x:.4f} {y:.4f} {z:.4f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.4f} {y:.4f} {z:.4f}\n")
    return True
