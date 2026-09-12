# src/performance/shadows.py
"""
Ombres dynamiques — Niveau 4.
Occlusion basée sur la depth map (ray marching vers la lumière).
"""

import time
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


N_STEPS = 16
BIAS = 0.02
SHADOW_INTENSITY = 0.6
SHADOW_SOFTNESS = 0.5
SHADOW_DOWNSAMPLE = 4


class ShadowRenderer:
    def __init__(self, n_steps=N_STEPS, bias=BIAS,
                 intensity=SHADOW_INTENSITY,
                 softness=SHADOW_SOFTNESS,
                 downsample=SHADOW_DOWNSAMPLE):
        self.n_steps = n_steps
        self.bias = bias
        self.intensity = intensity
        self.softness = softness
        self.downsample = downsample
        self.total_time_ms = 0.0
        self.n_calls = 0

    def compute(self, depth, light_pos_norm):
        """
        depth          : (H, W) float32 [0,1]
        light_pos_norm : (x, y) lumiere normalise [0,1]
        Retourne shadow_mask (H, W) : 0 = ombre, 1 = eclaire.
        """
        t0 = time.perf_counter()
        if depth is None or depth.size == 0:
            return np.ones((1, 1), dtype=np.float32)

        H, W = depth.shape
        ds = self.downsample
        depth_small = cv2.resize(depth, (W // ds, H // ds), interpolation=cv2.INTER_AREA)
        h, w = depth_small.shape

        lx = light_pos_norm[0] * (w - 1)
        ly = light_pos_norm[1] * (h - 1)

        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx = lx - xx
        dy = ly - yy
        step_x = dx / self.n_steps
        step_y = dy / self.n_steps

        visibility = np.ones((h, w), dtype=np.float32)

        for i in range(1, self.n_steps):
            sx = xx + step_x * i
            sy = yy + step_y * i
            sx_i = np.clip(sx.astype(np.int32), 0, w - 1)
            sy_i = np.clip(sy.astype(np.int32), 0, h - 1)
            sample_depth = depth_small[sy_i, sx_i]
            t = i / self.n_steps
            ray_depth = depth_small * (1 - t) + 0.5 * t
            is_occluded = (sample_depth < ray_depth - self.bias)
            visibility[is_occluded] = 0.0

        if self.softness > 0:
            ksize = int(1 + self.softness * 6) | 1
            visibility = cv2.GaussianBlur(visibility, (ksize, ksize), 0)

        shadow_mask = cv2.resize(visibility, (W, H), interpolation=cv2.INTER_LINEAR)
        self.total_time_ms += (time.perf_counter() - t0) * 1000
        self.n_calls += 1
        return shadow_mask

    def apply(self, img, shadow_mask):
        if img is None or shadow_mask is None:
            return img
        h, w = img.shape[:2]
        if shadow_mask.shape != (h, w):
            shadow_mask = cv2.resize(shadow_mask, (w, h), interpolation=cv2.INTER_LINEAR)
        factor = (1 - self.intensity) + self.intensity * shadow_mask
        factor = factor[:, :, None]
        return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)

    def get_stats(self):
        return {
            "n_steps": self.n_steps,
            "n_calls": self.n_calls,
            "avg_time_ms": (self.total_time_ms / self.n_calls if self.n_calls > 0 else 0),
        }
