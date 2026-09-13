# src/performance/shadows.py
"""
Ombres dynamiques — Niveau 4.

Occlusion basee sur la depth map (ray marching vers la lumiere), avec un
modele REALISTE et NUANCE (ombre douce : coeur sombre + penombre claire).

Deux sources de realisme :

1) Occlusion GRADUELLE (soft shadow) : au lieu de decider 0/1 (pleine ombre ou
   pleine lumiere), on compte la PROPORTION de rayons bloques le long du trajet
   vers la lumiere. Un pixel profondement cache (beaucoup de blocages, obstacle
   proche) -> coeur sombre. Un pixel de bordure (peu de blocages) -> penombre
   claire. Cela cree le degrade naturel du bord de l'ombre.

2) Intensite selon la PROFONDEUR du receveur : une ombre sur une surface proche
   est nette et forte ; sur une surface lointaine (fond/vide) elle s'attenue
   jusqu'a disparaitre. (Idee : pas d'ombre plaquee dans l'espace vide.)

Le resultat n'est plus un bloc noir uniforme mais une ombre avec des nuances,
comme dans la vie reelle.
"""

import time
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


N_STEPS = 20
BIAS = 0.02
SHADOW_INTENSITY = 0.6
SHADOW_SOFTNESS = 0.5
SHADOW_DOWNSAMPLE = 4

# Courbe intensite-vs-profondeur du receveur (l'ombre "fond" avec la distance)
RECV_FADE_LO = 0.12   # en dessous : plus d'ombre (vide lointain)
RECV_FADE_HI = 0.75   # au dessus : ombre a pleine force (proche)
RECV_GAMMA = 1.5

# Douceur de l'occlusion graduelle : exposant applique a la proportion de
# rayons bloques. >1 = penombre plus large et douce ; 1 = lineaire.
SOFT_OCCLUSION_GAMMA = 1.3


class ShadowRenderer:
    def __init__(self, n_steps=N_STEPS, bias=BIAS,
                 intensity=SHADOW_INTENSITY,
                 softness=SHADOW_SOFTNESS,
                 downsample=SHADOW_DOWNSAMPLE,
                 recv_fade_lo=RECV_FADE_LO,
                 recv_fade_hi=RECV_FADE_HI,
                 recv_gamma=RECV_GAMMA,
                 soft_occlusion_gamma=SOFT_OCCLUSION_GAMMA):
        self.n_steps = n_steps
        self.bias = bias
        self.intensity = intensity
        self.softness = softness
        self.downsample = downsample
        self.recv_fade_lo = recv_fade_lo
        self.recv_fade_hi = recv_fade_hi
        self.recv_gamma = recv_gamma
        self.soft_occlusion_gamma = soft_occlusion_gamma
        self.total_time_ms = 0.0
        self.n_calls = 0

    def compute(self, depth, light_pos_norm):
        """
        depth          : (H, W) float32 [0,1] (proche = 1)
        light_pos_norm : (x, y) lumiere normalise [0,1]
        Retourne shadow_mask (H, W) : 0 = ombre pleine, 1 = eclaire.
        Valeurs INTERMEDIAIRES = penombre (nuances).
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

        # === Occlusion GRADUELLE : on compte les rayons bloques (pas 0/1) ===
        # occ_count : nb de pas ou le trajet est bloque, pondere par la proximite
        # de l'obstacle (obstacle proche -> ombre plus dense).
        occ_accum = np.zeros((h, w), dtype=np.float32)
        n_inner = max(1, self.n_steps - 1)

        for i in range(1, self.n_steps):
            sx = xx + step_x * i
            sy = yy + step_y * i
            sx_i = np.clip(sx.astype(np.int32), 0, w - 1)
            sy_i = np.clip(sy.astype(np.int32), 0, h - 1)
            sample_depth = depth_small[sy_i, sx_i]
            t = i / self.n_steps
            ray_depth = depth_small * (1 - t) + 0.5 * t
            # profondeur de blocage : combien l'obstacle depasse le rayon
            over = (ray_depth - self.bias) - sample_depth  # >0 si bloque
            blocked = np.clip(over * 6.0, 0.0, 1.0)         # blocage doux [0,1]
            # obstacle proche (t petit) pese plus qu'un obstacle lointain
            weight = 1.0 - 0.5 * t
            occ_accum += blocked * weight

        # Proportion moyenne de blocage -> occlusion douce [0,1]
        occlusion = np.clip(occ_accum / n_inner * 2.0, 0.0, 1.0)
        occlusion = occlusion ** self.soft_occlusion_gamma
        visibility = 1.0 - occlusion

        if self.softness > 0:
            ksize = int(1 + self.softness * 6) | 1
            visibility = cv2.GaussianBlur(visibility, (ksize, ksize), 0)

        # === Intensite selon la profondeur du receveur ===
        span = max(self.recv_fade_hi - self.recv_fade_lo, 1e-6)
        receiver = np.clip((depth_small - self.recv_fade_lo) / span, 0.0, 1.0)
        receiver = receiver ** self.recv_gamma

        occlusion = (1.0 - visibility) * receiver
        visibility = 1.0 - occlusion

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
