# src/performance/fast_normals.py
"""
Calcul rapide des normales (vectorisé, meshgrid caché, calcul sur depth réduite).
"""

import numpy as np
import cv2

_UV_CACHE = {}

def _uv(h, w):
    key = (h, w)
    if key not in _UV_CACHE:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        _UV_CACHE[key] = (u, v)
    return _UV_CACHE[key]


def compute_normals_fast(depth, K=None, scale=2):
    """
    Normales par différences finies (vectorisé).
    scale : facteur de réduction pour le calcul (2 = 2x plus petit = 4x plus rapide),
            le résultat est réagrandi à la taille d'origine.
    """
    H, W = depth.shape

    # Calcul sur une version réduite -> beaucoup plus rapide
    if scale > 1:
        small = cv2.resize(depth, (W // scale, H // scale), interpolation=cv2.INTER_LINEAR)
    else:
        small = depth
    h, w = small.shape

    if K is None:
        fx = fy = float(w)
        cx, cy = w / 2.0, h / 2.0
    else:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]

    u, v = _uv(h, w)
    X = (u - cx) * small / fx
    Y = (v - cy) * small / fy
    Z = small
    points = np.stack([X, Y, Z], axis=-1)

    Tu = np.zeros_like(points)
    Tv = np.zeros_like(points)
    Tu[:, :-1] = points[:, 1:] - points[:, :-1]
    Tv[:-1, :] = points[1:, :] - points[:-1, :]

    N = np.cross(Tu, Tv)
    norms = np.linalg.norm(N, axis=-1, keepdims=True)
    N = N / np.maximum(norms, 1e-8)

    # Réagrandir à la taille d'origine
    if scale > 1:
        N = cv2.resize(N, (W, H), interpolation=cv2.INTER_LINEAR)

    return N.astype(np.float32)


def normals_to_rgb_fast(normals):
    rgb = ((normals + 1.0) * 0.5 * 255.0)
    rgb = np.clip((rgb - 128.0) * 1.35 + 128.0, 0, 255)   # boost contraste (notre "wow")
    return rgb.astype(np.uint8)
