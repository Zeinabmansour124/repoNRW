# src/performance/lighting.py
"""
Level 2 — DYNAMIC RELIGHTING
Applique un eclairage Lambertian (diffus) + specular sur la geometrie 3D,
a partir de la normal map et d'une lumiere virtuelle mobile.
"""

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class VirtualLight:
    """
    Lumiere virtuelle ponctuelle contralee par la souris.
    Position dans un espace normalise :
        x, y  dans [-1, 1]  (gauche/droite, haut/bas)
        z     dans [0.2, 3] (proche/loin de la scene)
    """
    def __init__(self, x=0.0, y=0.0, z=1.0):
        self.x = x
        self.y = y
        self.z = z

    def set_from_mouse(self, mx, my, w, h):
        # Souris (pixels) -> position lumiere normalisee
        self.x = (mx / w - 0.5) * 2.0     # -1 .. 1
        self.y = (my / h - 0.5) * 2.0
        # z reste controle par la molette / touches

    def move_z(self, delta):
        self.z = float(np.clip(self.z + delta, 0.2, 3.0))

    def as_tuple(self):
        return (self.x, self.y, self.z)


def relight(normal_map, depth, light, rgb=None,
            ambient=0.15, diffuse_strength=1.0,
            specular_strength=0.5, shininess=32):
    """
    Calcule l'image eclairee.

    normal_map : (H, W, 3) float32, valeurs [-1,1]  (des normales, Role 2)
    depth      : (H, W)   float32, [0,1]
    light      : VirtualLight
    rgb        : (H, W, 3) uint8 optionnel — couleur de base de la scene

    Retour : (H, W, 3) uint8 — scene eclairee (BGR)
    """
    h, w = depth.shape

    # --- Position 3D de chaque pixel (deprojection simple) ---
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    px = (u / w - 0.5) * 2.0
    py = (v / h - 0.5) * 2.0
    pz = depth                      # profondeur [0,1]
    P = np.stack([px, py, pz], axis=-1).astype(np.float32)

    # --- Vecteur vers la lumiere (L), normalise ---
    L = np.array([light.x, light.y, light.z], dtype=np.float32) - P
    L_norm = np.linalg.norm(L, axis=-1, keepdims=True) + 1e-6
    L = L / L_norm

    # --- Normales ---
    N = normal_map.astype(np.float32)

    # --- Diffus (Lambert) : max(0, N . L) ---
    diff = np.sum(N * L, axis=-1)
    diff = np.clip(diff, 0.0, 1.0)

    # --- Specular (Blinn-Phong) ---
    # Vue = vers la camera (0,0,1)
    V = np.zeros_like(P); V[..., 2] = 1.0
    Hh = L + V
    Hh = Hh / (np.linalg.norm(Hh, axis=-1, keepdims=True) + 1e-6)
    spec = np.sum(N * Hh, axis=-1)
    spec = np.clip(spec, 0.0, 1.0) ** shininess

    # --- Attenuation avec la distance a la lumiere ---
    atten = 1.0 / (1.0 + 0.3 * (L_norm.squeeze(-1) ** 2))

    # --- Combinaison ---
    intensity = ambient + diffuse_strength * diff * atten
    intensity = np.clip(intensity, 0.0, 1.0)

    # Couleur de base : la scene RGB, ou gris neutre
    if rgb is not None and rgb.shape[:2] == (h, w):
        base = rgb.astype(np.float32)          # deja BGR dans le pipeline
    else:
        base = np.full((h, w, 3), 200.0, dtype=np.float32)

    lit = base * intensity[..., None]
    # Ajouter le reflet specular (blanc)
    lit += (specular_strength * spec * atten * 255.0)[..., None]

    lit = np.clip(lit, 0, 255).astype(np.uint8)
    return lit


def draw_light_marker(img, light, w, h):
    """Dessine un petit repere de la lumiere (position projetee) sur l'image."""
    if not _CV2_AVAILABLE:
        return img
    lx = int((light.x / 2.0 + 0.5) * w)
    ly = int((light.y / 2.0 + 0.5) * h)
    # taille du halo selon z (proche = gros)
    r = int(np.clip(30 / light.z, 6, 40))
    cv2.circle(img, (lx, ly), r, (0, 255, 255), 2)
    cv2.circle(img, (lx, ly), 3, (0, 255, 255), -1)
    cv2.putText(img, f"Z={light.z:.1f}", (lx + 10, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return img
