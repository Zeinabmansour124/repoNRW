# src/performance/shadows.py
"""
Ombres dynamiques — Niveau 4.

Occlusion basée sur la depth map :
    Pour chaque pixel, on marche le long du rayon vers la lumière
    et on vérifie si un objet plus proche bloque.
"""

import time
import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


# ============================================================
# CONFIGURATION
# ============================================================

# Nombre de pas pour le ray marching (plus = meilleur mais plus lent)
N_STEPS = 16

# Seuil de biais (évite les auto-ombres)
BIAS = 0.02

# Intensité de l'ombre (0 = noir total, 1 = pas d'ombre)
SHADOW_INTENSITY = 0.6

# Douceur (0 = dure, 1 = très douce)
SHADOW_SOFTNESS = 0.5

# Échantillonnage (résolution à laquelle on calcule les ombres)
SHADOW_DOWNSAMPLE = 4


# ============================================================
# CLASSE SHADOW RENDERER
# ============================================================

class ShadowRenderer:
    """
    Calcule les ombres dynamiques à partir de la depth map.
    
    Usage :
        renderer = ShadowRenderer()
        shadow_mask = renderer.compute(depth, light_pos_2d)
        shaded_img = renderer.apply(img, shadow_mask)
    """
    
    def __init__(self, n_steps=N_STEPS, bias=BIAS,
                 intensity=SHADOW_INTENSITY,
                 softness=SHADOW_SOFTNESS,
                 downsample=SHADOW_DOWNSAMPLE):
        self.n_steps = n_steps
        self.bias = bias
        self.intensity = intensity
        self.softness = softness
        self.downsample = downsample
        
        # Stats
        self.total_time_ms = 0.0
        self.n_calls = 0
    
    def compute(self, depth, light_pos_norm):
        """
        Calcule la shadow mask.
        
        Args:
            depth           : np.ndarray (H, W) float32 dans [0, 1]
            light_pos_norm  : (x, y) de la lumière normalisé [0, 1]
                              (0,0 = haut-gauche, 1,1 = bas-droite)
        
        Returns:
            shadow_mask : np.ndarray (H, W) float32
                          0 = ombre totale, 1 = pas d'ombre
        """
        t0 = time.perf_counter()
        
        if depth is None or depth.size == 0:
            return np.ones((1, 1), dtype=np.float32)
        
        H, W = depth.shape
        
        # Sous-échantillonner pour la vitesse
        ds = self.downsample
        depth_small = cv2.resize(depth, (W // ds, H // ds),
                                  interpolation=cv2.INTER_AREA)
        h, w = depth_small.shape
        
        # Position de la lumière en coordonnées pixel
        lx = light_pos_norm[0] * (w - 1)
        ly = light_pos_norm[1] * (h - 1)
        
        # Grille de pixels
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        
        # Vecteur de chaque pixel vers la lumière
        dx = lx - xx
        dy = ly - yy
        
        # Normaliser par pas
        step_x = dx / self.n_steps
        step_y = dy / self.n_steps
        
        # Accumulateur : 0 = dans l'ombre, 1 = éclairé
        visibility = np.ones((h, w), dtype=np.float32)
        
        # Ray marching
        for i in range(1, self.n_steps):
            # Position actuelle le long du rayon
            sx = xx + step_x * i
            sy = yy + step_y * i
            
            # Coordonnées entières
            sx_i = np.clip(sx.astype(np.int32), 0, w - 1)
            sy_i = np.clip(sy.astype(np.int32), 0, h - 1)
            
            # Profondeur du point échantillonné
            sample_depth = depth_small[sy_i, sx_i]
            
            # Profondeur du rayon (interpolée linéairement)
            # Plus on s'approche de la lumière, plus le rayon est "haut"
            t = i / self.n_steps
            ray_depth = depth_small * (1 - t) + 0.5 * t
            
            # Ombre : si l'échantillon est plus proche que le rayon
            is_occluded = (sample_depth < ray_depth - self.bias)
            
            # Accumuler
            visibility[is_occluded] = 0.0
        
        # Adoucir l'ombre
        if self.softness > 0:
            ksize = int(1 + self.softness * 6) | 1   # impair
            visibility = cv2.GaussianBlur(visibility,
                                          (ksize, ksize), 0)
        
        # Ramener à la taille originale
        shadow_mask = cv2.resize(visibility, (W, H),
                                  interpolation=cv2.INTER_LINEAR)
        
        self.total_time_ms += (time.perf_counter() - t0) * 1000
        self.n_calls += 1
        
        return shadow_mask
    
    def apply(self, img, shadow_mask):
        """
        Applique les ombres à une image.
        
        Args:
            img          : np.ndarray (H, W, 3) uint8
            shadow_mask  : np.ndarray (H, W) float32 [0, 1]
        
        Returns:
            img ombrée
        """
        if img is None or shadow_mask is None:
            return img
        
        # Ajuster la résolution de la shadow mask
        h, w = img.shape[:2]
        if shadow_mask.shape != (h, w):
            shadow_mask = cv2.resize(shadow_mask, (w, h),
                                      interpolation=cv2.INTER_LINEAR)
        
        # Facteur : entre (1 - intensity) et 1
        factor = (1 - self.intensity) + self.intensity * shadow_mask
        factor = factor[:, :, None]
        
        result = (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
        return result
    
    def get_stats(self):
        return {
            "n_steps": self.n_steps,
            "n_calls": self.n_calls,
            "avg_time_ms": (self.total_time_ms / self.n_calls
                            if self.n_calls > 0 else 0),
        }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST — Shadow Renderer")
    print("=" * 70)
    
    # Créer une scene simple : un mur + un objet au premier plan
    h, w = 240, 320
    depth = np.ones((h, w), dtype=np.float32) * 0.8   # mur loin
    
    # Objet au premier plan (sphère)
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) < 40 ** 2
    depth[mask] = 0.3
    
    # Image de test (grayscale → RGB)
    img = np.stack([depth * 200] * 3, axis=-1).astype(np.uint8)
    
    renderer = ShadowRenderer()
    
    # Tester différentes positions de lumière
    for light_pos in [(0.2, 0.2), (0.5, 0.2), (0.8, 0.2)]:
        shadow_mask = renderer.compute(depth, light_pos)
        shaded = renderer.apply(img, shadow_mask)
        
        print(f"Light at {light_pos} → "
              f"{renderer.total_time_ms / renderer.n_calls:.1f} ms")
    
    print(f"\nStats : {renderer.get_stats()}")
    
    if _CV2_AVAILABLE:
        shadow_mask = renderer.compute(depth, (0.2, 0.2))
        shaded = renderer.apply(img, shadow_mask)
        cv2.imshow("Original", img)
        cv2.imshow("Shadow Mask", (shadow_mask * 255).astype(np.uint8))
        cv2.imshow("Shaded", shaded)
        cv2.waitKey(0)
        cv2.destroyAllWindows()