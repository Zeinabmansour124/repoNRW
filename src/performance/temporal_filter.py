
"""
Filtre temporel — Stabilité entre les frames.

Réduit le "flickering" en mélangeant les frames successives.

"""

import numpy as np




# Alpha : facteur de mélange
#   1.0 → pas de filtre (depth brut)
#   0.5 → bon compromis
#   0.3 → filtre fort (plus stable mais plus lent à réagir)
#   0.1 → très stable (léger retard visuel)
ALPHA_DEPTH   = 0.4     #  0.3 si trop de flicker, 0.6 si trop de retard
ALPHA_NORMALS = 0.4     #   pareil


# ============================================================
# CLASSE TEMPORAL FILTER
# ============================================================

class TemporalFilter:
    """
    Filtre temporel par moyenne exponentielle (EMA).
    
    Principe :
        filtered = α × new + (1-α) × old
    
    Usage :
        filt = TemporalFilter(alpha=0.4)
        depth_filtered = filt.apply_depth(depth)
        normals_filtered = filt.apply_normals(normals)
    """
    
    def __init__(self, alpha_depth=ALPHA_DEPTH,
                 alpha_normals=ALPHA_NORMALS,
                 enabled=True):
        """
        Args:
            alpha_depth   : facteur pour la depth (0.0 à 1.0)
            alpha_normals : facteur pour les normales (0.0 à 1.0)
            enabled       : activer/désactiver
        """
        self.alpha_depth = alpha_depth
        self.alpha_normals = alpha_normals
        self.enabled = enabled
        
        # État interne (dernières valeurs filtrées)
        self._prev_depth = None
        self._prev_normals = None
        
        # Stats
        self.n_applied = 0
        self.n_reset = 0
    
    def apply_depth(self, depth):
        """
        Applique le filtre temporel à une depth map.
        
        Args:
            depth : np.ndarray (H, W) float32
        
        Returns:
            depth filtrée
        """
        if not self.enabled or depth is None:
            return depth
        
        # Première frame : initialiser
        if self._prev_depth is None or self._prev_depth.shape != depth.shape:
            self._prev_depth = depth.copy()
            self.n_applied += 1
            return depth
        
        # Filtre EMA
        filtered = (self.alpha_depth * depth +
                    (1.0 - self.alpha_depth) * self._prev_depth)
        
        self._prev_depth = filtered.copy()
        self.n_applied += 1
        return filtered.astype(np.float32)
    
    def apply_normals(self, normals):
        """
        Applique le filtre temporel à une normal map.
        
        Args:
            normals : np.ndarray (H, W, 3) float32
        
        Returns:
            normales filtrées et renormalisées
        """
        if not self.enabled or normals is None:
            return normals
        
        if self._prev_normals is None or self._prev_normals.shape != normals.shape:
            self._prev_normals = normals.copy()
            return normals
        
        # Filtre EMA
        filtered = (self.alpha_normals * normals +
                    (1.0 - self.alpha_normals) * self._prev_normals)
        
        # Renormaliser (important !)
        norms = np.linalg.norm(filtered, axis=-1, keepdims=True)
        filtered = filtered / np.maximum(norms, 1e-8)
        
        self._prev_normals = filtered.copy()
        return filtered.astype(np.float32)
    
    def reset(self):
        """Réinitialise le filtre (nouvelle scène)."""
        self._prev_depth = None
        self._prev_normals = None
        self.n_reset += 1
    
    def get_stats(self):
        return {
            "enabled": self.enabled,
            "alpha_depth": self.alpha_depth,
            "alpha_normals": self.alpha_normals,
            "n_applied": self.n_applied,
            "n_reset": self.n_reset,
        }


# ============================================================
# FONCTION UTILITAIRE — Vérifier la stabilité
# ============================================================

def measure_flickering(depth_sequence):
    """
    Mesure le flickering dans une séquence de depth maps.
    
    Args:
        depth_sequence : liste de np.ndarray
    
    Returns:
        dict avec les stats de flickering
    """
    if len(depth_sequence) < 2:
        return {"mean_diff": 0.0, "max_diff": 0.0}
    
    diffs = []
    for i in range(1, len(depth_sequence)):
        diff = np.mean(np.abs(depth_sequence[i] - depth_sequence[i-1]))
        diffs.append(diff)
    
    return {
        "mean_diff": float(np.mean(diffs)),
        "max_diff": float(np.max(diffs)),
        "n_frames": len(depth_sequence),
    }


if __name__ == "__main__":
    print("=" * 70)
    print("TEST — Filtre temporel (stabilité)")
    print("=" * 70)
    
    # Simuler une depth map stable (mur immobile)
    h, w = 480, 640
    base_depth = np.ones((h, w), dtype=np.float32) * 0.5
    
    # Générer une séquence bruitée (flickering)
    np.random.seed(42)
    n_frames = 50
    noisy_sequence = []
    for i in range(n_frames):
        noise = np.random.normal(0, 0.05, (h, w))
        noisy_sequence.append((base_depth + noise).astype(np.float32))
    
    # Mesurer le flickering SANS filtre
    stats_raw = measure_flickering(noisy_sequence)
    print(f"\n[SANS FILTRE]")
    print(f"  Flickering moyen : {stats_raw['mean_diff']:.4f}")
    print(f"  Flickering max   : {stats_raw['max_diff']:.4f}")
    
    # Appliquer le filtre
    filt = TemporalFilter(alpha_depth=0.3)
    filtered_sequence = []
    for depth in noisy_sequence:
        filtered = filt.apply_depth(depth)
        filtered_sequence.append(filtered)
    
    # Mesurer le flickering AVEC filtre
    stats_filtered = measure_flickering(filtered_sequence)
    print(f"\n[AVEC FILTRE alpha=0.3]")
    print(f"  Flickering moyen : {stats_filtered['mean_diff']:.4f}")
    print(f"  Flickering max   : {stats_filtered['max_diff']:.4f}")
    
    # Réduction
    reduction = (1 - stats_filtered['mean_diff'] / stats_raw['mean_diff']) * 100
    print(f"\n   Réduction du flickering : {reduction:.1f}%")
    
    print(f"\n  Stats filtre : {filt.get_stats()}")
    
    print("\n" + "=" * 70)