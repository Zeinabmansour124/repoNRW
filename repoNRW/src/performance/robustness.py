# src/performance/robustness.py


import time

try:
    import cv2
    import numpy as np
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False




TARGET_FPS = 30            # 🔴 JOUR J : 8 (CPU lent), 12 (moyen), 15 (rapide), 30 (GPU)
ENABLE_CLAHE = False         # 🔴 JOUR J : True si scènes sombres




class LowLightEnhancer:
    """
    Améliore les frames sombres via CLAHE.
    
    Applique CLAHE UNIQUEMENT si la frame est très sombre
    (mean < threshold=50) pour éviter de dégrader les scènes claires.
    
    Usage :
        enhancer = LowLightEnhancer(threshold=50)
        frame = enhancer.enhance(frame)
    """
    
    DEFAULT_THRESHOLD = 50     # Seuil bas (nuit seulement)
    DEFAULT_CLIP_LIMIT = 2.0
    DEFAULT_TILE_SIZE = (8, 8)
    
    def __init__(self, threshold=None, clip_limit=None,
                 tile_size=None, enabled=True):
        if not _CV2_AVAILABLE:
            self.enabled = False
            return
        
        self.threshold = threshold or self.DEFAULT_THRESHOLD
        self.clip_limit = clip_limit or self.DEFAULT_CLIP_LIMIT
        self.tile_size = tile_size or self.DEFAULT_TILE_SIZE
        self.enabled = enabled
        
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_size
        )
        
        # Stats
        self.n_applied = 0
        self.n_skipped = 0
        self.total_time_ms = 0.0
        
        # Warm-up
        if self.enabled:
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.enhance(dummy)
            self.reset_stats()
    
    def enhance(self, frame):
        if not self.enabled or frame is None:
            return frame
        
        brightness = frame.mean()
        if brightness >= self.threshold:
            self.n_skipped += 1
            return frame
        
        t0 = time.perf_counter()
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        self.total_time_ms += (time.perf_counter() - t0) * 1000
        self.n_applied += 1
        return result
    
    def reset_stats(self):
        self.n_applied = 0
        self.n_skipped = 0
        self.total_time_ms = 0.0
    
    def get_stats(self):
        total = self.n_applied + self.n_skipped
        return {
            "enabled": self.enabled,
            "threshold": self.threshold,
            "applied": self.n_applied,
            "skipped": self.n_skipped,
            "apply_rate": self.n_applied / total if total > 0 else 0,
            "avg_time_ms": (self.total_time_ms / self.n_applied
                            if self.n_applied > 0 else 0),
        }




class AdaptiveMode:
    """
    Mode dégradé avec 3 niveaux.
    
    NORMAL   : imgsz=384, skip=1  (qualité max)
    DEGRADED : imgsz=256, skip=1  (compromis)
    MINIMAL  : imgsz=192, skip=2  (survie)
    """
    
    NORMAL = "normal"
    DEGRADED = "degraded"
    MINIMAL = "minimal"
    
    CONFIG = {
        NORMAL:   {"imgsz": 384, "skip": 1},
        DEGRADED: {"imgsz": 256, "skip": 1},
        MINIMAL:  {"imgsz": 192, "skip": 1},
    }
    
    def __init__(self, target_fps=TARGET_FPS, hysteresis=3):
        self.target_fps = target_fps
        self.hysteresis = hysteresis
        self.mode = self.DEGRADED
        self._candidate_mode = self.NORMAL
        self._candidate_count = 0
        self.fps_history = []
        self.window_size = 10
        self.mode_changes = []
        self.frames_in_mode = {
            self.NORMAL: 0, self.DEGRADED: 0, self.MINIMAL: 0,
        }
    
    @property
    def imgsz(self):
        return self.CONFIG[self.mode]["imgsz"]
    
    @property
    def skip_frames(self):
        return self.CONFIG[self.mode]["skip"]
    
    def _decide_mode(self, fps):
        if fps < self.target_fps * 0.5:
            return self.MINIMAL
        elif fps < self.target_fps * 0.8:
            return self.DEGRADED
        return self.NORMAL
    
    def update(self, measured_fps, frame_id=None):
        self.fps_history.append(measured_fps)
        if len(self.fps_history) > self.window_size:
            self.fps_history.pop(0)
        avg_fps = sum(self.fps_history) / len(self.fps_history)
        
        self.frames_in_mode[self.mode] += 1
        target = self._decide_mode(avg_fps)
        
        if target == self.mode:
            self._candidate_count = 0
            return
        
        if target == self._candidate_mode:
            self._candidate_count += 1
        else:
            self._candidate_mode = target
            self._candidate_count = 1
        
        if self._candidate_count >= self.hysteresis:
            old = self.mode
            self.mode = target
            self.mode_changes.append({
                "frame_id": frame_id, "from": old,
                "to": target, "avg_fps": avg_fps,
            })
            self._candidate_count = 0
            print(f"[MODE] {old} → {target} "
                  f"(avg_fps={avg_fps:.1f}, imgsz={self.imgsz}, "
                  f"skip={self.skip_frames})")
    
    def should_skip(self, frame_id):
        return (frame_id % self.skip_frames) != 0
    
    def get_stats(self):
        return {
            "mode": self.mode,
            "imgsz": self.imgsz,
            "skip_frames": self.skip_frames,
            "avg_fps": (sum(self.fps_history) / len(self.fps_history)
                        if self.fps_history else 0),
            "mode_changes": len(self.mode_changes),
            "frames_per_mode": dict(self.frames_in_mode),
        }


# ============================================================
# 3. GESTION D'ERREURS — SafeDepth
# ============================================================

class SafeDepth:
    """Wrapper autour du depth avec gestion d'erreurs."""
    
    def __init__(self, depth_fn, max_consecutive_errors=10):
        self.depth_fn = depth_fn
        self.max_consecutive_errors = max_consecutive_errors
        self.n_successes = 0
        self.n_errors = 0
        self.n_consecutive_errors = 0
        self.circuit_open = False
        self.circuit_opened_at = None
    
    def _check_circuit(self):
        if not self.circuit_open:
            return
        if time.perf_counter() - self.circuit_opened_at >= 5.0:
            self.circuit_open = False
            self.n_consecutive_errors = 0
            print("[CIRCUIT] Refermé")
    
    def _fallback(self):
        """Retourne une depth map uniforme."""
        return np.full((480, 640), 0.5, dtype=np.float32)
    
    def estimate(self, frame, imgsz=384):
        self._check_circuit()
        if self.circuit_open:
            return self._fallback()
        try:
            result = self.depth_fn(frame, imgsz=imgsz)
            self.n_successes += 1
            self.n_consecutive_errors = 0
            return result
        except Exception as e:
            self.n_errors += 1
            self.n_consecutive_errors += 1
            print(f"[ERROR] Depth échoué: {type(e).__name__}: {e}")
            if self.n_consecutive_errors >= self.max_consecutive_errors:
                self.circuit_open = True
                self.circuit_opened_at = time.perf_counter()
                print(f"[CIRCUIT] OUVERT après "
                      f"{self.n_consecutive_errors} erreurs")
            return self._fallback()
    
    def get_stats(self):
        total = self.n_successes + self.n_errors
        return {
            "successes": self.n_successes,
            "errors": self.n_errors,
            "error_rate": self.n_errors / total if total > 0 else 0,
            "circuit_open": self.circuit_open,
        }




if __name__ == "__main__":
    print("=" * 70)
    print("TEST — CLAHE + Mode dégradé 3 niveaux")
    print("=" * 70)
    
    if not _CV2_AVAILABLE:
        print("❌ OpenCV non installé")
        exit(1)
   
    print("\n[TEST 1] CLAHE")
    enhancer = LowLightEnhancer(threshold=50)
    
    dark = np.random.randint(10, 40, (480, 640, 3), dtype=np.uint8)
    bright = np.full((480, 640, 3), 200, dtype=np.uint8)
    mid = np.full((480, 640, 3), 80, dtype=np.uint8)
    
    enhancer.enhance(dark)
    enhancer.enhance(bright)
    enhancer.enhance(mid)
    
    print(f"  Stats: {enhancer.get_stats()}")
    
    print("\n[TEST 2] AdaptiveMode 3 niveaux")
    mode = AdaptiveMode(target_fps=15)
    
    scenario = [15, 14, 12, 10, 8, 6, 4, 3, 5, 8, 12, 15]
    for i, fps in enumerate(scenario):
        mode.update(fps, frame_id=i)
        print(f"  Frame {i:2d} | FPS={fps:2d} | mode={mode.mode}")
    
    print(f"\n  Stats: {mode.get_stats()}")
    
    print("\n" + "=" * 70)
    print("OK")
    print("=" * 70)