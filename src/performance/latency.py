# src/performance/latency.py
"""
Mesure de latence bout-en-bout.

═══════════════════════════════════════════════════════════
🚀 GUIDE JOUR J
═══════════════════════════════════════════════════════════

Aucun changement nécessaire.

Usage :
    lm = LatencyMeter()
    
    t0 = lm.start()           # au moment de la capture
    # ... pipeline ...
    lm.end(t0)                # après l'affichage
    
    lm.report()
═══════════════════════════════════════════════════════════
"""

import time
import numpy as np


# ============================================================
# CLASSE LATENCY METER
# ============================================================

class LatencyMeter:
    """
    Mesure la latence bout-en-bout (capture → affichage).
    
    Usage :
        lm = LatencyMeter()
        
        while True:
            t0 = lm.start()
            ret, frame = cap.read()
            # ... pipeline ...
            cv2.imshow("Demo", frame)
            lm.end(t0)
        
        lm.report()
    """
    
    def __init__(self, target_ms=100, enabled=True):
        """
        Args:
            target_ms : latence cible (pour les couleurs)
            enabled   : activer/désactiver
        """
        self.target_ms = target_ms
        self.enabled = enabled
        
        self.latencies = []      # toutes les latences mesurées
        self.n_frames = 0
        self.total_time = 0.0
    
    def start(self):
        """Marque le début d'une frame."""
        if not self.enabled:
            return 0.0
        return time.perf_counter()
    
    def end(self, t0):
        """
        Marque la fin d'une frame et enregistre la latence.
        
        Args:
            t0 : timestamp retourné par start()
        
        Returns:
            latence en millisecondes
        """
        if not self.enabled:
            return 0.0
        
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000
        
        self.latencies.append(latency_ms)
        self.n_frames += 1
        self.total_time += latency_ms
        
        return latency_ms
    
    def reset(self):
        self.latencies = []
        self.n_frames = 0
        self.total_time = 0.0
    
    def get_stats(self):
        """Retourne les stats de latence."""
        if not self.latencies:
            return {
                "n_frames": 0,
                "mean_ms": 0.0,
                "p50_ms": 0.0,
                "p95_ms": 0.0,
                "p99_ms": 0.0,
                "min_ms": 0.0,
                "max_ms": 0.0,
                "fps": 0.0,
            }
        
        arr = np.array(self.latencies)
        mean = float(np.mean(arr))
        
        return {
            "n_frames": self.n_frames,
            "mean_ms": mean,
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "min_ms": float(np.min(arr)),
            "max_ms": float(np.max(arr)),
            "fps": 1000.0 / mean if mean > 0 else 0,
        }
    
    def color(self, latency_ms):
        """Retourne une couleur BGR selon la latence."""
        if latency_ms <= self.target_ms:
            return (0, 255, 0)       # Vert
        elif latency_ms <= self.target_ms * 2:
            return (0, 165, 255)     # Orange
        return (0, 0, 255)           # Rouge
    
    def report(self, title="Latency Report"):
        """Affiche le rapport."""
        stats = self.get_stats()
        
        print("\n" + "=" * 70)
        print(f"⏱️  {title} — {stats['n_frames']} frames")
        print("=" * 70)
        print(f"  Latence moyenne  : {stats['mean_ms']:.1f} ms")
        print(f"  p50 (médiane)    : {stats['p50_ms']:.1f} ms")
        print(f"  p95              : {stats['p95_ms']:.1f} ms")
        print(f"  p99              : {stats['p99_ms']:.1f} ms")
        print(f"  Min / Max        : {stats['min_ms']:.1f} / {stats['max_ms']:.1f} ms")
        print(f"  FPS équivalent   : {stats['fps']:.1f}")
        print("=" * 70)
        
        # Verdict
        if stats['p95_ms'] <= self.target_ms:
            print(f"  ✅ Excellent (p95 ≤ {self.target_ms} ms)")
        elif stats['p95_ms'] <= self.target_ms * 2:
            print(f"  ⚠️  Acceptable (p95 ≤ {self.target_ms * 2} ms)")
        else:
            print(f"  ❌ Trop lent (p95 > {self.target_ms * 2} ms)")
        
        print("=" * 70)


# ============================================================
# FONCTION UTILITAIRE — Configurer le buffer caméra
# ============================================================

def open_camera_low_latency(source=0):
    """
    Ouvre une caméra avec buffer minimal (latence réduite).
    
    Args:
        source : 0 pour webcam, ou chemin vidéo
    
    Returns:
        cv2.VideoCapture configuré
    """
    import cv2
    
    cap = cv2.VideoCapture(source)
    
    if cap.isOpened():
        # Buffer minimal (important pour la latence)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Résolution standard
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # FPS caméra
        cap.set(cv2.CAP_PROP_FPS, 60)
        
        print(f"[CAMERA] Buffer=1, 640x480, 60 FPS demande")
    
    return cap


# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEST — LatencyMeter")
    print("=" * 70)
    
    # Simuler des latences
    lm = LatencyMeter(target_ms=100)
    
    np.random.seed(42)
    for i in range(100):
        t0 = lm.start()
        # Simuler un pipeline de 70-120 ms
        time.sleep(np.random.uniform(0.070, 0.120))
        lm.end(t0)
    
    lm.report("Pipeline simulé")

# ============================================================
# CAMÉRA THREADÉE : la capture tourne dans son propre thread,
# la boucle principale ne bloque jamais sur cap.read().
# ============================================================
import threading

class ThreadedCamera:
    def __init__(self, source=0):
        self.cap = open_camera_low_latency(source)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()
