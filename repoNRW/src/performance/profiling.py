

import time
from contextlib import contextmanager
from typing import Dict, List

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False




VIDEO_PATH = 0              # 0 = webcam, ou "data/videos/day.mp4"
IMGSZ      = 384            # 384 (qualité) ou 256 (vitesse)
N_FRAMES   = 100            # Nombre de frames à profiler




class Profiler:
    """
    Mesure le temps passé dans chaque bloc du pipeline.
    
    Usage :
        profiler = Profiler()
        with profiler.block("depth"):
            depth = model(frame)
        profiler.report()
    """
    
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.durations: Dict[str, List[float]] = {}
        self.n_frames = 0
    
    @contextmanager
    def block(self, name: str):
        """Context manager pour mesurer un bloc."""
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            t1 = time.perf_counter()
            duration_ms = (t1 - t0) * 1000
            if name not in self.durations:
                self.durations[name] = []
            self.durations[name].append(duration_ms)
    
    def frame_done(self):
        if self.enabled:
            self.n_frames += 1
    
    def reset(self):
        self.durations = {}
        self.n_frames = 0
    
    # ------- STATS -------
    
    def _stats(self, values: List[float]) -> Dict[str, float]:
        if not values:
            return {"mean": 0, "p50": 0, "p95": 0, "p99": 0,
                    "min": 0, "max": 0, "n": 0}
        if _NUMPY_AVAILABLE:
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "n": len(values),
            }
        else:
            sorted_v = sorted(values)
            n = len(sorted_v)
            return {
                "mean": sum(values) / n,
                "p50": sorted_v[n // 2],
                "p95": sorted_v[int(n * 0.95)],
                "p99": sorted_v[int(n * 0.99)],
                "min": min(values),
                "max": max(values),
                "n": n,
            }
    
    def get_stats(self):
        return {name: self._stats(values)
                for name, values in self.durations.items()}
    
    def get_bottleneck(self) -> str:
        if not self.durations:
            return ""
        stats = self.get_stats()
        return max(stats.items(), key=lambda x: x[1]["mean"])[0]
    
    # ------- RAPPORT -------
    
    def report(self, title="Profiling Report"):
        print("\n" + "=" * 70)
        print(f"📊 {title} — {self.n_frames} frames")
        print("=" * 70)
        
        if not self.durations:
            print("Aucune donnée collectée.")
            return
        
        print(f"{'Bloc':<18} | {'mean':>8} | {'p50':>8} | "
              f"{'p95':>8} | {'p99':>8} | {'n':>5}")
        print("─" * 70)
        
        stats = self.get_stats()
        total_mean = sum(s["mean"] for s in stats.values())
        
        for name, s in stats.items():
            print(f"{name:<18} | {s['mean']:>6.2f}ms | {s['p50']:>6.2f}ms | "
                  f"{s['p95']:>6.2f}ms | {s['p99']:>6.2f}ms | {s['n']:>5}")
        
        print("─" * 70)
        total_p50 = sum(s["p50"] for s in stats.values())
        total_p95 = sum(s["p95"] for s in stats.values())
        total_p99 = sum(s["p99"] for s in stats.values())
        print(f"{'TOTAL/frame':<18} | {total_mean:>6.2f}ms | "
              f"{total_p50:>6.2f}ms | {total_p95:>6.2f}ms | "
              f"{total_p99:>6.2f}ms |")
        
        # Goulot
        bottleneck = self.get_bottleneck()
        if bottleneck and total_mean > 0:
            pct = stats[bottleneck]["mean"] / total_mean * 100
            print(f"\n🎯 Goulot : {bottleneck} "
                  f"({stats[bottleneck]['mean']:.1f} ms, {pct:.1f}%)")
        
        if total_mean > 0:
            print(f"📈 FPS estimé : {1000 / total_mean:.1f}")
        
        print("=" * 70)




def load_midas_small(device="cpu"):
    """Charge MiDaS-small via torch.hub."""
    print("[LOAD] Chargement MiDaS-small...")
    t0 = time.perf_counter()
    
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                            trust_repo=True, verbose=False)
    midas.to(device)
    midas.eval()
    
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                 trust_repo=True, verbose=False)
    transform = transforms.small_transform
    
    print(f"[OK] Modèle chargé en {time.perf_counter() - t0:.1f}s")
    return midas, transform


def infer_midas(frame_bgr, midas, transform, device="cpu"):
    """Inférence MiDaS-small sur une frame BGR."""
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(frame_rgb).to(device)
    
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    return prediction.cpu().numpy()



def profile_depth_pipeline(video_path=0, imgsz=384, n_frames=100):
    """Profile le pipeline depth complet."""
    
    device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    
    print("=" * 70)
    print(f"PROFILING — Pipeline Depth (MiDaS-small)")
    print("=" * 70)
    print(f"  Device     : {device}")
    print(f"  imgsz      : {imgsz}")
    print(f"  n_frames   : {n_frames}")
    print(f"  video      : {video_path}")
    print("=" * 70)
    
    # Charger le modèle
    midas, transform = load_midas_small(device)
    
    # Ouvrir la source vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" Impossible d'ouvrir {video_path}")
        return None
    
    # Warm-up
    print("\n[WARM-UP] 3 inférences à vide...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        _ = infer_midas(dummy, midas, transform, device)
    print("[OK] Warm-up terminé\n")
    
    # Profiler
    profiler = Profiler()
    
    for i in range(n_frames):
        # 1. CAPTURE
        with profiler.block("capture"):
            ret, frame = cap.read()
            if not ret:
                break
        
        # 2. PREPROCESS (resize)
        with profiler.block("preprocess"):
            frame_resized = cv2.resize(frame, (imgsz, imgsz))
        
        # 3. DEPTH INFERENCE
        with profiler.block("depth_inference"):
            depth = infer_midas(frame_resized, midas, transform, device)
        
        # 4. POSTPROCESS (resize depth)
        with profiler.block("postprocess"):
            depth_full = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        
        profiler.frame_done()
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Rapport
    profiler.report("Pipeline Depth")
    
    return profiler.get_stats()




if __name__ == "__main__":
    stats = profile_depth_pipeline(
        video_path=VIDEO_PATH,
        imgsz=IMGSZ,
        n_frames=N_FRAMES,
    )
    
    for imgsz in [384, 256, 192]:
         print(f"\n\n{'#' * 70}")
         print(f"# TEST imgsz = {imgsz}")
         print(f"{'#' * 70}")
         profile_depth_pipeline(video_path=VIDEO_PATH, imgsz=imgsz, n_frames=50)