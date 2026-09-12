
import time
import numpy as np

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



VIDEO_PATH = 0             
IMGSZ      = 384            # 384 (rapide), 512 (précis)
N_FRAMES   = 100            # Nombre de frames à benchmarker




def benchmark_depth(video_path=0, imgsz=384, n_frames=100):
    """Benchmark MiDaS-small en conditions réelles."""
    
    device = "cuda" if (_TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
    
    print("=" * 70)
    print(f"BENCHMARK MiDaS-small — JOUR J")
    print("=" * 70)
    print(f"  Device     : {device}")
    print(f"  imgsz      : {imgsz} (carré {imgsz}×{imgsz})")
    print(f"  n_frames   : {n_frames}")
    print(f"  video      : {video_path}")
    print("=" * 70)
    
    # === CHARGEMENT DU MODÈLE ===
    print("\n[LOAD] Chargement de MiDaS-small...")
    t0 = time.perf_counter()
    
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small",
                            trust_repo=True, verbose=False)
    midas.to(device)
    midas.eval()
    
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms",
                                       trust_repo=True, verbose=False)
    transform = midas_transforms.small_transform
    
    t1 = time.perf_counter()
    print(f"  [OK] Modèle chargé en {t1 - t0:.1f}s")
    
    # === SOURCE VIDÉO ===
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Impossible d'ouvrir la source : {video_path}")
        return None
    
    # === WARM-UP ===
    print("\n[WARM-UP] 3 inférences à vide...")
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(3):
        _ = _infer_midas(dummy, midas, transform, device)
    print("  [OK] Warm-up terminé")
    
    # === BENCHMARK ===
    times = {"capture": [], "preprocess": [],
             "depth_inference": [], "postprocess": []}
    
    t_start = time.perf_counter()
    frames_processed = 0
    
    for i in range(n_frames):
        # 1. CAPTURE
        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break
        t1 = time.perf_counter()
        
        # 2. PRÉ-TRAITEMENT
        frame_resized = cv2.resize(frame, (imgsz, imgsz))
        t2 = time.perf_counter()
        
        # 3. INFÉRENCE MiDaS
        depth = _infer_midas(frame_resized, midas, transform, device)
        t3 = time.perf_counter()
        
        # 4. POST-TRAITEMENT
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]))
        _ = depth.shape
        t4 = time.perf_counter()
        
        times["capture"].append((t1 - t0) * 1000)
        times["preprocess"].append((t2 - t1) * 1000)
        times["depth_inference"].append((t3 - t2) * 1000)
        times["postprocess"].append((t4 - t3) * 1000)
        frames_processed += 1
        
        # Affichage live
        if _CV2_AVAILABLE:
            cv2.imshow("MiDaS Depth", depth / depth.max())
            if cv2.waitKey(1) & 0xFF == 27:
                break
    
    cap.release()
    cv2.destroyAllWindows()
    
    t_end = time.perf_counter()
    total_time = t_end - t_start
    real_fps = frames_processed / total_time if total_time > 0 else 0
    
    # === AFFICHAGE RÉSULTATS ===
    print("\n" + "=" * 70)
    print("RÉSULTATS")
    print("=" * 70)
    print(f"{'Bloc':<18} | {'mean':>8} | {'p50':>8} | "
          f"{'p95':>8} | {'p99':>8}")
    print("─" * 70)
    
    total_mean = 0
    for name, values in times.items():
        if not values:
            continue
        arr = np.array(values)
        mean = float(np.mean(arr))
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        total_mean += mean
        print(f"{name:<18} | {mean:>6.2f}ms | {p50:>6.2f}ms | "
              f"{p95:>6.2f}ms | {p99:>6.2f}ms")
    
    print("─" * 70)
    print(f"{'TOTAL/frame':<18} | {total_mean:>6.2f}ms")
    print(f"\n📊 FPS réel : {real_fps:.1f}")
    print(f"📊 Frames   : {frames_processed}/{n_frames}")
    
    if times["depth_inference"]:
        depth_mean = np.mean(times["depth_inference"])
        pct = depth_mean / total_mean * 100 if total_mean > 0 else 0
        print(f"\n🎯 Goulot : depth_inference "
              f"({depth_mean:.1f} ms, {pct:.1f}%)")
    
    return {
        "device": device,
        "imgsz": imgsz,
        "real_fps": real_fps,
        "total_ms": total_mean,
    }


# ============================================================
# HELPER — Inférence MiDaS-small
# ============================================================

def _infer_midas(frame_rgb, midas, transform, device):
    """Inférence MiDaS-small sur une frame BGR."""
    # BGR → RGB
    frame_rgb = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
    
    # Transform
    input_batch = transform(frame_rgb).to(device)
    
    # Inférence
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    return prediction.cpu().numpy()



if __name__ == "__main__":
    result = benchmark_depth(
        video_path=VIDEO_PATH,
        imgsz=IMGSZ,
        n_frames=N_FRAMES,
    )