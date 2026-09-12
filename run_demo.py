# scripts/run_demo.py
"""
DEMO — Pipeline temps réel avec depth THREADÉE (worker dédié, ~90 FPS d'affichage).
Garde NOTRE 3D, nos normales, notre interface.
"""

import sys
import os
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from performance.fast_normals import compute_normals_fast, normals_to_rgb_fast
# Depth threadée (technique de l'amie : MiDaS dans un thread séparé -> l'affichage n'attend jamais)
from depth.depth_module import DepthEstimator, DepthWorker

from performance.hud import HUD
from performance.latency import open_camera_low_latency, ThreadedCamera
from performance.point_cloud import PointCloudBuilder


# ============================================================
# CONFIG
# ============================================================
TARGET_FPS    = 15
IMGSZ         = 256
INFER_EVERY_N = 2       # MiDaS 1 frame sur 2 (dans le thread)
NORMALS_EVERY = 2       # normales 1 frame sur 2 (reutilise l'ancienne sinon)
PC_EVERY      = 2       # point cloud 1 frame sur 2
EMA_ALPHA     = 0.4     # lissage temporel depth (anti-scintillement)
ENABLE_3D     = True
VIDEO_SOURCE  = 0
RECT_W, RECT_H = 426, 320


def main():
    print("=" * 70)
    print("DEMO — Depth threadée + notre 3D / normales / interface")
    print("=" * 70)

    cap = ThreadedCamera(VIDEO_SOURCE)   # capture dans un thread -> ne bloque plus la boucle
    hud = HUD(target_fps=TARGET_FPS)
    pc_builder = PointCloudBuilder(downsample=5, output_size=(RECT_H, RECT_W))

    # === Depth dans un thread dédié (le secret des 90 FPS) ===
    estimator = DepthEstimator(imgsz=IMGSZ, infer_every_n=INFER_EVERY_N, ema_alpha=EMA_ALPHA)
    worker = DepthWorker(estimator)
    worker.start()

    WINDOW = "AI and VISION | Real-Time 3D"
    cv2.namedWindow(WINDOW)

    frame_id = 0
    fps_hist = []
    last_result = None
    last_normal_vis = None
    last_pc_2d = None
    print("\nDémarrage. Appuie sur 'q' pour quitter.\n")

    while True:
        t0 = time.perf_counter()

        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_bgr = cv2.resize(frame_bgr, (640, 480))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Pousser la frame au thread depth (NON bloquant) et récupérer le dernier résultat prêt
        worker.push_frame(frame_rgb, frame_id)
        result = worker.get_latest_result()
        if result is not None:
            last_result = result
        frame_id += 1

        # Tant qu'aucune depth n'est prête, afficher juste la caméra
        if last_result is None:
            cv2.imshow(WINDOW, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        packet = last_result
        depth = packet["depth_map"]

        depth_vis = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # === Normales RAPIDES, 1 frame sur NORMALS_EVERY ===
        if last_normal_vis is None or frame_id % NORMALS_EVERY == 0:
            normals = compute_normals_fast(depth, scale=2)
            rgb = normals_to_rgb_fast(normals)
            last_normal_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        normal_vis = last_normal_vis

        # === Notre 3D, 1 frame sur PC_EVERY ===
        if ENABLE_3D and (last_pc_2d is None or frame_id % PC_EVERY == 0):
            points_3d, colors_3d = pc_builder.build(depth, rgb=frame_bgr)
            last_pc_2d = pc_builder.render_2d(points_3d, colors_3d)
        pc_2d = last_pc_2d

        # === Composite : CAMERA | DEPTH | NORMALS | 3D ===
        a = cv2.resize(frame_bgr,  (RECT_W, RECT_H))
        b = cv2.resize(depth_vis,  (RECT_W, RECT_H))
        c = cv2.resize(normal_vis, (RECT_W, RECT_H))
        if ENABLE_3D and pc_2d is not None:
            d = pc_2d
            combined = np.hstack((a, b, c, d))
            labels = ["CAMERA", "DEPTH", "NORMALS", "3D"]
        else:
            combined = np.hstack((a, b, c))
            labels = ["CAMERA", "DEPTH", "NORMALS"]

        # === FPS (de la boucle d'affichage) ===
        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0
        fps_hist.append(fps)
        info = {"fps": fps, "latency_ms": dt * 1000, "mode": "threaded", "imgsz": IMGSZ}

        combined = hud.draw(combined, info, labels=labels)
        cv2.imshow(WINDOW, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    if fps_hist:
        print(f"\nFPS moyen (affichage) : {np.mean(fps_hist):.1f}  |  min {np.min(fps_hist):.1f}")


if __name__ == "__main__":
    main()
