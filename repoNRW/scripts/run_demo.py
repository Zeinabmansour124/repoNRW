# scripts/run_demo.py
"""
DEMO — Pipeline temps réel avec depth THREADÉE + OMBRES DYNAMIQUES.
"""

import sys
import os
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from performance.fast_normals import compute_normals_fast, normals_to_rgb_fast
from depth.depth_module import DepthEstimator, DepthWorker

from performance.hud import HUD
from performance.latency import open_camera_low_latency, ThreadedCamera
from performance.point_cloud import PointCloudBuilder
from performance.lighting3d import apply_lighting_3d, LightParams, MaterialParams, draw_light_halo
from performance.fast_normals import compute_normals_fast as _cnf_full

# ⭐ OMBRES
from performance.shadows import ShadowRenderer

# from performance.hand_tracking import HandTracker   # ← désactivé si pas prêt

# ============================================================
# CONFIG
# ============================================================
TARGET_FPS    = 15
IMGSZ         = 256
INFER_EVERY_N = 2
NORMALS_EVERY = 2
PC_EVERY      = 2
RELIGHT_EVERY = 2
RELIGHT_SIZE  = (240, 180)
EMA_ALPHA     = 0.4
ENABLE_3D     = True

# ⭐ CONFIG OMBRES
ENABLE_SHADOWS    = True
SHADOWS_EVERY     = 2        # calculer les ombres 1 frame sur 2
SHADOW_INTENSITY  = 0.6      # 0 = pas d'ombre, 1 = ombre noire
SHADOW_SOFTNESS   = 0.5      # 0 = ombre dure, 1 = ombre douce
SHADOW_DOWNSAMPLE = 4        # résolution de calcul (plus grand = plus rapide)

VIDEO_SOURCE  = 0
RECT_W, RECT_H = 300, 225


def main():
    print("=" * 70)
    print("DEMO — Depth threadée + 3D / normales + OMBRES DYNAMIQUES")
    print("=" * 70)

    cap = ThreadedCamera(VIDEO_SOURCE)
    hud = HUD(target_fps=TARGET_FPS)
    pc_builder = PointCloudBuilder(downsample=5, output_size=(RECT_H, RECT_W))

    # ⭐ SHADOWS
    shadow_renderer = ShadowRenderer(
        intensity=SHADOW_INTENSITY,
        softness=SHADOW_SOFTNESS,
        downsample=SHADOW_DOWNSAMPLE,
    )
    last_shadow_mask = None

    # === Depth dans un thread dédié ===
    estimator = DepthEstimator(imgsz=IMGSZ, infer_every_n=INFER_EVERY_N, ema_alpha=EMA_ALPHA)
    worker = DepthWorker(estimator)
    worker.start()

    WINDOW = "AI and VISION | Real-Time 3D"
    cv2.namedWindow(WINDOW)

    # === LEVEL 2 : lumière contrôlable (souris X/Y, molette Z) ===
    light_state = {"x": 0.0, "y": 0.0, "z": -0.8}
    halo_state = {"x": 0.0, "y": 0.0}

    def on_mouse(event, mx, my, flags, param):
        win_w = param[0]
        panel = mx % RECT_W if RECT_W else mx
        px = (panel / RECT_W - 0.5) * 2.0
        py = (my / RECT_H - 0.5) * 2.0
        halo_state["x"] = px
        halo_state["y"] = py
        light_state["x"] = -px
        light_state["y"] = py
        if event == cv2.EVENT_MOUSEWHEEL:
            dz = 0.1 if flags > 0 else -0.1
            light_state["z"] = float(np.clip(light_state["z"] + dz, -2.0, -0.2))

    cv2.setMouseCallback(WINDOW, on_mouse, param=[RECT_W * 5])

    frame_id = 0
    fps_hist = []
    last_result = None
    last_normal_vis = None
    last_pc_2d = None
    last_relit = None
    material = MaterialParams(ambient=0.25, shininess=32.0, diffuse_color=(0.9, 0.85, 0.8))
    print("\nDémarrage. Appuie sur 'q' pour quitter.\n")

    while True:
        t0 = time.perf_counter()

        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_bgr = cv2.resize(frame_bgr, (640, 480))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        worker.push_frame(frame_rgb, frame_id)
        result = worker.get_latest_result()
        if result is not None:
            last_result = result
        frame_id += 1

        if last_result is None:
            cv2.imshow(WINDOW, frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        packet = last_result
        depth = packet["depth_map"]

        # S'assurer que depth est 2D
        depth = np.squeeze(depth)
        if depth.ndim != 2:
            depth = depth.reshape(depth.shape[-2:])
        depth = depth.astype(np.float32)

        depth_vis = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # === Normales ===
        if last_normal_vis is None or frame_id % NORMALS_EVERY == 0:
            normals = compute_normals_fast(depth, scale=2)
            rgb = normals_to_rgb_fast(normals)
            last_normal_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        normal_vis = last_normal_vis

        # === 3D ===
        if ENABLE_3D and (last_pc_2d is None or frame_id % PC_EVERY == 0):
            points_3d, colors_3d = pc_builder.build(depth, rgb=frame_bgr)
            last_pc_2d = pc_builder.render_2d(points_3d, colors_3d)
        pc_2d = last_pc_2d

        # ⭐ ============================================================
        # ⭐ OMBRES DYNAMIQUES — SEUL AJOUT
        # ⭐ ============================================================
        if ENABLE_SHADOWS and (last_shadow_mask is None
                                or frame_id % SHADOWS_EVERY == 0):
            # Convertir la position lumière [-1,1] → [0,1] pour le renderer
            lx_norm = (halo_state["x"] + 1.0) / 2.0
            ly_norm = (halo_state["y"] + 1.0) / 2.0
            last_shadow_mask = shadow_renderer.compute(
                depth, (lx_norm, ly_norm)
            )

        # Appliquer les ombres sur la frame caméra
        if ENABLE_SHADOWS and last_shadow_mask is not None:
            camera_vis = shadow_renderer.apply(frame_bgr, last_shadow_mask)
        else:
            camera_vis = frame_bgr
        # ⭐ ============================================================
        # ⭐ FIN AJOUT OMBRES
        # ⭐ ============================================================

        # === RELIT ===
        if last_relit is None or frame_id % RELIGHT_EVERY == 0:
            rw, rh = RELIGHT_SIZE
            d_small = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_AREA)
            d_small = cv2.bilateralFilter(d_small, 5, 0.1, 5)

            yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
            fxs = fys = float(rw)
            P = np.stack([(xx - rw/2) * d_small / fxs,
                          (yy - rh/2) * d_small / fys,
                          d_small], axis=-1)
            Nn = _cnf_full(d_small, scale=1)

            light = LightParams(
                position=(light_state["x"], light_state["y"], light_state["z"]),
                color=(1.0, 0.93, 0.82),
                intensity=1.8,
                attenuation_radius=2.5,
            )
            relit_small = apply_lighting_3d(P, Nn, light, view_pos=(0, 0, -1.5),
                                            material=material)

            cam_small = cv2.resize(frame_bgr, (rw, rh)).astype(np.float32) / 255.0
            intensity = relit_small.astype(np.float32) / 255.0
            colored = cam_small * (0.35 + 1.5 * intensity)
            colored = np.clip((colored - 0.5) * 1.25 + 0.5, 0, 1)
            relit_small = (colored * 255).astype(np.uint8)

            relit_big = cv2.resize(relit_small, (RECT_W, RECT_H), interpolation=cv2.INTER_CUBIC)
            last_relit = draw_light_halo(relit_big,
                                         halo_state["x"], halo_state["y"], light_state["z"])
        relit = last_relit

        # === Composite : CAMERA | DEPTH | NORMALS | 3D | RELIT ===
        # ⭐ CHANGEMENT : utiliser camera_vis (avec ombres) au lieu de frame_bgr
        a = cv2.resize(camera_vis,  (RECT_W, RECT_H))
        b = cv2.resize(depth_vis,  (RECT_W, RECT_H))
        c = cv2.resize(normal_vis, (RECT_W, RECT_H))
        if ENABLE_3D and pc_2d is not None:
            d = pc_2d
            combined = np.hstack((a, b, c, d, relit))
            labels = ["CAMERA", "DEPTH", "NORMALS", "3D", "RELIT 3D"]
        else:
            combined = np.hstack((a, b, c))
            labels = ["CAMERA", "DEPTH", "NORMALS"]

        # === FPS ===
        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0
        fps_hist.append(fps)
        info = {"fps": fps, "latency_ms": dt * 1000, "mode": "threaded", "imgsz": IMGSZ,
                "light_xyz": (light_state["x"], light_state["y"], light_state["z"])}

        combined = hud.draw(combined, info, labels=labels)

        MAX_W = 1600
        if combined.shape[1] > MAX_W:
            scale = MAX_W / combined.shape[1]
            combined = cv2.resize(combined, (MAX_W, int(combined.shape[0] * scale)))

        cv2.imshow(WINDOW, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    if fps_hist:
        print(f"\nFPS moyen (affichage) : {np.mean(fps_hist):.1f}  |  min {np.min(fps_hist):.1f}")
    
    # ⭐ Stats ombres
    if ENABLE_SHADOWS:
        print("\n=== Shadows ===")
        for k, v in shadow_renderer.get_stats().items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()