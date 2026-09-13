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
from performance.lighting3d import apply_lighting_3d, LightParams, MaterialParams, draw_light_halo
from performance.shadows import ShadowRenderer
from performance.fast_normals import compute_normals_fast as _cnf_full
from performance.hand_light import (HandLightTracker, HandLightWorker,
                                    hand_state_to_light, draw_hand_panel)


# ============================================================
# CONFIG
# ============================================================
TARGET_FPS    = 15
IMGSZ         = 256
INFER_EVERY_N = 2       # MiDaS 1 frame sur 2 (dans le thread)
NORMALS_EVERY = 2       # normales 1 frame sur 2 (reutilise l'ancienne sinon)
PC_EVERY      = 2       # point cloud 1 frame sur 2
RELIGHT_EVERY = 2       # eclairage 1 frame sur 2
RELIGHT_SIZE  = (240, 180)   # (w,h) : plus fin qu'avant -> rendu plus net
EMA_ALPHA     = 0.4     # lissage temporel depth (anti-scintillement)
ENABLE_3D     = True
VIDEO_SOURCE  = 0
ENABLE_HAND   = True    # Level 3 : controle par la main (MediaPipe)
RECT_W, RECT_H = 300, 225   # 5 panneaux x 300 = 1500px (tient a l ecran)


def make_relit_panel(depth, frame_bgr, lights, halos, material,
                     relight_size, rect_w, rect_h,
                     shadow_renderer=None, light_colors=None):
    """
    Panneau eclaire par UNE OU PLUSIEURS lumieres (multi-light) + ombres.
    lights : liste de (x,y,z) ; halos : liste de (x,y) ecran.
    """
    rw, rh = relight_size
    d_small = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_AREA)
    d_small = cv2.bilateralFilter(d_small, 5, 0.1, 5)

    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    fxs = fys = float(rw)
    P = np.stack([(xx - rw/2) * d_small / fxs,
                  (yy - rh/2) * d_small / fys,
                  d_small], axis=-1)
    Nn = _cnf_full(d_small, scale=1)

    # Accumuler l'eclairage de TOUTES les lumieres EN COULEUR (garde la richesse)
    light_rgb = np.zeros((rh, rw, 3), dtype=np.float32)
    for i, (lx, ly, lz) in enumerate(lights):
        col = light_colors[i % len(light_colors)] if light_colors else (1.0, 0.93, 0.82)
        light = LightParams(position=(lx, ly, lz),
                            color=col, intensity=1.8, attenuation_radius=8.0)
        rs = apply_lighting_3d(P, Nn, light, view_pos=(0, 0, -1.5), material=material)
        light_rgb += rs.astype(np.float32) / 255.0
    if lights:
        light_rgb = np.clip(light_rgb, 0, 1.5)
    else:
        light_rgb = np.full((rh, rw, 3), 0.5, dtype=np.float32)   # pas de lumiere -> neutre

    cam_small = cv2.resize(frame_bgr, (rw, rh)).astype(np.float32) / 255.0
    colored = cam_small * (0.6 + 0.9 * light_rgb)   # couleur scene x lumiere coloree
    colored = np.clip((colored - 0.5) * 1.25 + 0.5, 0, 1)
    relit_small = (colored * 255).astype(np.uint8)

    relit_big = cv2.resize(relit_small, (rect_w, rect_h), interpolation=cv2.INTER_CUBIC)

    # OMBRES : une par lumiere, combinees par MOYENNE (somme douce).
    # - 1 seule lumiere  -> ombre identique au panneau souris.
    # - plusieurs lumieres -> les ombres s'additionnent progressivement :
    #   une zone vue par toutes les lumieres reste claire, une zone occultee
    #   par plusieurs devient plus sombre (somme des contributions d'ombre).
    if shadow_renderer is not None and lights:
        vis_accum = np.zeros((rect_h, rect_w), dtype=np.float32)
        for (lx, ly, lz) in lights:
            # position lumiere -> normalise [0,1] (x ecran = -lx a cause du miroir)
            lxn = (-lx / 2.0 + 0.5)
            lyn = (ly / 2.0 + 0.5)
            m = shadow_renderer.compute(depth, (lxn, lyn))
            m = cv2.resize(m, (rect_w, rect_h), interpolation=cv2.INTER_LINEAR)
            vis_accum += m
        # visibilite moyenne : chaque lumiere apporte sa part d'eclairage
        shadow_total = vis_accum / float(len(lights))
        relit_big = shadow_renderer.apply(relit_big, shadow_total)

    # Halos de toutes les lumieres (couleur par main)
    for i, ((hx, hy), (lx, ly, lz)) in enumerate(zip(halos, lights)):
        if light_colors:
            c = light_colors[i % len(light_colors)]
            halo_bgr = (int(c[2]*255), int(c[1]*255), int(c[0]*255))  # RGB->BGR
        else:
            halo_bgr = (210, 240, 255)
        relit_big = draw_light_halo(relit_big, hx, hy, lz, color_bgr=halo_bgr)
    return relit_big


def main():
    print("=" * 70)
    print("DEMO — Depth threadée + notre 3D / normales / interface")
    print("=" * 70)

    cap = ThreadedCamera(VIDEO_SOURCE)   # capture dans un thread -> ne bloque plus la boucle
    hud = HUD(target_fps=TARGET_FPS)
    pc_builder = PointCloudBuilder(downsample=5, output_size=(RECT_H, RECT_W))
    shadow_renderer = ShadowRenderer(intensity=0.82, softness=0.5, downsample=8, n_steps=12)

    # === Depth dans un thread dédié (le secret des 90 FPS) ===
    estimator = DepthEstimator(imgsz=IMGSZ, infer_every_n=INFER_EVERY_N, ema_alpha=EMA_ALPHA)
    worker = DepthWorker(estimator)
    worker.start()

    # === LEVEL 3 : hand tracking dans un thread dédié ===
    hand_worker = None
    if ENABLE_HAND:
        try:
            hand_tracker = HandLightTracker(max_hands=4, model_complexity=0)
            hand_worker = HandLightWorker(hand_tracker, process_every_n=1)
            hand_worker.start()
            print("Hand tracking ACTIF (bouge ta main pour deplacer la lumiere).")
        except Exception as e:
            print("Hand tracking desactive :", e)
            hand_worker = None
    last_hand_states = []

    WINDOW = "AI and VISION | Real-Time 3D"
    cv2.namedWindow(WINDOW)

    # === LEVEL 2 : lumiere CONTROLABLE (souris X/Y, molette Z) ===
    # light_state : [x, y, z] position de la lumiere dans la scene
    light_state = {"x": 0.0, "y": 0.0, "z": -0.8}   # position LUMIERE (pour l eclairage)
    halo_state = {"x": 0.0, "y": 0.0}                # position ECRAN du halo (sous la souris)

    def on_mouse(event, mx, my, flags, param):
        # Le panneau RELIT est le 5e (dernier). On mappe la souris sur toute la
        # largeur de la fenetre -> position X/Y de la lumiere dans [-1, 1].
        win_w = param[0]
        panel = mx % RECT_W if RECT_W else mx
        px = (panel / RECT_W - 0.5) * 2.0
        py = (my / RECT_H - 0.5) * 2.0
        halo_state["x"] = px          # halo SOUS la souris (non inverse)
        halo_state["y"] = py
        light_state["x"] = -px        # lumiere inversee (webcam miroir) pour l eclairage
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
    last_hand_relit = None
    # Palette : chaque main a sa couleur (Level 5 multi-light)
    HAND_COLORS = [(1.0, 0.85, 0.7), (0.6, 0.75, 1.0), (0.7, 1.0, 0.75), (1.0, 0.7, 0.85)]
    hand_lights = []   # liste de (x,y,z) : une lumiere par main detectee
    hand_halos = []    # liste de (x,y) : halos ecran correspondants
    material = MaterialParams(ambient=0.5, shininess=24.0, diffuse_color=(0.9, 0.85, 0.8))   # ambiant fort -> pas de zones noires
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

        # === LEVEL 3 : hand tracking -> pilote la lumiere ===
        if hand_worker is not None:
            hand_worker.push_frame(frame_rgb)
            hs = hand_worker.get_latest_states()
            if hs is not None:
                last_hand_states = hs   # peut etre [] si main absente -> souris reprend
            # TOUTES les mains -> une lumiere + un halo chacune (multi-light)
            hand_lights = []
            hand_halos = []
            for st in last_hand_states:
                hlight = hand_state_to_light(st)
                lx, ly, lz = hlight.position
                lz = float(np.clip(lz, -2.0, -0.2))
                hand_lights.append((lx, ly, lz))
                hand_halos.append((-lx, ly))   # halo en position ecran
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

        # === RELIT : eclaire par la SOURIS (1 lumiere) + ombre ===
        if last_relit is None or frame_id % RELIGHT_EVERY == 0:
            mouse_lights = [(light_state["x"], light_state["y"], light_state["z"])]
            mouse_halos = [(halo_state["x"], halo_state["y"])]
            last_relit = make_relit_panel(
                depth, frame_bgr, mouse_lights, mouse_halos,
                material, RELIGHT_SIZE, RECT_W, RECT_H,
                shadow_renderer=shadow_renderer)
        relit = last_relit

        # === HAND PANEL : ombre comme la souris (1 main) ou somme (N mains) ===
        if hand_worker is not None and (last_hand_relit is None or frame_id % RELIGHT_EVERY == 0):
            last_hand_relit = make_relit_panel(
                depth, frame_bgr, hand_lights, hand_halos,
                material, RELIGHT_SIZE, RECT_W, RECT_H,
                shadow_renderer=shadow_renderer, light_colors=HAND_COLORS)

        # === Composite : CAMERA | DEPTH | NORMALS | 3D ===
        a = cv2.resize(frame_bgr,  (RECT_W, RECT_H))
        b = cv2.resize(depth_vis,  (RECT_W, RECT_H))
        c = cv2.resize(normal_vis, (RECT_W, RECT_H))
        if ENABLE_3D and pc_2d is not None:
            d = pc_2d
            combined = np.hstack((a, b, c, d, relit))
            labels = ["CAMERA", "DEPTH", "NORMALS", "3D", "RELIT 3D"]
        else:
            combined = np.hstack((a, b, c))
            labels = ["CAMERA", "DEPTH", "NORMALS"]

        # === FPS (de la boucle d'affichage) ===
        dt = time.perf_counter() - t0
        fps = 1.0 / dt if dt > 0 else 0
        fps_hist.append(fps)
        info = {"fps": fps, "latency_ms": dt * 1000, "mode": "threaded", "imgsz": IMGSZ,
                "light_xyz": (light_state["x"], light_state["y"], light_state["z"])}

        combined = hud.draw(combined, info, labels=labels)

        # === PANNEAU DU BAS : scene eclairee PAR LA MAIN (sans squelette) ===
        if hand_worker is not None and last_hand_relit is not None:
            hand_panel = last_hand_relit.copy()
        else:
            hand_panel = (cv2.resize(frame_bgr, (RECT_W, RECT_H)) * 0.3).astype(np.uint8)
            cv2.putText(hand_panel, "hand tracking OFF", (12, RECT_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 220), 2, cv2.LINE_AA)
        cv2.putText(hand_panel, "HAND LIGHT", (12, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 40), 2, cv2.LINE_AA)
        row_w = combined.shape[1]
        bottom = np.zeros((RECT_H, row_w, 3), dtype=np.uint8)
        x0 = (row_w - RECT_W) // 2
        bottom[:, x0:x0 + RECT_W] = hand_panel
        combined = np.vstack((combined, bottom))

        # Securite : si l'image totale depasse 1600px de large, la reduire pour tenir a l'ecran
        MAX_W = 1600
        if combined.shape[1] > MAX_W:
            scale = MAX_W / combined.shape[1]
            combined = cv2.resize(combined, (MAX_W, int(combined.shape[0] * scale)))

        cv2.imshow(WINDOW, combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    worker.stop()
    if hand_worker is not None:
        hand_worker.stop()
    cap.release()
    cv2.destroyAllWindows()
    if fps_hist:
        print(f"\nFPS moyen (affichage) : {np.mean(fps_hist):.1f}  |  min {np.min(fps_hist):.1f}")


if __name__ == "__main__":
    main()
