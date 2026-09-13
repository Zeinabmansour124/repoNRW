# webapp/pipeline.py
"""
Pipeline temps réel SANS fenêtre OpenCV, pour l'interface web.

Reprend exactement la chaîne de scripts/run_demo.py :
    CAMERA -> DEPTH (MiDaS, threadé) -> NORMALES -> 3D -> RELIGHT + OMBRES + MAIN
mais au lieu d'afficher avec cv2.imshow, il :
    * pousse la dernière frame composite (JPEG) dans un buffer partagé,
    * calcule le masque d'ombre piloté par la main,
    * appelle GameController.update()/render() quand un niveau est en cours.

Le serveur Flask lit ce buffer pour le stream MJPEG et interroge le
GameController pour l'état JSON.
"""

import os
import sys
import threading
import time

import cv2
import numpy as np

# rendre src/ importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for sub in ("src", "scripts"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from performance.fast_normals import compute_normals_fast, normals_to_rgb_fast
from depth.depth_module import DepthEstimator, DepthWorker
from performance.latency import ThreadedCamera
from performance.point_cloud import PointCloudBuilder
from performance.lighting3d import MaterialParams
from performance.shadows import ShadowRenderer
from performance.hand_light import (HandLightTracker, HandLightWorker,
                                    hand_state_to_light)

# on réutilise la fonction de rendu du relight telle quelle
from run_demo import make_relit_panel


# ------------------------------------------------------------------ CONFIG
IMGSZ         = 256
INFER_EVERY_N = 2
NORMALS_EVERY = 2
PC_EVERY      = 2
RELIGHT_EVERY = 2
RELIGHT_SIZE  = (240, 180)
EMA_ALPHA     = 0.4
VIDEO_SOURCE  = 0
RECT_W, RECT_H = 300, 225
HAND_COLORS = [(1.0, 0.85, 0.7), (0.6, 0.75, 1.0), (0.7, 1.0, 0.75), (1.0, 0.7, 0.85)]


class VisionPipeline:
    """Tourne dans un thread. Expose la dernière frame JPEG et le masque d'ombre."""

    def __init__(self, controller):
        self.controller = controller           # GameController
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

        self._jpeg_lock = threading.Lock()
        self._jpeg = None                       # bytes du dernier composite encodé
        self.ready = False
        self.error = None
        self.fps = 0.0

    # -------------------------------------------------- API publique
    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def get_jpeg(self):
        with self._jpeg_lock:
            return self._jpeg

    def _set_jpeg(self, frame_bgr):
        ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with self._jpeg_lock:
                self._jpeg = buf.tobytes()

    # -------------------------------------------------- boucle
    def _run(self):
        try:
            self._loop()
        except Exception as e:            # on remonte l'erreur à l'UI plutôt que de crasher en silence
            import traceback
            self.error = f"{e}"
            traceback.print_exc()

    def _loop(self):
        cap = ThreadedCamera(VIDEO_SOURCE)
        pc_builder = PointCloudBuilder(downsample=5, output_size=(RECT_H, RECT_W))
        shadow_renderer = ShadowRenderer(intensity=0.82, softness=0.5, downsample=8, n_steps=12)

        estimator = DepthEstimator(imgsz=IMGSZ, infer_every_n=INFER_EVERY_N, ema_alpha=EMA_ALPHA)
        worker = DepthWorker(estimator)
        worker.start()

        hand_worker = None
        try:
            tracker = HandLightTracker(max_hands=4, model_complexity=0)
            hand_worker = HandLightWorker(tracker, process_every_n=1)
            hand_worker.start()
        except Exception as e:
            print("[pipeline] hand tracking OFF:", e)
            hand_worker = None

        material = MaterialParams(ambient=0.5, shininess=24.0, diffuse_color=(0.9, 0.85, 0.8))

        last_result = last_normal_vis = last_pc_2d = None
        last_hand_relit = None
        last_hand_shadow = None
        last_hand_states = []
        frame_id = 0

        while not self._stop.is_set():
            t0 = time.perf_counter()
            ret, frame_bgr = cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            frame_bgr = cv2.resize(frame_bgr, (640, 480))
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            worker.push_frame(frame_rgb, frame_id)
            r = worker.get_latest_result()
            if r is not None:
                last_result = r

            # main -> lumières
            hand_lights, hand_halos = [], []
            if hand_worker is not None:
                hand_worker.push_frame(frame_rgb)
                hs = hand_worker.get_latest_states()
                if hs is not None:
                    last_hand_states = hs
                for st in last_hand_states:
                    hl = hand_state_to_light(st)
                    lx, ly, lz = hl.position
                    lz = float(np.clip(lz, -2.0, -0.2))
                    hand_lights.append((lx, ly, lz))
                    hand_halos.append((-lx, ly))
            frame_id += 1

            if last_result is None:
                # pas encore de depth : on montre juste la caméra
                self._set_jpeg(frame_bgr)
                self.ready = True
                continue

            depth = last_result["depth_map"]
            depth_vis = cv2.cvtColor((depth * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

            if last_normal_vis is None or frame_id % NORMALS_EVERY == 0:
                normals = compute_normals_fast(depth, scale=2)
                rgb = normals_to_rgb_fast(normals)
                last_normal_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            if last_pc_2d is None or frame_id % PC_EVERY == 0:
                pts, cols = pc_builder.build(depth, rgb=frame_bgr)
                last_pc_2d = pc_builder.render_2d(pts, cols)

            # panneau éclairé par la MAIN (+ ombre) -> c'est LUI le terrain de jeu
            if hand_worker is not None and (last_hand_relit is None or frame_id % RELIGHT_EVERY == 0):
                last_hand_relit, last_hand_shadow = make_relit_panel(
                    depth, frame_bgr, hand_lights, hand_halos,
                    material, RELIGHT_SIZE, RECT_W, RECT_H,
                    shadow_renderer=shadow_renderer, light_colors=HAND_COLORS)

            # ---- jeu ----
            sm = None
            if last_hand_shadow is not None:
                sm = cv2.resize(last_hand_shadow, (RECT_W, RECT_H), interpolation=cv2.INTER_LINEAR)
            self.controller.update(sm)

            if last_hand_relit is not None:
                game_panel = last_hand_relit.copy()
            else:
                game_panel = (cv2.resize(frame_bgr, (RECT_W, RECT_H)) * 0.3).astype(np.uint8)
                cv2.putText(game_panel, "montre ta main", (12, RECT_H // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 220), 2, cv2.LINE_AA)

            game_panel = self.controller.render(game_panel, sm)

            # composite: on met le panneau de jeu en GRAND + une bande de vignettes
            # (depth / normales / 3d) comme dans la maquette du HUD
            big = cv2.resize(game_panel, (RECT_W * 2, RECT_H * 2), interpolation=cv2.INTER_CUBIC)

            thumbs = []
            for img, name in ((depth_vis, "DEPTH"),
                              (last_normal_vis, "NORMALES"),
                              (last_pc_2d, "3D")):
                t = cv2.resize(img, (RECT_W, RECT_H))
                cv2.putText(t, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 200), 1, cv2.LINE_AA)
                thumbs.append(t)
            side = np.vstack(thumbs[:2] + [thumbs[2]])
            side = cv2.resize(side, (RECT_W, RECT_H * 2))
            composite = np.hstack((big, side))

            self._set_jpeg(composite)
            self.ready = True

            dt = time.perf_counter() - t0
            self.fps = 1.0 / dt if dt > 0 else 0.0

        worker.stop()
        if hand_worker is not None:
            hand_worker.stop()
        cap.release()
