# src/performance/hand_light.py
"""
Level 3 — Gesture-controlled Lighting (MediaPipe Hands, threade).
Supporte les DEUX API mediapipe : 'solutions' (0.10.x classique) et 'tasks' (recent).
"""

import threading
import queue
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    import mediapipe as mp
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    _MEDIAPIPE_AVAILABLE = False

from performance.lighting3d import LightParams


_PALM_LANDMARKS = (0, 5, 9, 13, 17)
_THUMB_TIP = 4
_INDEX_TIP = 8


@dataclass
class HandLightState:
    hand_index: int
    light_xy: Tuple[float, float]
    depth_proxy: float
    pinch: float
    landmarks_px: np.ndarray


class HandLightTracker:
    def __init__(self, max_hands: int = 2, detection_conf: float = 0.6,
                 tracking_conf: float = 0.6, model_complexity: int = 0):
        if not _MEDIAPIPE_AVAILABLE:
            raise ImportError("mediapipe n'est pas installe.")
        self._max_hands = max_hands

        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self._api = "solutions"
            self._mp_hands = mp.solutions.hands
            self.hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=max_hands,
                model_complexity=model_complexity,
                min_detection_confidence=detection_conf,
                min_tracking_confidence=tracking_conf,
            )
        elif hasattr(mp, "tasks"):
            self._api = "tasks"
            self._init_tasks_api(max_hands, detection_conf, tracking_conf)
        else:
            raise ImportError("Version de mediapipe incompatible (ni solutions ni tasks).")

    def _init_tasks_api(self, max_hands, detection_conf, tracking_conf):
        """Nouvelle API (tasks). Telecharge le modele .task si absent."""
        import os
        import urllib.request

        model_path = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
        if not os.path.exists(model_path):
            url = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            print("[hand] telechargement du modele hand_landmarker.task ...")
            urllib.request.urlretrieve(url, model_path)
            print("[hand] modele telecharge.")

        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )
        self.landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame_rgb: np.ndarray) -> List[HandLightState]:
        h, w = frame_rgb.shape[:2]

        # Liste de mains, chaque main = liste de (x,y) normalises [0,1]
        hands_norm = []
        if self._api == "solutions":
            frame_rgb.flags.writeable = False
            result = self.hands.process(frame_rgb)
            frame_rgb.flags.writeable = True
            if result.multi_hand_landmarks:
                for hl in result.multi_hand_landmarks:
                    hands_norm.append([(lm.x, lm.y) for lm in hl.landmark])
        else:  # tasks
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=np.ascontiguousarray(frame_rgb))
            result = self.landmarker.detect(mp_image)
            if result.hand_landmarks:
                for hl in result.hand_landmarks:
                    hands_norm.append([(lm.x, lm.y) for lm in hl])

        states: List[HandLightState] = []
        if not hands_norm:
            return states

        for idx, hand_pts in enumerate(hands_norm):
            pts = np.array(hand_pts, dtype=np.float32)
            pts_px = pts * np.array([w, h], dtype=np.float32)

            palm = pts[list(_PALM_LANDMARKS)].mean(axis=0)

            light_x = -(palm[0] - 0.5) * 2.0
            light_y = (palm[1] - 0.5) * 2.0

            bbox_w = pts_px[:, 0].max() - pts_px[:, 0].min()
            bbox_h = pts_px[:, 1].max() - pts_px[:, 1].min()
            diag = float(np.hypot(bbox_w, bbox_h))
            img_diag = float(np.hypot(w, h))
            depth_proxy = float(np.clip(diag / (0.6 * img_diag), 0.0, 1.0))

            thumb = pts[_THUMB_TIP]
            index = pts[_INDEX_TIP]
            pinch_dist = float(np.linalg.norm(thumb - index))
            hand_scale = max(diag / img_diag, 1e-3)
            pinch = float(np.clip(1.0 - (pinch_dist / (0.35 * hand_scale)), 0.0, 1.0))

            states.append(HandLightState(
                hand_index=idx,
                light_xy=(float(light_x), float(light_y)),
                depth_proxy=depth_proxy,
                pinch=pinch,
                landmarks_px=pts_px,
            ))
        return states

    def close(self):
        if self._api == "solutions":
            self.hands.close()
        else:
            self.landmarker.close()


def hand_state_to_light(state: HandLightState,
                        z_near: float = -0.4, z_far: float = -1.2,
                        base_intensity: float = 1.8, pinch_boost: float = 1.5,
                        color: Tuple[float, float, float] = (1.0, 0.95, 0.85),
                        attenuation_radius: float = 8.0) -> LightParams:
    """
    Convertit l'etat de la main en lumiere.
    - Z reste dans une plage PROCHE (-0.4 a -1.2) pour que la lumiere
      atteigne vraiment la scene (avant: -2.5 = trop loin, n'eclairait rien).
    - attenuation_radius large (8.0, comme le panneau souris) pour que la
      lumiere illumine les objets au lieu de s'eteindre trop vite.
    """
    x, y = state.light_xy
    z = z_far + (z_near - z_far) * state.depth_proxy
    intensity = base_intensity + pinch_boost * state.pinch
    return LightParams(
        position=(x, y, z),
        color=color,
        intensity=intensity,
        attenuation=True,
        attenuation_radius=attenuation_radius,
    )


class HandLightWorker:
    def __init__(self, tracker: HandLightTracker, process_every_n: int = 1):
        self.tracker = tracker
        self.process_every_n = max(1, process_every_n)
        self._frame_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._counter = 0

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self.tracker.close()

    def push_frame(self, frame_rgb: np.ndarray):
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put(frame_rgb)

    def get_latest_states(self) -> Optional[List[HandLightState]]:
        result = None
        while not self._result_queue.empty():
            result = self._result_queue.get_nowait()
        return result

    def _run(self):
        while not self._stop_event.is_set():
            try:
                frame_rgb = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self._counter += 1
            if self._counter % self.process_every_n != 0:
                continue

            try:
                states = self.tracker.process(frame_rgb)
            except Exception as e:
                print("[hand_light] erreur tracking, ignoree :", e)
                states = []

            if self._result_queue.full():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
            self._result_queue.put(states)


# ------------------------------------------------------------------ #
# Overlay : dessiner la main detectee (squelette) pour le panneau debug
# ------------------------------------------------------------------ #

_HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


def draw_hand_panel(frame_bgr, states, size=(300, 225)):
    """Affiche la vue camera propre (SANS dessiner le squelette de la main).
    La detection tourne quand meme (elle pilote la lumiere), on ne l'overlay pas."""
    import cv2
    w, h = size
    canvas = cv2.resize(frame_bgr, (w, h))

    # Petit indicateur discret : main detectee ou non (sans dessiner dessus)
    if states:
        st = states[0]
        cv2.putText(canvas, f"main detectee  Z {st.depth_proxy:.2f}", (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 255, 120), 2, cv2.LINE_AA)
    else:
        cv2.putText(canvas, "aucune main", (12, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 2, cv2.LINE_AA)

    return canvas
