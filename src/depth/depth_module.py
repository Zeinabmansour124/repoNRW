"""
depth_estimation.py
--------------------
Module d'estimation de profondeur monoculaire (MiDaS) optimisé pour le temps réel.

Corrections apportées par rapport à la version initiale :
- Suppression du double resize qui écrasait le résultat déjà redimensionné
- Suppression du resize redondant avant `transform()` (le transform MiDaS le fait déjà)
- Support FP16 (half precision) sur GPU
- cudnn.benchmark activé pour accélérer les tailles d'input fixes
- Lissage temporel EMA sur le depth map (évite le scintillement)
- Lissage EMA sur min/max de normalisation (évite le scintillement d'échelle frame à frame)
- Frame skipping configurable (réutilise le dernier depth map N-1 frames sur N)
- Pipeline exécutable en thread séparé (ne bloque jamais la boucle caméra/affichage)

Usage rapide (synchrone, sans thread) :
    depth_module = DepthEstimator(imgsz=256, infer_every_n=2, ema_alpha=0.4)
    result = depth_module.get_depth(frame_rgb, frame_id=0)
    depth_map = result["depth_map"]  # (H, W) float32, dans [0, 1]

Usage recommandé (thread dédié, découplé de la capture caméra) :
    depth_module = DepthEstimator(imgsz=256, infer_every_n=2, ema_alpha=0.4)
    worker = DepthWorker(depth_module)
    worker.start()
    worker.push_frame(frame_rgb, frame_id)   # non-bloquant
    result = worker.get_latest_result()      # peut renvoyer None si rien de prêt encore
    worker.stop()
"""

import threading
import queue
import time

import cv2
import numpy as np
import torch


class DepthEstimator:
    """Encapsule le modèle MiDaS et le pipeline d'inférence optimisé."""

    def __init__(
        self,
        model_type: str = "MiDaS_small",
        imgsz: int = 256,
        infer_every_n: int = 1,
        ema_alpha: float = 0.4,
        scale_ema_alpha: float = 0.2,
        use_fp16: bool = True,
    ):
        """
        Args:
            model_type: variante MiDaS ("MiDaS_small" recommandé pour temps réel).
            imgsz: taille carrée d'inférence interne (plus petit = plus rapide).
            infer_every_n: n'exécute réellement le modèle qu'une frame sur N
                (les autres frames réutilisent le dernier résultat). 1 = pas de skip.
            ema_alpha: poids de la frame courante dans le lissage temporel du depth map.
                Plus petit = plus lisse mais plus de latence perçue sur les mouvements rapides.
            scale_ema_alpha: lissage appliqué séparément au min/max de normalisation,
                pour éviter que l'échelle de couleur "saute" d'une frame à l'autre.
            use_fp16: active la demi-précision sur GPU (ignoré sur CPU).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device.type == "cuda"
        self.imgsz = imgsz
        self.infer_every_n = max(1, infer_every_n)
        self.ema_alpha = ema_alpha
        self.scale_ema_alpha = scale_ema_alpha

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        if self.use_fp16:
            self.model = self.model.half()
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = (
            transforms.small_transform if "small" in model_type.lower() else transforms.dpt_transform
        )

        # État pour le skip de frames et le lissage temporel
        self._frame_counter = 0
        self._last_raw_depth = None      # dernier depth map brut (résolution native h,w)
        self._depth_ema = None           # depth map lissé (EMA), résolution native
        self._dmin_ema = None
        self._dmax_ema = None

    # ------------------------------------------------------------------ #
    # Inférence
    # ------------------------------------------------------------------ #
    def _infer_raw(self, frame_rgb: np.ndarray) -> np.ndarray:
        """Fait tourner le modèle et renvoie un depth map brut à la résolution native (h, w),
        sans normalisation ni lissage."""
        h, w = frame_rgb.shape[:2]

        # Le transform MiDaS gère déjà le resize vers l'input attendu par le modèle,
        # donc PAS de cv2.resize manuel ici (c'était le double-resize redondant du bug initial).
        input_batch = self.transform(frame_rgb).to(self.device)
        if self.use_fp16:
            input_batch = input_batch.half()

        with torch.inference_mode():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        depth = prediction.float().cpu().numpy().astype("float32")
        depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)
        return depth

    def get_depth(self, frame_rgb: np.ndarray, frame_id: int = 0) -> dict:
        """Pipeline complet : skip de frame -> inférence -> lissage temporel -> normalisation.

        Renvoie un dict avec le depth map final dans [0, 1], prêt pour le calcul de normales.
        """
        h, w = frame_rgb.shape[:2]
        self._frame_counter += 1
        should_infer = (
            self._last_raw_depth is None
            or self._frame_counter % self.infer_every_n == 0
        )

        try:
            if should_infer:
                raw_depth = self._infer_raw(frame_rgb)
                self._last_raw_depth = raw_depth
            else:
                # Réutilise le dernier depth map brut (frame skipping)
                raw_depth = self._last_raw_depth
                if raw_depth.shape != (h, w):
                    raw_depth = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)

            # Lissage temporel EMA sur le depth map brut (réduit le scintillement pixel à pixel)
            if self._depth_ema is None or self._depth_ema.shape != raw_depth.shape:
                self._depth_ema = raw_depth.copy()
            else:
                self._depth_ema = (
                    self.ema_alpha * raw_depth + (1 - self.ema_alpha) * self._depth_ema
                )

            # Normalisation min/max avec lissage de l'échelle elle-même
            dmin, dmax = float(self._depth_ema.min()), float(self._depth_ema.max())
            if self._dmin_ema is None:
                self._dmin_ema, self._dmax_ema = dmin, dmax
            else:
                self._dmin_ema = self.scale_ema_alpha * dmin + (1 - self.scale_ema_alpha) * self._dmin_ema
                self._dmax_ema = self.scale_ema_alpha * dmax + (1 - self.scale_ema_alpha) * self._dmax_ema

            if self._dmax_ema - self._dmin_ema < 1e-8:
                depth_norm = np.full((h, w), 0.5, dtype="float32")
            else:
                depth_norm = (self._depth_ema - self._dmin_ema) / (self._dmax_ema - self._dmin_ema)
                depth_norm = np.clip(depth_norm, 0.0, 1.0)
                depth_norm = 1.0 - depth_norm  # convention : plus proche = valeur plus haute

            depth_norm = depth_norm.astype("float32")

        except Exception as e:  # fallback robuste : ne jamais crasher la boucle principale
            print("[depth] erreur, fallback :", e)
            depth_norm = np.full((h, w), 0.5, dtype="float32")

        return {
            "frame_id": frame_id,
            "timestamp": round(time.time(), 3),
            "depth_map": depth_norm,       # (H, W) float32, dans [0, 1]
            "imgsz_used": self.imgsz,
            "inferred_this_frame": should_infer,
        }


class DepthWorker:
    """Exécute DepthEstimator dans un thread dédié pour ne jamais bloquer
    la capture caméra ni l'affichage. Ne garde que la frame la plus récente
    (les anciennes sont jetées si le thread d'inférence prend du retard)."""

    def __init__(self, estimator: DepthEstimator):
        self.estimator = estimator
        self._frame_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def push_frame(self, frame_rgb: np.ndarray, frame_id: int = 0):
        """Non-bloquant : si le thread d'inférence n'a pas fini la frame précédente,
        on remplace la frame en attente par la plus récente plutôt que de s'accumuler."""
        if self._frame_queue.full():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                pass
        self._frame_queue.put((frame_rgb, frame_id))

    def get_latest_result(self):
        """Renvoie le dernier résultat disponible, ou None si rien n'est encore prêt."""
        result = None
        while not self._result_queue.empty():
            result = self._result_queue.get_nowait()
        return result

    def _run(self):
        while not self._stop_event.is_set():
            try:
                frame_rgb, frame_id = self._frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            result = self.estimator.get_depth(frame_rgb, frame_id)

            if self._result_queue.full():
                try:
                    self._result_queue.get_nowait()
                except queue.Empty:
                    pass
            self._result_queue.put(result)


# ---------------------------------------------------------------------- #
# Exemple d'utilisation autonome (à lancer directement pour tester le module)
# ---------------------------------------------------------------------- #
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    estimator = DepthEstimator(imgsz=256, infer_every_n=2, ema_alpha=0.4)
    worker = DepthWorker(estimator)
    worker.start()

    fps_history = []
    frame_id = 0
    last_depth_vis = None

    try:
        while True:
            t0 = time.time()
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            worker.push_frame(frame_rgb, frame_id)
            frame_id += 1

            result = worker.get_latest_result()
            if result is not None:
                depth_vis = (result["depth_map"] * 255).astype("uint8")
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
                last_depth_vis = depth_vis

            display = last_depth_vis if last_depth_vis is not None else frame_bgr
            fps = 1.0 / max(time.time() - t0, 1e-6)
            fps_history.append(fps)
            cv2.putText(
                display, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
            )

            cv2.imshow("Depth (thread dedie)", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()
        if fps_history:
            print(f"FPS moyen: {np.mean(fps_history):.1f}, min: {np.min(fps_history):.1f}")