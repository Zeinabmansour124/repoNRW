import time
import cv2
import numpy as np
import torch

# ---- Chargement du modèle (une seule fois, au démarrage) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# ---- Constantes du contrat ----
OUT_W, OUT_H = 640, 480      # résolution verrouillée (largeur, hauteur)


def get_depth(frame_rgb, frame_id=0, imgsz=256):
    try:
        img_resized = cv2.resize(frame_rgb, (imgsz, imgsz))
        input_batch = transform(img_resized).to(device)
        h, w = frame_rgb.shape[:2]   # taille native

        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),   # garder la taille originale
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy().astype("float32")
        depth = np.nan_to_num(depth, nan=0.5, posinf=1.0, neginf=0.0)

        dmin, dmax = depth.min(), depth.max()
        if dmax - dmin < 1e-8:
            depth = np.full((h, w), 0.5, dtype="float32")
        else:
            depth = (depth - dmin) / (dmax - dmin)
            depth = 1.0 - depth

        depth = depth.astype("float32")

    except Exception as e:
        print("[depth] erreur, fallback :", e)
        depth = np.full((h, w), 0.5, dtype="float32")

    return {
        "frame_id": frame_id,
        "timestamp": round(time.time(), 3),
        "depth_map": depth,          # (H, W) float32, 0..1
        "imgsz_used": imgsz,
    }
