import cv2
import numpy as np
import torch
import time
from normalisation import DepthNormalizer, build_geometry_frame, normals_to_rgb
from depth_module import get_depth   # <-- ton fichier avec la fonction Personne 1

# Initialiser le normalizer
normalizer = DepthNormalizer(alpha=0.15)

# Ouvrir la webcam
cap = cv2.VideoCapture(0)  # 0 = webcam par défaut

frame_id = 0
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    # Convertir en RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Récupérer la depth map via MiDaS
    packet = get_depth(frame_rgb, frame_id=frame_id, imgsz=256)
    frame_id += 1

    # Dimensions
    h, w = packet["depth_map"].shape
    fx = fy = w
    cx, cy = w / 2, h / 2

    # Construire le GeometryFrame
    frame = build_geometry_frame(packet, fx, fy, cx, cy, normalizer, is_disparity=False)

    # Convertir la normal map en RGB
    rgb = normals_to_rgb(frame.normal_map)

    # Afficher côte à côte : image originale, depth map, normal map
    depth_vis = (packet["depth_map"] * 255).astype(np.uint8)
    depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

    normal_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    combined = np.hstack((frame_bgr, depth_vis, normal_vis))
    cv2.imshow("Webcam + Depth + Normals", combined)

    # Quitter avec 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
