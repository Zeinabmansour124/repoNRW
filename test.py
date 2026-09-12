import cv2
import numpy as np
from normalisation import DepthNormalizer, build_geometry_frame, normals_to_rgb
from depth_module import get_depth

normalizer = DepthNormalizer(alpha=0.15)
cap = cv2.VideoCapture(0)

# Taille d'UN rectangle à l'écran (chaque map aura exactement cette taille)
RECT_W, RECT_H = 426, 320     # 3 x 426 = 1278 de large au total

frame_id = 0
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    packet = get_depth(frame_rgb, frame_id=frame_id, imgsz=256)
    frame_id += 1

    h, w = packet["depth_map"].shape
    fx = fy = w
    cx, cy = w / 2, h / 2

    frame = build_geometry_frame(packet, fx, fy, cx, cy, normalizer, is_disparity=False)

    rgb = normals_to_rgb(frame.normal_map)
    normal_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_vis = cv2.cvtColor((packet["depth_map"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Forcer les 3 à EXACTEMENT le même rectangle
    a = cv2.resize(frame_bgr,  (RECT_W, RECT_H))
    b = cv2.resize(depth_vis,  (RECT_W, RECT_H))
    c = cv2.resize(normal_vis, (RECT_W, RECT_H))

    combined = np.hstack((a, b, c))
    cv2.imshow("Webcam | Depth | Normals", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()