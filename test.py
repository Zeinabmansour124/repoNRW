import cv2
import numpy as np
from normalisation import DepthNormalizer, build_geometry_frame, normals_to_rgb

def test_depth_image(image_path="image.png"):
    # Charger la depth map (grayscale)
    depth = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        raise FileNotFoundError(f"Impossible de charger {image_path}")
    
    # Convertir en float32
    depth = depth.astype(np.float32)
    h, w = depth.shape
    print(f"Depth map chargée : résolution = {h}x{w}, dtype={depth.dtype}")

    # Paramètres caméra approximatifs basés sur la taille de l'image
    fx = fy = w
    cx, cy = w / 2, h / 2

    # Créer UNE seule instance du normalizer
    normalizer = DepthNormalizer(alpha=0.15)

    # Construire le paquet comme si c'était envoyé par la Personne A
    packet = {"frame_id": 1, "timestamp": 0.0, "depth_map": depth, "imgsz_used": w}

    # Construire le GeometryFrame (ici is_disparity=False car ton image est déjà une depth map)
    frame = build_geometry_frame(packet, fx, fy, cx, cy, normalizer, is_disparity=False)

    # Vérifier les infos
    print("Frame ID:", frame.frame_id)
    print("Timestamp:", frame.timestamp)
    print("Normal map shape:", frame.normal_map.shape)

    # Convertir la normal map en RGB pour affichage
    rgb = normals_to_rgb(frame.normal_map)

    # Afficher la carte de normales
    cv2.imshow("Normal Map (depth PNG)", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Sauvegarde en PNG pour debug
    cv2.imwrite("normal_map_from_depth.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print("✅ Image sauvegardée : normal_map_from_depth.png")

if __name__ == "__main__":
    test_depth_image("image.png")
