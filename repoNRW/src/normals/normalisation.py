import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from dataclasses import dataclass


def load_depth_map(source, is_uint8_png: bool = False) -> np.ndarray:
    """
    Charge une depth map depuis un fichier .npy (float brut) ou une image
    .png/.jpg (uint8, comme souvent exporté pour visualiser une sortie MiDaS).

    source        : chemin du fichier, OU déjà un np.ndarray (dans ce cas on
                    le retourne tel quel après conversion en float64).
    is_uint8_png  : True si le fichier est une image 8 bits normalisée
                    (valeurs 0-255) plutôt qu'un tableau float brut.

    Retourne : depth map en float64, valeurs BRUTES du modèle (pas encore
    corrigées de sens ni d'échelle -- ça, c'est le job de la fonction suivante).
    """
    if isinstance(source, np.ndarray):
        return source.astype(np.float64)

    if is_uint8_png:
        img = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger {source}")
        return img.astype(np.float64)

    return np.load(source).astype(np.float64)


class DepthNormalizer:
    """
    Normalise la depth brute en [0,1] de façon STABLE dans le temps.
    """
    def __init__(self, alpha: float = 0.15, p_low: float = 2.0, p_high: float = 98.0):
        self.alpha = alpha
        self.p_low = p_low
        self.p_high = p_high
        self._d_min = None
        self._d_max = None

    def normalize(self, raw_depth: np.ndarray) -> np.ndarray:
        # La depth arrive DEJA normalisee [0,1] depuis get_depth.
        # On ne re-normalise PAS (evite la double normalisation).
        # On garde juste une stabilisation temporelle (anti-jitter) tres legere.
        depth = np.clip(raw_depth.astype(np.float32), 0.0, 1.0)
        if self._d_min is None:              # 1re frame : on memorise
            self._prev = depth
            self._d_min = 0.0
        else:
            depth = (1 - self.alpha) * self._prev + self.alpha * depth
            self._prev = depth
        return depth


def disparity_to_depth(raw_depth: np.ndarray, normalizer: "DepthNormalizer",
                        is_disparity: bool = True, eps: float = 1e-6) -> np.ndarray:
    disparity_norm = normalizer.normalize(raw_depth)

    if not is_disparity:
        return disparity_norm

    return 1.0 / (disparity_norm + eps)


# Cache du meshgrid (u,v) : identique a chaque frame pour une taille donnee
_UV_CACHE = {}

def _get_uv_grid(h, w):
    key = (h, w)
    if key not in _UV_CACHE:
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        _UV_CACHE[key] = (u, v)
    return _UV_CACHE[key]


def deproject(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    h, w = depth.shape
    u, v = _get_uv_grid(h, w)

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return np.stack([X, Y, Z], axis=-1)


def compute_normals(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                     smooth_sigma: float = 3.0, eps: float = 1e-6) -> tuple:
    # cv2.GaussianBlur est ~100x plus rapide que scipy.gaussian_filter
    if smooth_sigma > 0:
        k = int(smooth_sigma * 4) | 1   # taille de noyau impaire
        depth_smooth = cv2.GaussianBlur(depth, (k, k), smooth_sigma)
    else:
        depth_smooth = depth

    P = deproject(depth_smooth, fx, fy, cx, cy)

    P_pad = np.pad(P, ((1, 1), (1, 1), (0, 0)), mode="edge")

    Tu = P_pad[1:-1, 2:, :] - P_pad[1:-1, :-2, :]
    Tv = P_pad[2:, 1:-1, :] - P_pad[:-2, 1:-1, :]

    N = np.cross(Tu, Tv)

    view_vec = -P
    dot = np.sum(N * view_vec, axis=-1, keepdims=True)
    N = np.where(dot < 0, -N, N)

    N = np.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(N, axis=-1, keepdims=True)

    degenerate = (norm < eps).squeeze(-1)
    norm_safe = np.where(norm < eps, 1.0, norm)
    N = N / norm_safe
    N[degenerate] = np.array([0.0, 0.0, -1.0])

    return N, P, depth_smooth


def normals_to_rgb(N: np.ndarray) -> np.ndarray:
    # Encodage standard normal map : [-1,1] -> [0,255]
    rgb = ((N + 1.0) * 0.5 * 255.0)
    # Boost de contraste/saturation pour un rendu plus vif ("wow")
    rgb = np.clip((rgb - 128.0) * 1.35 + 128.0, 0, 255)
    return rgb.astype(np.uint8)


@dataclass
class GeometryFrame:
    frame_id: int
    timestamp: float
    normal_map: np.ndarray
    point_map: np.ndarray
    depth_map: np.ndarray
    valid_mask: np.ndarray
    intrinsics: tuple
    resolution: tuple


def parse_depth_packet(packet: dict) -> tuple:
    required = {"frame_id", "timestamp", "depth_map"}
    missing = required - packet.keys()
    if missing:
        raise ValueError(f"Paquet incomplet, champs manquants : {missing}")

    depth_map = packet["depth_map"]
    if not isinstance(depth_map, np.ndarray) or depth_map.ndim != 2:
        raise ValueError(f"depth_map doit être un np.ndarray 2D (H,W), reçu : "
                          f"{type(depth_map)} shape={getattr(depth_map, 'shape', None)}")

    return packet["frame_id"], packet["timestamp"], depth_map.astype(np.float32)


def build_geometry_frame(depth_packet: dict, fx: float, fy: float, cx: float, cy: float,
                          normalizer: "DepthNormalizer", is_disparity: bool = True) -> GeometryFrame:
    frame_id, timestamp, raw_depth = parse_depth_packet(depth_packet)

    depth = disparity_to_depth(raw_depth, normalizer, is_disparity=is_disparity)
    N, P, depth_smooth = compute_normals(depth, fx, fy, cx, cy)

    # valid_mask : simplement les pixels où la normale est non nulle
    # (evite de recalculer tout le cross-product une 2e fois -> gros gain FPS)
    valid_mask = np.abs(N).sum(axis=-1) > 1e-6

    return GeometryFrame(
        frame_id=frame_id,
        timestamp=timestamp,
        normal_map=N.astype(np.float32),
        point_map=P.astype(np.float32),
        depth_map=depth_smooth.astype(np.float32),
        valid_mask=valid_mask,
        intrinsics=(fx, fy, cx, cy),
        resolution=depth.shape,
    )


def _generate_fake_depth_packet(frame_id=0, timestamp=0.0, h=480, w=640,
                                 raw_min=-37.0, raw_max=1032.0, frame_shift=0.0) -> dict:
    u, v = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r2 = u ** 2 + v ** 2
    radius = 0.5

    disparity01 = np.where(r2 < radius ** 2, np.sqrt(np.clip(radius ** 2 - r2, 0, None)), 0.0)
    disparity01 = disparity01 / disparity01.max()
    disparity01 = disparity01 * 0.7 + 0.3

    noise = np.random.normal(0, 0.015, size=disparity01.shape)
    disparity01 = np.clip(disparity01 + noise, 0.01, 1.0)

    raw = disparity01 * (raw_max - raw_min) + raw_min + frame_shift

    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "depth_map": raw.astype(np.float32),
        "imgsz_used": 384,
    }


def run_isolated_test():
    print("=== TEST EN ISOLATION (vrai contrat d'entrée, 480x640) ===\n")

    h, w = 1332, 2048
    fx = fy = w
    cx, cy = w / 2, h / 2
    normalizer = DepthNormalizer(alpha=0.15)

    packet_1 = _generate_fake_depth_packet(frame_id=42, timestamp=1.400, h=h, w=w, frame_shift=0.0)
    print(f"Paquet reçu : frame_id={packet_1['frame_id']}, timestamp={packet_1['timestamp']}, "
          f"depth_map.shape={packet_1['depth_map'].shape}, imgsz_used={packet_1['imgsz_used']}")
    print(f"  depth_map brut : min={packet_1['depth_map'].min():.1f}, "
          f"max={packet_1['depth_map'].max():.1f}")

    frame_1 = build_geometry_frame(packet_1, fx, fy, cx, cy, normalizer, is_disparity=True)

    packet_2 = _generate_fake_depth_packet(frame_id=43, timestamp=1.433, h=h, w=w, frame_shift=150.0)
    frame_2 = build_geometry_frame(packet_2, fx, fy, cx, cy, normalizer, is_disparity=True)

    diff_center = np.linalg.norm(
        frame_1.normal_map[h // 2, w // 2] - frame_2.normal_map[h // 2, w // 2]
    )
    print(f"\nÉcart de normale au centre entre frame 42 et 43 (scène quasi identique) "
          f"= {diff_center:.4f} (doit rester petit grâce à l'EMA)")

    print(f"\nGeometryFrame (frame_id={frame_1.frame_id}, timestamp={frame_1.timestamp}) :")
    print(f"  normal_map  shape={frame_1.normal_map.shape}, dtype={frame_1.normal_map.dtype}")
    print(f"  point_map   shape={frame_1.point_map.shape}, dtype={frame_1.point_map.dtype}")
    print(f"  depth_map   shape={frame_1.depth_map.shape}, dtype={frame_1.depth_map.dtype}")
    print(f"  valid_mask  {frame_1.valid_mask.sum()}/{frame_1.valid_mask.size} pixels valides")
    print(f"  intrinsics  = {frame_1.intrinsics}")
    print(f"  resolution  = {frame_1.resolution}")

    norms = np.linalg.norm(frame_1.normal_map, axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-3), "Les normales ne sont pas unitaires !"
    assert not np.isnan(frame_1.normal_map).any(), "NaN détecté dans normal_map !"
    assert not np.isinf(frame_1.normal_map).any(), "Inf détecté dans normal_map !"
    assert frame_1.normal_map.shape == frame_1.point_map.shape, "Shapes incohérentes !"
    assert frame_1.frame_id == packet_1["frame_id"], "frame_id non propagé correctement !"
    assert frame_1.resolution == (h, w), "La résolution ne correspond pas à depth_map.shape !"
    print("\n✅ Toutes les vérifications automatiques passent "
          "(normales unitaires, pas de NaN/Inf, shapes cohérentes, frame_id propagé).")

    center_normal = frame_1.normal_map[h // 2, w // 2]
    print(f"\nNormale au centre du sujet (attendu ~[0,0,-1]) : {center_normal}")

    rgb = normals_to_rgb(frame_1.normal_map)
    cv2.imwrite("test_normal_map.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    raw = packet_1["depth_map"]
    cv2.imwrite("test_input_depth.png",
                ((raw - raw.min()) / (raw.max() - raw.min()) * 255).astype(np.uint8))
    print("\nImages de debug écrites : test_input_depth.png, test_normal_map.png")

    return frame_1, frame_2


if __name__ == "__main__":
    run_isolated_test()