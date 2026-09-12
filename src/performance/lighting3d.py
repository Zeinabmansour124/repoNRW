# src/performance/lighting3d.py
"""
Level 2 — Relighting 3D (Lambert diffus + Blinn-Phong speculaire), multi-lumieres.
Module de Malek, corrige (emojis -> 0.8, __name__/__main__, imports).
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from normals.normalisation import (
    DepthNormalizer,
    GeometryFrame,
    build_geometry_frame,
    deproject,
    _generate_fake_depth_packet,
)


# ----------------------------------------------------------------------------
# 1. Parametres configurables (lumiere + materiau)
# ----------------------------------------------------------------------------

@dataclass
class LightParams:
    position: Tuple[float, float, float]        # (X, Y, Z) position 3D de la lumiere
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    attenuation: bool = True
    attenuation_radius: float = 2000.0


@dataclass
class MaterialParams:
    ambient: float = 0.08
    diffuse_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
    specular_color: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    shininess: float = 32.0


LightParamsOrList = Union[LightParams, List[LightParams]]


# ----------------------------------------------------------------------------
# 2. Utilitaires vectorises
# ----------------------------------------------------------------------------

def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Produit scalaire pixel-par-pixel (H,W,3)x(H,W,3) -> (H,W,1). einsum ~3x plus rapide."""
    return np.einsum("...i,...i->...", a, b)[..., np.newaxis]


def _normalize_vectors(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.sqrt(_dot(v, v))
    return v / np.maximum(norm, eps)


def _as_vec3(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).reshape(1, 1, 3)


def _as_light_list(light_params: LightParamsOrList) -> List[LightParams]:
    if isinstance(light_params, LightParams):
        return [light_params]
    if not light_params:
        raise ValueError("light_params ne peut pas etre une liste vide.")
    return list(light_params)


# ----------------------------------------------------------------------------
# 3. Contribution d'une seule lumiere (Lambert + Blinn-Phong)
# ----------------------------------------------------------------------------

def _shade_one_light(P: np.ndarray, N: np.ndarray, V: np.ndarray,
                     light: LightParams, material: MaterialParams) -> np.ndarray:
    light_pos_v = _as_vec3(light.position)

    L_vec = light_pos_v - P
    dist = np.sqrt(_dot(L_vec, L_vec))
    L = L_vec / np.maximum(dist, 1e-8)

    H_vec = _normalize_vectors(L + V)

    if light.attenuation:
        d_norm = dist / max(light.attenuation_radius, 1e-6)
        attenuation = 1.0 / (1.0 + d_norm ** 2)
    else:
        attenuation = np.ones_like(dist)

    n_dot_l = np.clip(_dot(N, L), 0.0, 1.0)
    diffuse = n_dot_l * _as_vec3(material.diffuse_color)

    n_dot_h = np.clip(_dot(N, H_vec), 0.0, 1.0)
    specular = (n_dot_h ** material.shininess) * _as_vec3(material.specular_color)
    specular = specular * (n_dot_l > 0.0)

    return attenuation * light.intensity * _as_vec3(light.color) * (diffuse + specular)


# ----------------------------------------------------------------------------
# 4. Fonction principale : apply_lighting_3d
# ----------------------------------------------------------------------------

def apply_lighting_3d(point_map, normal_map, light_params, view_pos,
                      *, material: Optional[MaterialParams] = None,
                      output_uint8: bool = True) -> np.ndarray:
    if material is None:
        material = MaterialParams()

    lights = _as_light_list(light_params)

    P = point_map.astype(np.float32)
    N = normal_map.astype(np.float32)
    view_pos_v = _as_vec3(view_pos)

    V = _normalize_vectors(view_pos_v - P)

    accumulated = np.zeros_like(P)
    for light in lights:
        accumulated += _shade_one_light(P, N, V, light, material)

    ambient_term = material.ambient * _as_vec3(material.diffuse_color)
    lit = np.clip(ambient_term + accumulated, 0.0, 1.0)

    if not output_uint8:
        return lit

    img_uint8 = (lit * 255.0).astype(np.uint8)
    return cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# 5. Compatibilite ascendante
# ----------------------------------------------------------------------------

def apply_lighting(normal_map, depth_map, light_pos, view_pos, *,
                   point_map=None, intrinsics=None, light=None,
                   material=None, output_uint8=True) -> np.ndarray:
    if point_map is None:
        h, w = depth_map.shape
        if intrinsics is None:
            fx = fy = float(max(h, w))
            cx, cy = w / 2.0, h / 2.0
        else:
            fx, fy, cx, cy = intrinsics
        point_map = deproject(depth_map, fx, fy, cx, cy)

    if light is None:
        light = LightParams(position=light_pos)

    return apply_lighting_3d(point_map=point_map, normal_map=normal_map,
                             light_params=light, view_pos=view_pos,
                             material=material, output_uint8=output_uint8)


# ----------------------------------------------------------------------------
# 6. Relighting direct depuis un GeometryFrame
# ----------------------------------------------------------------------------

def relight_geometry_frame(frame: GeometryFrame, light_params, view_pos,
                           material: Optional[MaterialParams] = None) -> np.ndarray:
    return apply_lighting_3d(point_map=frame.point_map, normal_map=frame.normal_map,
                             light_params=light_params, view_pos=view_pos,
                             material=material)


# ----------------------------------------------------------------------------
# 7. Demo / bench
# ----------------------------------------------------------------------------

def run_demo(n_frames: int = 60, h: int = 480, w: int = 640,
             n_lights: int = 2, save_image: bool = True):
    fx = fy = w
    cx, cy = w / 2.0, h / 2.0
    normalizer = DepthNormalizer(alpha=0.15)
    view_pos = (0.0, 0.0, 0.0)

    material = MaterialParams(ambient=0.05, diffuse_color=(0.75, 0.75, 0.8),
                              specular_color=(1.0, 1.0, 1.0), shininess=48.0)

    light_colors = [(1.0, 0.95, 0.85), (0.5, 0.6, 1.0), (1.0, 0.4, 0.4), (0.4, 1.0, 0.5)]
    times = []
    last_frame = None

    for i in range(n_frames):
        packet = _generate_fake_depth_packet(frame_id=i, timestamp=i / 30.0, h=h, w=w)
        frame = build_geometry_frame(packet, fx, fy, cx, cy, normalizer, is_disparity=True)

        lights = []
        for k in range(n_lights):
            phase = 2 * np.pi * k / n_lights
            angle = 2 * np.pi * (i / n_frames) + phase
            light_pos = (600 * np.cos(angle), 300 * np.sin(angle), -400.0)
            lights.append(LightParams(position=light_pos,
                                      color=light_colors[k % len(light_colors)],
                                      intensity=1.4 / n_lights * 1.6))

        t0 = time.perf_counter()
        img = relight_geometry_frame(frame, light_params=lights, view_pos=view_pos, material=material)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        last_frame = img

    times = np.array(times)
    fps = 1.0 / times.mean()
    print(f"Resolution {h}x{w} | Lumieres {n_lights} | {times.mean()*1000:.2f} ms/frame | FPS {fps:.1f}")

    if last_frame is not None and save_image:
        cv2.imwrite("relighting3d_demo_last_frame.png", last_frame)

    return fps


if __name__ == "__main__":
    run_demo()


# ----------------------------------------------------------------------------
# 8. Halo lumineux visible (la source de lumiere dessinee dans l'image)
# ----------------------------------------------------------------------------

def draw_light_halo(img, lx, ly, lz, color_bgr=(210, 240, 255)):
    """
    Dessine un halo lumineux additif a la position (lx, ly) sur l'image (BGR uint8).
    lx, ly : position normalisee [-1,1] de la lumiere (x: droite, y: bas).
    lz     : profondeur [-2,-0.2] ; plus proche (vers -0.2) = halo plus gros/vif.
    Rend la SOURCE visible -> le spectateur "voit" la lumiere qu'il deplace.
    """
    h, w = img.shape[:2]
    # position pixel
    cx = int((lx / 2.0 + 0.5) * w)
    cy = int((ly / 2.0 + 0.5) * h)

    # taille du halo selon la proximite (z proche -> plus gros)
    prox = (abs(lz) - 0.2) / 1.8          # 0 (proche) .. 1 (loin)
    prox = max(0.0, min(1.0, prox))
    radius = int((0.16 - 0.08 * prox) * w)   # halo plus petit et net
    radius = max(12, radius)

    # champ radial doux (gaussien) calcule une seule fois
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
    glow = np.exp(-dist2 / (2.0 * (radius * 0.5) ** 2))    # halo diffus
    glow = glow[..., None]                                  # (H,W,1)

    color = np.array(color_bgr, dtype=np.float32).reshape(1, 1, 3)
    out = img.astype(np.float32) + glow * color * (1.1 - 0.5 * prox)
    out = np.clip(out, 0, 255).astype(np.uint8)

    # coeur brillant net au centre (petit et propre)
    cv2.circle(out, (cx, cy), max(3, radius // 14), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(out, (cx, cy), max(5, radius // 9), (255, 255, 245), 1, cv2.LINE_AA)
    return out
