"""
ghost_game.py
=============
Mini-jeu "Ghost in the Shadow" - NRW 8th Edition, AI & Vision Challenge.

Principe
--------
Le joueur deplace une lumiere virtuelle (main, X/Y/Z via MediaPipe) pour que
l'OMBRE projetee d'un objet reel recouvre une zone cible ("fantome") affichee
a l'ecran. Quand le recouvrement depasse un seuil, le fantome est "capture"
et le jeu passe au niveau suivant.

5 niveaux progressifs :
    Level 1 : fantome grand, centre                -> facile
    Level 2 : fantome plus petit, position laterale
    Level 3 : fantome encore plus petit, en coin (profondeur du placement)
    Level 4 : fantome mobile (deplacement lent et lisible)
    Level 5 : multi-fantomes (necessite plusieurs lumieres / mains)

Integration avec le pipeline existant
--------------------------------------
Ce module ne fait AUCUNE hypothese sur la source de la depth : il consomme
un shadow_mask (H, W) float32 dans [0, 1], EXACTEMENT le format retourne
par ShadowRenderer.compute() (performance/shadows.py) :
    1.0 = pixel eclaire
    0.0 = pixel dans l'ombre (occlus)

Exemple d'integration dans scripts/run_demo.py (dans la boucle principale,
juste apres avoir calcule shadow_total / le masque d'ombre courant) :

    from game.ghost_game import GhostGame

    # une seule fois, avant la boucle :
    game = GhostGame(frame_size=(RECT_W, RECT_H))

    # dans la boucle, apres avoir obtenu shadow_total (H,W) et le panneau
    # "relit" deja calcule (ex. relit ou last_hand_relit) :
    status = game.update(shadow_total)          # avance l'etat du jeu
    relit  = game.render(relit, shadow_total)    # dessine fantomes + HUD

    if status["finished"]:
        print("Victoire ! score =", status["score"])
"""

import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import cv2
import numpy as np


# ============================================================
# 1. CONFIGURATION DES NIVEAUX
# ============================================================

@dataclass
class GhostTarget:
    """Une cible (fantome) a capturer, en coordonnees normalisees [0,1]."""
    pos: Tuple[float, float]     # (x, y) normalise dans le panneau d'affichage
    radius: float                # rayon normalise (fraction de la largeur du panneau)
    captured: bool = False


@dataclass
class LevelConfig:
    level_id: int
    n_ghosts: int                          # nb de fantomes simultanes (multi-light au niv 5)
    radius: float                          # taille du/des fantome(s), normalisee
    positions: List[Tuple[float, float]]   # position(s) de depart, normalisees
    moving: bool = False                   # fantome qui se deplace lentement
    speed: float = 0.0                     # amplitude du mouvement (niveaux mobiles)
    time_limit: float = 20.0               # secondes pour reussir le niveau
    threshold: float = 0.65                # % de recouvrement requis pour capturer
    label: str = ""


# Difficulte croissante : position plus excentree, taille plus petite,
# puis mouvement, puis multi-cibles. Calibrer positions/radius sur le
# terrain (salle de demo reelle) avant le passage devant le jury.
LEVELS: List[LevelConfig] = [
    LevelConfig(level_id=1, n_ghosts=1, radius=0.140, positions=[(0.50, 0.50)],
                moving=False, time_limit=20.0, threshold=0.60,
                label="Level 1 - Centre, facile"),
    LevelConfig(level_id=2, n_ghosts=1, radius=0.110, positions=[(0.25, 0.40)],
                moving=False, time_limit=20.0, threshold=0.62,
                label="Level 2 - Lateral, plus petit"),
    LevelConfig(level_id=3, n_ghosts=1, radius=0.086, positions=[(0.75, 0.65)],
                moving=False, time_limit=22.0, threshold=0.65,
                label="Level 3 - Petit, coin difficile"),
    LevelConfig(level_id=4, n_ghosts=1, radius=0.068, positions=[(0.50, 0.50)],
                moving=True, speed=0.18, time_limit=25.0, threshold=0.65,
                label="Level 4 - Fantome mobile"),
    LevelConfig(level_id=5, n_ghosts=2, radius=0.053,
                positions=[(0.28, 0.35), (0.72, 0.65)],
                moving=True, speed=0.12, time_limit=30.0, threshold=0.60,
                label="Level 5 - Multi-fantomes"),
]


# ============================================================
# 2. GENERATION DE LA CIBLE COURANTE
# ============================================================

def get_current_target(level: int, elapsed: float,
                        frame_size: Tuple[int, int]) -> List[GhostTarget]:
    """
    Retourne la liste des fantomes actifs pour level, a l'instant elapsed
    (secondes depuis le debut du niveau).

    frame_size : (w, h) en pixels du panneau d'affichage. Sert a garder les
                 fantomes mobiles dans les bords (clip sur le rayon).
    """
    cfg = LEVELS[level - 1]
    targets = []

    for i, (px, py) in enumerate(cfg.positions):
        x, y = px, py
        if cfg.moving:
            # Mouvement circulaire doux et lisible (jamais brutal) :
            # chaque fantome oscille autour de sa position de depart.
            phase = i * (2 * np.pi / max(cfg.n_ghosts, 1))
            x = px + cfg.speed * 0.5 * np.sin(elapsed * 0.8 + phase)
            y = py + cfg.speed * 0.5 * np.cos(elapsed * 0.6 + phase)
            x = float(np.clip(x, cfg.radius, 1.0 - cfg.radius))
            y = float(np.clip(y, cfg.radius, 1.0 - cfg.radius))
        targets.append(GhostTarget(pos=(x, y), radius=cfg.radius))

    return targets


def _target_mask(target: GhostTarget, w: int, h: int) -> np.ndarray:
    """Masque binaire (H,W) uint8 : disque du fantome = 1, reste = 0."""
    mask = np.zeros((h, w), dtype=np.uint8)
    cx, cy = int(target.pos[0] * w), int(target.pos[1] * h)
    r = max(int(target.radius * w), 4)
    cv2.circle(mask, (cx, cy), r, 1, -1)
    return mask


# ============================================================
# 3. CALCUL DU RECOUVREMENT OMBRE / CIBLE
# ============================================================

def compute_overlap(shadow_mask: np.ndarray, target: GhostTarget,
                     shadow_threshold: float = 0.5) -> float:
    """
    shadow_mask : (H,W) float32 [0,1], convention ShadowRenderer.compute()
                  -> 1.0 = eclaire, 0.0 = ombre.
    target      : GhostTarget a tester.

    Retourne le taux (0.0-1.0) de la zone cible recouverte par l'OMBRE
    (le fantome se revele dans le NOIR, pas dans la lumiere).
    """
    h, w = shadow_mask.shape
    shadow_bin = (shadow_mask < shadow_threshold).astype(np.uint8)   # 1 = zone d'ombre
    target_bin = _target_mask(target, w, h)

    inter = cv2.bitwise_and(shadow_bin, target_bin)
    target_pixels = int(target_bin.sum())
    if target_pixels == 0:
        return 0.0
    return float(inter.sum()) / float(target_pixels)


# ============================================================
# 4. ETAT DU JEU (STATE MACHINE)
# ============================================================

class GhostGame:
    """
    Gere la progression Level 1 -> Level 5, le timer, le score et l'etat de
    chaque fantome. State machine :
        waiting -> captured -> level_up -> waiting (niveau suivant)
                -> failed   -> waiting (meme niveau, nouvel essai)
        (dernier niveau reussi) -> victory
    """

    STATE_WAITING  = "waiting"     # niveau en cours, capture pas encore complete
    STATE_FAILED   = "failed"      # temps ecoule sans capture -> a rejouer
    STATE_LEVEL_UP = "level_up"    # transition affichee brievement
    STATE_VICTORY  = "victory"     # tous les niveaux termines

    _FAIL_PAUSE = 1.2       # secondes d'affichage "LEVEL FAILED" avant reset
    _LEVEL_UP_PAUSE = 1.2   # secondes d'affichage "LEVEL UP !" avant le niveau suivant

    def __init__(self, frame_size: Tuple[int, int] = (300, 225),
                 n_levels: int = len(LEVELS)):
        self.frame_size = frame_size
        self.n_levels = n_levels
        self.level = 1
        self.score = 0
        self.state = self.STATE_WAITING
        self._level_start_t = time.perf_counter()
        self._pause_start_t: Optional[float] = None
        self.targets: List[GhostTarget] = get_current_target(self.level, 0.0, frame_size)
        self.last_overlaps: List[float] = [0.0] * len(self.targets)

    # -------------------------------------------------- #
    def _cfg(self) -> Optional[LevelConfig]:
        if self.level > self.n_levels:
            return None
        return LEVELS[self.level - 1]

    def _reset_level(self):
        """Relance le niveau courant a zero (nouvel essai, ou niveau suivant)."""
        self._level_start_t = time.perf_counter()
        self.targets = get_current_target(self.level, 0.0, self.frame_size)
        self.last_overlaps = [0.0] * len(self.targets)
        self.state = self.STATE_WAITING

    # -------------------------------------------------- #
    def update(self, shadow_mask: np.ndarray) -> dict:
        """
        A appeler une fois par frame avec le masque d'ombre courant, DEJA
        redimensionne a frame_size. Retourne l'etat courant du jeu.
        """
        now = time.perf_counter()

        # --- Etats de pause (transition ou echec) : on attend juste la fin ---
        if self.state == self.STATE_LEVEL_UP:
            if now - self._pause_start_t >= self._LEVEL_UP_PAUSE:
                self.level += 1
                if self.level > self.n_levels:
                    self.state = self.STATE_VICTORY
                else:
                    self._reset_level()
            return self._status(now)

        if self.state == self.STATE_FAILED:
            if now - self._pause_start_t >= self._FAIL_PAUSE:
                self._reset_level()
            return self._status(now)

        if self.state == self.STATE_VICTORY:
            return self._status(now)

        # --- Niveau en cours ---
        cfg = self._cfg()
        elapsed = now - self._level_start_t

        if elapsed >= cfg.time_limit:
            self.state = self.STATE_FAILED
            self._pause_start_t = now
            return self._status(now)

        # Position(s) courante(s) du/des fantome(s) (geree ici pour les niveaux mobiles)
        self.targets = get_current_target(self.level, elapsed, self.frame_size)

        # Recouvrement ombre/cible pour chaque fantome actif
        self.last_overlaps = [compute_overlap(shadow_mask, t) for t in self.targets]
        for t, ov in zip(self.targets, self.last_overlaps):
            if ov >= cfg.threshold:
                t.captured = True

        # Niveau reussi quand TOUS les fantomes sont captures
        # (Level 5 : necessite donc plusieurs lumieres/mains simultanement)
        if all(t.captured for t in self.targets):
            self.score += len(self.targets)
            self.state = self.STATE_LEVEL_UP
            self._pause_start_t = now

        return self._status(now)

    # -------------------------------------------------- #
    def _status(self, now: float) -> dict:
        cfg = self._cfg()
        elapsed = now - self._level_start_t
        time_left = max(0.0, cfg.time_limit - elapsed) if cfg else 0.0
        return {
            "state": self.state,
            "level": self.level,
            "level_label": cfg.label if cfg else "TERMINE",
            "score": self.score,
            "time_left": time_left,
            "overlaps": list(self.last_overlaps),
            "finished": self.state == self.STATE_VICTORY,
        }

    # -------------------------------------------------- #
    def render(self, panel: np.ndarray, shadow_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Dessine les fantomes (opacite = recouvrement courant) + HUD par-dessus
        un panneau deja rendu (ex. le panneau "RELIT 3D"). Retourne une copie,
        ne modifie pas panel en place.

        NOTE VISIBILITE : le cercle est dessine en NOIR OPAQUE + contour blanc
        epais, SANS transparence. Il est donc visible des le debut du niveau
        (alpha = 0), meme sur une image sombre. Le remplissage jaune au centre
        grandit avec le recouvrement de l'ombre (feedback de progression).
        """
        out = panel.copy()
        h, w = out.shape[:2]

        for t, ov in zip(self.targets, self.last_overlaps):
            cx, cy = int(t.pos[0] * w), int(t.pos[1] * h)
            r = max(int(t.radius * w), 4)
            alpha = float(np.clip(ov, 0.0, 1.0))

            # Cercle TOUJOURS visible : fond noir plein + contour blanc epais.
            # Aucun blend/transparence ici -> impossible de ne pas le voir,
            # meme quand alpha = 0 (avant que l'ombre n'atteigne la cible).
            cv2.circle(out, (cx, cy), r, (0, 0, 0), -1, cv2.LINE_AA)        # fond noir plein
            cv2.circle(out, (cx, cy), r, (255, 255, 255), 3, cv2.LINE_AA)   # contour blanc epais

            # Remplissage colore qui grandit avec l'ombre (feedback de progression)
            if alpha > 0.01:
                color = (0, int(255 * alpha), int(255 * alpha))   # jaune qui apparait progressivement
                cv2.circle(out, (cx, cy), max(r - 4, 2), color, -1, cv2.LINE_AA)

            # Petite croix au centre pour bien reperer la cible a viser
            cv2.drawMarker(out, (cx, cy), (255, 255, 255),
                           cv2.MARKER_CROSS, max(8, r // 3), 1, cv2.LINE_AA)

            # Jauge de progression au-dessus du fantome (0 -> seuil)
            cv2.putText(out, f"{int(alpha * 100)}%", (cx - 14, cy - r - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            if t.captured:
                cv2.putText(out, "CAPTURED", (cx - r, cy - r - 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 255, 120), 1, cv2.LINE_AA)

        status = self._status(time.perf_counter())
        cv2.putText(out, status["level_label"], (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 60), 1, cv2.LINE_AA)
        cv2.putText(out, f"Score {status['score']}", (10, h - 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"Temps {status['time_left']:.1f}s", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        if self.state == self.STATE_LEVEL_UP:
            cv2.putText(out, "LEVEL UP !", (w // 2 - 70, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 255, 120), 2, cv2.LINE_AA)
        elif self.state == self.STATE_FAILED:
            cv2.putText(out, "LEVEL FAILED", (w // 2 - 95, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 255), 2, cv2.LINE_AA)
        elif self.state == self.STATE_VICTORY:
            cv2.putText(out, "VICTOIRE !", (w // 2 - 75, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 220, 255), 2, cv2.LINE_AA)

        return out
