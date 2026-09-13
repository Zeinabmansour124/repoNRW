# webapp/game_controller.py
"""
Contrôleur de jeu pour l'interface web "Ghost in the Shadow".

Rôle
----
Faire le pont entre :
  - le moteur de jeu existant  src/game/ghost_game.py  (GhostGame),
  - l'interface web (menu des niveaux + HUD) via un état JSON simple.

Ce contrôleur ajoute ce que le hackathon demande, SANS toucher au moteur :
  * un écran MENU (on ne joue pas tant qu'on n'a pas cliqué "PLAY"),
  * un déblocage progressif : le niveau N+1 se débloque quand le niveau N
    est réussi (comme les cadenas des maquettes),
  * la possibilité de (re)lancer un niveau débloqué depuis le menu,
  * pas de compte utilisateur : la progression vit en mémoire, le temps
    d'une session de démo.

Il est THREAD-SAFE : le pipeline vidéo (thread serveur) appelle update()
pendant que les requêtes HTTP lisent snapshot() / déclenchent start_level().
"""

import threading
import time
from typing import Optional

import numpy as np

from game.ghost_game import GhostGame, LEVELS


# Métadonnées d'affichage (titres/sous-titres des maquettes). Purement cosmétique :
# la difficulté réelle vient de LEVELS dans ghost_game.py.
LEVEL_META = [
    {"id": 1, "name": "The First Appearition", "room": "Salon"},
    {"id": 2, "name": "The Side Ghost",        "room": "Chambre"},
    {"id": 3, "name": "The Corner",            "room": "Couloir"},
    {"id": 4, "name": "The Moving Ghost",      "room": "Bureau"},
    {"id": 5, "name": "The Multi-Ghosts",      "room": "Grenier"},
]

# Écrans de l'application web
SCREEN_MENU = "menu"      # on choisit un niveau (les verrouillés sont grisés)
SCREEN_PLAY = "play"      # un niveau est en cours


class GameController:
    def __init__(self, frame_size=(300, 225)):
        self._lock = threading.Lock()
        self.frame_size = frame_size
        self.game: Optional[GhostGame] = None

        self.screen = SCREEN_MENU
        self.max_unlocked = 1          # niveau 1 débloqué d'entrée
        self.total_score = 0           # cumulé sur la session
        self._last_state = None        # pour détecter les transitions level_up / victory

    # ------------------------------------------------------------------ #
    # Actions déclenchées par l'interface (routes HTTP)
    # ------------------------------------------------------------------ #
    def start_level(self, level: int) -> dict:
        """Lance un niveau débloqué. Retourne le snapshot résultant."""
        with self._lock:
            level = int(level)
            if level < 1 or level > len(LEVELS):
                return self._snapshot_locked()
            if level > self.max_unlocked:
                # niveau verrouillé : on ignore, on reste au menu
                return self._snapshot_locked()

            # On instancie un GhostGame frais et on le positionne AU niveau voulu.
            g = GhostGame(frame_size=self.frame_size)
            g.level = level
            g.score = 0
            g._reset_level()          # (ré)arme le timer et les cibles du niveau
            self.game = g
            self.screen = SCREEN_PLAY
            self._last_state = g.state
            return self._snapshot_locked()

    def back_to_menu(self) -> dict:
        with self._lock:
            self.screen = SCREEN_MENU
            self.game = None
            return self._snapshot_locked()

    def restart_current(self) -> dict:
        with self._lock:
            if self.game is not None:
                self.game._reset_level()
                self._last_state = self.game.state
            return self._snapshot_locked()

    # ------------------------------------------------------------------ #
    # Appelé par le pipeline vidéo, une fois par frame
    # ------------------------------------------------------------------ #
    def update(self, shadow_mask: Optional[np.ndarray]):
        """Avance l'état du jeu si un niveau est en cours."""
        with self._lock:
            if self.screen != SCREEN_PLAY or self.game is None:
                return
            if shadow_mask is None:
                return

            prev_level = self.game.level
            prev_state = self.game.state
            self.game.update(shadow_mask)

            # Détecter le passage d'un niveau : GhostGame incrémente self.level
            # tout seul après le petit temps d'affichage "LEVEL UP".
            if self.game.level > prev_level:
                # niveau précédent réussi -> on débloque le suivant
                self.max_unlocked = max(self.max_unlocked, self.game.level)
                self.total_score += 1

            # Victoire finale : GhostGame passe en STATE_VICTORY quand le
            # dernier niveau est franchi.
            if self.game.state == GhostGame.STATE_VICTORY and prev_state != GhostGame.STATE_VICTORY:
                self.max_unlocked = len(LEVELS)   # tout débloqué
                self.total_score += 500           # bonus "challenge complet"

            self._last_state = self.game.state

    def render(self, panel_bgr: np.ndarray, shadow_mask: Optional[np.ndarray]):
        """Dessine fantômes + HUD interne du moteur sur le panneau vidéo."""
        with self._lock:
            if self.screen == SCREEN_PLAY and self.game is not None:
                return self.game.render(panel_bgr, shadow_mask)
            return panel_bgr

    # ------------------------------------------------------------------ #
    # État lisible par l'interface (JSON)
    # ------------------------------------------------------------------ #
    def snapshot(self) -> dict:
        with self._lock:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict:
        levels = []
        for meta in LEVEL_META:
            cfg = LEVELS[meta["id"] - 1]
            levels.append({
                "id": meta["id"],
                "name": meta["name"],
                "room": meta["room"],
                "time_limit": cfg.time_limit,
                "threshold": int(cfg.threshold * 100),
                "unlocked": meta["id"] <= self.max_unlocked,
                "completed": meta["id"] < self.max_unlocked
                              or (self.max_unlocked == len(LEVELS)
                                  and self.game is None and self.screen == SCREEN_MENU
                                  and meta["id"] <= self.max_unlocked),
            })

        snap = {
            "screen": self.screen,
            "max_unlocked": self.max_unlocked,
            "total_score": self.total_score,
            "n_levels": len(LEVELS),
            "levels": levels,
            "playing": None,
        }

        if self.screen == SCREEN_PLAY and self.game is not None:
            st = self.game._status(time.perf_counter())
            cfg = LEVELS[self.game.level - 1]
            overlaps = st.get("overlaps", [])
            best = int(round(max(overlaps) * 100)) if overlaps else 0
            snap["playing"] = {
                "level": self.game.level,
                "name": LEVEL_META[self.game.level - 1]["name"],
                "room": LEVEL_META[self.game.level - 1]["room"],
                "state": st["state"],
                "score": st["score"],
                "time_left": round(st["time_left"], 1),
                "threshold": int(cfg.threshold * 100),
                "overlap_pct": best,               # meilleur recouvrement courant
                "overlaps": [int(round(o * 100)) for o in overlaps],
                "n_ghosts": cfg.n_ghosts,
                "finished": st["finished"],
            }
        return snap
