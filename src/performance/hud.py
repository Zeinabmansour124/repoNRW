"""
Interface de démonstration — titre, labels par panneau, barre de statut.
Donne une allure "produit fini" pour le jury.
"""

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

import numpy as np

# Couleurs (BGR)
COLOR_GOOD = (80, 220, 100)
COLOR_WARN = (0, 165, 255)
COLOR_BAD  = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_DIM  = (170, 170, 170)
COLOR_ACCENT = (255, 180, 40)     # bleu clair accent
BAR_BG     = (30, 30, 30)
PANEL_BG   = (18, 18, 18)


class HUD:
    """
    Compose une interface de démo autour du bandeau de vues.
    Usage :
        hud = HUD(target_fps=15)
        ui = hud.draw(combined, info, labels=["CAMERA","DEPTH","NORMALS","3D"])
    """

    def __init__(self, target_fps=15):
        self.target_fps = target_fps
        self.font = cv2.FONT_HERSHEY_SIMPLEX if _CV2_AVAILABLE else None
        self.header_h = 46
        self.footer_h = 40

    def _color_fps(self, fps):
        if fps >= self.target_fps * 0.8:
            return COLOR_GOOD
        elif fps >= self.target_fps * 0.5:
            return COLOR_WARN
        return COLOR_BAD

    def _text(self, img, text, x, y, color, scale=0.6, thick=2):
        cv2.putText(img, text, (x, y), self.font, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), self.font, scale, color, thick, cv2.LINE_AA)

    def draw(self, combined, info, labels=None):
        """
        combined : le bandeau des vues collées (H, W, 3)
        info     : dict {fps, latency_ms, mode, imgsz, light_xyz(optionnel), hand(optionnel)}
        labels   : liste de noms, un par vue (ex. ["CAMERA","DEPTH","NORMALS","3D"])
        """
        if not _CV2_AVAILABLE or combined is None:
            return combined

        H, W = combined.shape[:2]
        header = np.full((self.header_h, W, 3), PANEL_BG, dtype=np.uint8)
        footer = np.full((self.footer_h, W, 3), PANEL_BG, dtype=np.uint8)

        # --- Header : titre ---
        self._text(header, "AI and VISION  -  REAL-TIME 3D", 16, 31, COLOR_TEXT, 0.8, 2)
        # petit point d'état à droite
        fps = info.get("fps", 0)
        self._text(header, "LIVE", W - 70, 31, self._color_fps(fps), 0.7, 2)
        cv2.circle(header, (W - 85, 25), 6, self._color_fps(fps), -1)

        # --- Labels par panneau (répartis en colonnes égales) ---
        if labels:
            n = len(labels)
            col_w = W // n
            for i, lab in enumerate(labels):
                x = i * col_w + 12
                self._text(combined, lab, x, 26, COLOR_ACCENT, 0.6, 2)
                # séparateur vertical
                if i > 0:
                    cv2.line(combined, (i * col_w, 0), (i * col_w, H), (60, 60, 60), 1)

        # --- Footer : barre de statut ---
        fps = info.get("fps", 0)
        lat = info.get("latency_ms", 0)
        mode = info.get("mode", "?")
        imgsz = info.get("imgsz", 0)
        y = 26

        self._text(footer, f"FPS {fps:4.1f}", 16, y, self._color_fps(fps), 0.6, 2)
        self._text(footer, f"Latency {lat:3.0f} ms", 130, y, COLOR_DIM, 0.55, 1)
        self._text(footer, f"Mode {mode}", 310, y, COLOR_DIM, 0.55, 1)
        self._text(footer, f"imgsz {imgsz}", 470, y, COLOR_DIM, 0.55, 1)

        # Statut des modules (ON/OFF) : Depth toujours ON ici
        depth_on = True
        hand_on = info.get("hand", False)
        light = info.get("light_xyz", None)

        self._text(footer, "Depth", 590, y, COLOR_DIM, 0.55, 1)
        self._text(footer, "ON", 655, y, COLOR_GOOD, 0.55, 2)

        self._text(footer, "Hand", 710, y, COLOR_DIM, 0.55, 1)
        self._text(footer, "ON" if hand_on else "OFF", 770, y,
                   COLOR_GOOD if hand_on else COLOR_BAD, 0.55, 2)

        # Position lumière (si dispo)
        if light is not None:
            lx, ly, lz = light
            self._text(footer, f"Light X{lx:+.1f} Y{ly:+.1f} Z{lz:+.1f}",
                       830, y, COLOR_ACCENT, 0.55, 1)

        # Assembler : header + vues + footer
        return np.vstack((header, combined, footer))


if __name__ == "__main__":
    import time
    hud = HUD(target_fps=15)
    labels = ["CAMERA", "DEPTH", "NORMALS", "3D"]
    while True:
        combined = np.random.randint(30, 90, (320, 426 * 4, 3), dtype=np.uint8)
        info = {"fps": 15.2, "latency_ms": 66, "mode": "degraded",
                "imgsz": 256, "hand": False}
        ui = hud.draw(combined, info, labels=labels)
        cv2.imshow("UI test", ui)
        if cv2.waitKey(30) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
