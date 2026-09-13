# webapp/server.py
"""
Serveur web "Ghost in the Shadow" — NRW 8th Edition, AI & Vision Challenge.

Lance :
    python webapp/server.py
puis ouvre  http://localhost:5000  dans le navigateur.

Écran 1 (menu)  : les 5 niveaux, verrouillés/débloqués, + le bandeau "Challenge".
Écran 2 (jeu)   : le flux vidéo du pipeline (depth/normales/3D + fantôme),
                  avec le HUD (niveau, temps, %, score).

Pas de compte : on clique un niveau débloqué, on joue, on débloque le suivant.
"""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for sub in ("src", "scripts", "webapp"):
    p = os.path.join(_ROOT, sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from flask import Flask, Response, jsonify, request, render_template

from game_controller import GameController
from pipeline import VisionPipeline, RECT_W, RECT_H

app = Flask(__name__, template_folder="templates", static_folder="static")

controller = GameController(frame_size=(RECT_W, RECT_H))
pipeline = VisionPipeline(controller)
_started = False


def _ensure_pipeline():
    """Démarre le pipeline vidéo au premier accès (webcam ouverte une seule fois)."""
    global _started
    if not _started:
        _started = True
        pipeline.start()


# ------------------------------------------------------------------ pages
@app.route("/")
def index():
    return render_template("index.html")


# ------------------------------------------------------------------ vidéo
def _mjpeg():
    _ensure_pipeline()
    # attendre que la première frame soit prête
    for _ in range(200):
        if pipeline.get_jpeg() is not None:
            break
        time.sleep(0.05)
    while True:
        frame = pipeline.get_jpeg()
        if frame is None:
            time.sleep(0.03)
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1 / 30)


@app.route("/video")
def video():
    return Response(_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ------------------------------------------------------------------ état + actions
@app.route("/state")
def state():
    snap = controller.snapshot()
    snap["pipeline"] = {
        "ready": pipeline.ready,
        "fps": round(pipeline.fps, 1),
        "error": pipeline.error,
    }
    return jsonify(snap)


@app.route("/play/<int:level>", methods=["POST"])
def play(level):
    _ensure_pipeline()
    return jsonify(controller.start_level(level))


@app.route("/menu", methods=["POST"])
def menu():
    return jsonify(controller.back_to_menu())


@app.route("/restart", methods=["POST"])
def restart():
    return jsonify(controller.restart_current())


if __name__ == "__main__":
    print("=" * 64)
    print("  GHOST IN THE SHADOW  —  http://localhost:5000")
    print("=" * 64)
    # threaded=True : le stream vidéo et les requêtes d'état cohabitent
    app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
