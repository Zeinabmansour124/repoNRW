# Ghost in the Shadow — Interface web (démo hackathon)

Interface jouable des **deux maquettes** (menu des niveaux + écran de jeu HUD),
branchée directement sur le pipeline de vision existant
(`CAMERA → DEPTH → NORMALES → 3D → RELIGHT + OMBRES + MAIN`).

Pas de compte utilisateur : on ouvre la page, on clique un niveau **débloqué**,
on joue avec l'ombre de sa main, on **débloque le suivant**, et ainsi de suite
jusqu'à la victoire (+500 points).

## Lancer

```bash
# depuis la racine du repo (repoNRW/)
pip install -r requirements.txt          # torch, mediapipe, opencv... + flask
python webapp/server.py
```

Puis ouvrir **http://localhost:5000**.

> La première frame prend quelques secondes (ouverture webcam + chargement MiDaS).
> L'écran de jeu affiche « Ouverture de la caméra… » tant que ce n'est pas prêt.

## Ce que fait chaque fichier

| Fichier | Rôle |
|---|---|
| `templates/index.html` | Les 2 écrans (menu néon + HUD de jeu), fidèles aux maquettes. Un seul fichier, tout le CSS/JS inclus. |
| `server.py` | Serveur Flask : sert la page, le flux vidéo (`/video`, MJPEG), l'état du jeu (`/state`, JSON) et les actions (`/play/<n>`, `/menu`, `/restart`). |
| `pipeline.py` | Le pipeline temps réel **sans fenêtre OpenCV** : mêmes modules que `scripts/run_demo.py`, mais il encode les frames en JPEG pour le navigateur et pousse le masque d'ombre au jeu. |
| `game_controller.py` | Enveloppe `GhostGame` : écran menu, **déblocage progressif** des niveaux, score de session. Ne modifie pas le moteur. |

## Correspondance avec les maquettes

- **Image 2 (menu)** → écran `menu` : 5 cartes de niveaux, cadenas sur les
  niveaux non débloqués, ✓ vert sur les niveaux réussis, bandeau « THE
  CHALLENGE » + récompense. Clic sur une carte débloquée = on lance ce niveau.
- **Image 1 (jeu)** → écran `play` : flux vidéo (fantôme + depth/normales/ombre),
  HUD niveau `2/5`, temps restant, barre de progression du recouvrement,
  pourcentage en direct, transitions `LEVEL UP ! / LEVEL FAILED / VICTOIRE !`.

## API (pour brancher autre chose dessus)

```
GET  /state          -> état complet (écran, niveaux, unlock, jeu en cours)
GET  /video          -> flux MJPEG du pipeline
POST /play/<level>   -> lance un niveau débloqué
POST /menu           -> retour au menu
POST /restart        -> relance le niveau courant
```

## Déployer pour le pitch

Le jeu **a besoin d'une webcam locale** (hand tracking) : il tourne donc sur
**la machine de démo**, pas sur un serveur distant sans caméra. Pour montrer
l'interface à distance sans webcam (mode vitrine), on peut :
- exposer le port avec un tunnel (`ngrok http 5000`) le temps du pitch, ou
- enregistrer une courte vidéo de l'écran de jeu et garder le menu web live.

## Fallback 100 % sûr

Si le web pose souci le jour J, la version native marche toujours :
```bash
python main.py     # fenêtre OpenCV, touche 'g' pour lancer le jeu Ghost
```
