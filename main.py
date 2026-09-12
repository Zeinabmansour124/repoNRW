# main.py
"""
Point d'entree officiel du projet — NRW 8th Edition | AI & Vision Challenge.

Conformement au cahier des charges ("Must launch directly via python main.py"),
ce fichier lance directement le pipeline temps reel complet :

    CAMERA -> DEPTH (MiDaS) -> NORMALES -> 3D -> RELIGHTING + OMBRES + GESTES

Il ne fait que rediriger vers la boucle principale definie dans
scripts/run_demo.py, en s'assurant que le dossier src/ est bien dans le
PYTHONPATH quel que soit l'endroit d'ou la commande est lancee.

Lancement :
    python main.py
"""

import os
import sys

# Racine du projet = dossier contenant ce fichier
ROOT = os.path.dirname(os.path.abspath(__file__))

# Rendre les paquets du projet importables (src/ et scripts/)
for sub in ("src", "scripts"):
    path = os.path.join(ROOT, sub)
    if path not in sys.path:
        sys.path.insert(0, path)


def main():
    # Import differe : on ajoute les chemins AVANT d'importer run_demo,
    # qui lui-meme importe depth/, normals/, performance/ depuis src/.
    from run_demo import main as run_demo_main
    run_demo_main()


if __name__ == "__main__":
    main()