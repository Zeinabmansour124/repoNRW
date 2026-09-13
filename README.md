
# Moteur de Vision Monoculaire & Ré-éclairage Temps Réel

**Un système de vision par ordinateur en temps réel qui transforme un flux caméra monoculaire en géométrie 3D, puis y rend une source de lumière virtuelle contrôlée à la main — éclairage dynamique, ombres portées réalistes, réactivité aux gestes en X, Y, Z.**

---

## Objectif principal

Développer un système de vision temps réel **à haut débit** qui :

1. prend un **flux caméra standard** (monoculaire, RGB) ;
2. extrait une **géométrie 3D de profondeur précise au pixel** ;
3. rend une **source de lumière virtuelle contrôlée par l'utilisateur** qui éclaire dynamiquement la scène ;
4. projette des **ombres dynamiques réalistes** ;
5. réagit **sans latence** aux gestes de la main, y compris le déplacement de la lumière **vers l'avant et vers l'arrière dans l'espace 3D (X, Y, Z)**.

Le tout **sans LiDAR**, uniquement à partir de modèles de profondeur monoculaire tournant **de façon synchrone** avec le pipeline de rendu, à très faible latence.

---

## Vision

- **Perception spatiale** — transformer un flux RGB 2D plat en environnement 3D réactif et mathématiquement cohérent, sans matériel LiDAR dédié.
- **IA temps réel embarquée** — prouver que des modèles modernes de profondeur monoculaire tournent **de façon synchrone** avec un pipeline graphique, à latence minimale.

---

## Le Pipeline

```
[ CAMÉRA ] -- flux RGB monoculaire
     v
[ ESTIMATION DE PROFONDEUR ] -- MiDaS · distance par pixel · fil d'exécution dédié
     v
[ MOTEUR DE GÉOMÉTRIE ] -- normales de surface N(x,y) depuis les gradients de profondeur
     v
[ RECONSTRUCTION 3D ] -- nuage de points / géométrie de la scène
     v
[ RÉ-ÉCLAIRAGE DYNAMIQUE ] -- diffus Lambertien + spéculaire sur la géométrie
     v
[ OMBRES DYNAMIQUES ] -- ombres d'occlusion projetées (lancer de rayons)
     ^
     |  position de la lumière (X, Y, Z)
[ CONTRÔLE GESTUEL SPATIAL ] -- MediaPipe · la main pilote la lumière
```

Chaque étape est **affichée à l'écran en direct** (profondeur, normales, 3D, eclairage controlee par la souris + ombre , eclairage controle par les mais + ombre) — c'est la **preuve visuelle** attendue à chaque niveau de la progression, et la garantie qu'aucun rendu n'est pré-calculé.

---

## Progression Technique

Notre système couvre toute la progression du cahier des charges. Chaque niveau ajoute une brique au pipeline, et chaque brique est **visible à l'écran** — c'est la preuve qu'elle fonctionne vraiment, en direct.

**Niveau 01 — Géométrie**
On calcule l'orientation de chaque surface (les *normales*) à partir de la carte de profondeur.
→ *Preuve : la carte des normales s'affiche en temps réel, synchronisée avec la caméra.*
→ Code : `fast_normals.py`, `normalisation.py`

**Niveau 02 — Ré-éclairage**
On éclaire la scène 3D avec une lumière virtuelle (éclairage diffus + reflets).
→ *Preuve : la scène s'illumine de façon réaliste selon la position de la lumière controlé par la souris .*
→ Code : `lighting3d.py`, `lighting.py`

**Niveau 03 — Contrôle par les gestes**
On suit la main ou les mains pour déplacer la lumière dans l'espace.
→ *Preuve : la lumière suit la main en X, Y et Z (on peut la rapprocher ou l'éloigner).*
→ Code : `hand_light.py`

**Niveau 04 — Ombres dynamiques**
On projette des ombres réalistes qui dépendent de la position de la lumière.
→ *Preuve : l'ombre se déplace et se déforme en même temps que la main.*
→ Code : `shadows.py`

**Niveau 05 — Multi-lumières**
On suit plusieurs mains pour gérer plusieurs lumières simultanément.
→ *Preuve : plusieurs ombres se croisent et se superposent.*
→ Code : `hand_light.py` (multi-mains)

**Niveau ∞**

**→ Optimisation des performances par multithreading**
- L'estimation de profondeur (MiDaS) tourne sur son propre fil : l'affichage n'attend jamais l'inférence.
- La caméra est lue sur un fil séparé, et le suivi de la main sur un autre.
- **Résultat :** un rendu fluide et à faible latence

**→ Ghost in the Shadow**
Nous transformons tout le pipeline de vision en une application jouable.
- Les cinq niveaux techniques deviennent cinq niveaux de jeu à débloquer.
- On peut manipuler la lumière et voir chaque brique réagir en direct — la preuve visuelle rendue interactive.


---

## Choix Techniques

## Choix Techniques

**Modèle de profondeur**
MiDaS_small en demi-précision (FP16) sur GPU, avec `cudnn.benchmark`.
→ *Pourquoi : meilleur compromis vitesse / qualité pour du temps réel monoculaire.*

**Extraction des normales**
Gradients de la carte de profondeur → `N(x,y)`, affichés en carte RGB visible.
→ *Pourquoi : calcul léger, synchrone à la caméra (Niveau 01).*

**Ré-éclairage**
Diffus Lambertien + reflets spéculaires sur la géométrie reconstruite.
→ *Pourquoi : répond à la preuve du Niveau 02 avec une source lumineuse ponctuelle.*

**Ombres**
Lancer de rayons vers la lumière sur la profondeur, avec pénombre douce et atténuation selon la distance.
→ *Pourquoi : ombres conscientes de la géométrie qui suivent le geste (Niveau 04).*

**Gestes**
MediaPipe Hands (multi-mains) → position de la lumière en **X, Y, Z**.
→ *Pourquoi : contrôle spatial, la profondeur = distance de la main (Niveaux 03 et 05).*

**Filtrage temporel**
Moyenne mobile exponentielle (EMA) sur la profondeur et sur les bornes de normalisation.
→ *Pourquoi : supprime le scintillement d'une image à l'autre.*

**Optimisation des IPS**
Profondeur et caméra sur fils dédiés, calcul 1 image sur 2, tailles d'inférence réduites.
→ *Pourquoi : l'affichage n'attend jamais l'inférence → fluidité et latence maîtrisée.*
---

## Démonstration Interactive — « Ghost in the Shadow »

Pour rendre le pipeline **testable en direct par le jury** (démonstration interactive, Phase 02), le système est habillé d'une couche ludique. Elle ne remplace pas le pipeline : elle le **pilote et le prouve**.

- Le joueur bouge sa main → la **lumière** se déplace en X, Y, Z (Niveau 03).
- La lumière projette une **ombre dynamique** sur la géométrie 3D (Niveau 04).
- Amener l'ombre sur une **cible** valide l'étape de façon visible et mesurable.
- Une progression de 5 niveaux fait monter la complexité (cible mobile, puis multi-lumières → Niveau 05).

C'est la **preuve visuelle rendue interactive** : le jury manipule la lumière et voit, en direct, chaque brique du pipeline réagir.

---

## Structure du Projet

```
repoNRW/
│
├── README.md
├── main.py                      # Point d'entrée officiel (python main.py)
├── requirements.txt             # Liste des dépendances
│
├── /scripts/
│   ├── run_demo.py              # Boucle principale du pipeline temps réel
│   └── test.py
│
├── /src/
│   ├── /depth/
│   │   └── depth_module.py      # MiDaS + fil dédié (DepthEstimator / DepthWorker)
│   ├── /normals/
│   │   └── normalisation.py     # Extraction / normalisation des normales
│   ├── /performance/
│   │   ├── fast_normals.py      # Normales N(x,y) depuis la profondeur   [Niveau 01]
│   │   ├── lighting3d.py        # Ré-éclairage Lambert + spéculaire       [Niveau 02]
│   │   ├── hand_light.py        # Suivi de la main -> lumière X,Y,Z       [Niveau 03/05]
│   │   ├── shadows.py           # Ombres d'occlusion (lancer de rayons)   [Niveau 04]
│   │   ├── point_cloud.py       # Reconstruction 3D
│   │   ├── temporal_filter.py   # Filtrage temporel (EMA)
│   │   ├── latency.py           # Caméra sur fil dédié / mesure de latence
│   │   └── hud.py · benchmark.py · profiling.py
│   └── /game/
│       └── ghost_game.py        # Couche de démonstration interactive
│
└── /webapp/                     # Interface web de démonstration (menu + vue en direct)
    ├── server.py                # Flask : flux vidéo + état JSON + actions
    ├── pipeline.py              # Pipeline temps réel sans fenêtre OpenCV
    ├── game_controller.py       # Progression de la démonstration
    └── /templates/index.html
```

---

## Démarrer & Réutiliser le Code

### Prérequis
- **Python 3.10 – 3.12**
- Une **webcam**
- GPU NVIDIA recommandé (fonctionne aussi sur processeur, plus lentement)

### Installation — Windows / PowerShell

```powershell
git clone <URL_DU_DEPOT>
cd repoNRW

python -m venv venv
.\venv\Scripts\Activate.ps1
# Si l'activation est bloquée :
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned

pip install -r requirements.txt
```

### Installation — Linux / macOS

```bash
git clone <URL_DU_DEPOT>
cd repoNRW

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Lancer

**Version officielle — ligne de commande** 

```bash
python main.py
```
Ouvre la fenêtre du pipeline temps réel complet (profondeur · normales · 3D · ré-éclairage · ombres · gestes).
**`g`** = lancer la démonstration interactive · **`q`** = quitter.

**Version web — vue de démonstration**

```bash
python webapp/server.py
```
Puis ouvrir **http://localhost:5000**.

>  Laisser quelques secondes au modèle MiDaS pour se charger au premier lancement. Autoriser la caméra si le navigateur le demande.


---

