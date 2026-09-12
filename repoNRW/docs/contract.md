
```
Rôle 1 → depth map → Rôle 2 → normal map → Rôle 4
```
## Vue d'ensemble du pipeline

### 📊 Le flux de données

```
┌─────────────────┐
│   CAMÉRA        │
│   Frame RGB     │  ← Image couleur classique
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  RÔLE 1         │
│  DEPTH          │  ← Estime la distance de chaque pixel
│  ESTIMATION     │
└────────┬────────┘
         │
         │  Sortie 1 : Depth map
         ▼
┌─────────────────┐
│  RÔLE 2         │
│  CALCUL DES     │  ← Calcule l'orientation des surfaces
│  NORMALES       │
└────────┬────────┘
         │
         │  Sortie 2 : Normal map
         ▼
┌─────────────────┐
│  RÔLE 4         │
│  VISUALISATION  │  ← Compose et affiche
└────────┬────────┘
         │
         │  Sortie 3 : Composite
         ▼
     

       

### 📋 Les 3 sorties

| Sortie | Producteur | Contenu | Taille |
|---|---|---|---|
| **Sortie 1** | Rôle 1 | Depth map | (480, 640) |
| **Sortie 2** | Rôle 2 | Normal map | (480, 640, 3) |
| **Sortie 3** | Rôle 4 | Composite + info | (480, 1920, 3) |

### 🔗 La règle du `frame_id`

**Toutes les sorties de la même frame DOIVENT avoir le même `frame_id`.**

```
Sortie 1 (frame 42) → frame_id = 42
Sortie 2 (frame 42) → frame_id = 42   ← MÊME numéro
Sortie 3 (frame 42) → frame_id = 42   ← MÊME numéro
```

👉 **C'est ce qui relie les données entre elles.**

---

## 3. Conventions numériques globales

### 🎯 Règle d'or

**Tous les rôles DOIVENT respecter ces conventions.** Sinon, c'est l'échec assuré.

### 📐 Système de coordonnées

On utilise le **système OpenCV** (standard en vision par ordinateur) :

```
             Z (profondeur)
             ↑
             │
             │
             │
             └──────────→ X (droite)
            /
           /
          ↙
         Y (bas)
```

| Axe | Direction | Exemple |
|---|---|---|
| **X** | Vers la droite | +X = droite de l'image |
| **Y** | Vers le bas | +Y = bas de l'image |
| **Z** | Vers la scène | +Z = devant la caméra |



### 🔢 Types de données

| Donnée | Type Python | Plage | Précision |
|---|---|---|---|
| **Frame RGB** | `np.uint8` | 0 à 255 | Entier |
| **Depth map** | `np.float32` | 0.0 à 1.0 | 32 bits |
| **Normal map** | `np.float32` | -1.0 à 1.0 | 32 bits |
| **Matrice K** | `np.float32` | — | 32 bits |
| **Timestamp** | `float` | secondes | 64 bits |

###  Résolution de référence

**Toutes les étapes travaillent en `640 × 480`** (largeur × hauteur).

```
Largeur : 640 pixels
Hauteur : 480 pixels
Canaux  : 3 (RGB) ou 1 (depth)
```

### 🎥 Matrice intrinsèque (K)

**C'est quoi ?** Une matrice 3×3 qui décrit les **paramètres optiques** de la caméra.

```python
K = np.array([
    [fx,  0, cx],
    [ 0, fy, cy],
    [ 0,  0,  1]
], dtype=np.float32)
```

| Paramètre | Signification | Valeur par défaut |
|---|---|---|
| `fx` | Focale horizontale (en pixels) | 500.0 |
| `fy` | Focale verticale (en pixels) | 500.0 |
| `cx` | Centre optique X | 320.0 |
| `cy` | Centre optique Y | 240.0 |

**Analogie** : c'est comme la **carte d'identité** de la caméra. Sans elle, impossible de calculer les positions 3D correctement.

**Valeur par défaut** (à utiliser si aucune caméra spécifique) :

```python
DEFAULT_K = np.array([
    [500.0,   0.0, 320.0],
    [  0.0, 500.0, 240.0],
    [  0.0,   0.0,   1.0]
], dtype=np.float32)
```

---

## 4. Sortie 1 — Depth Estimation

###  Qui la produit ?


### 🎯 Que reçoit-il ?

**Entrée** : une **frame RGB** (image couleur de la caméra)

```
Forme : (480, 640, 3)
Type  : np.uint8
Plage : 0 à 255
```

### 🎯 Que produit-il ?

**Sortie** : une **depth map** + métadonnées

```python
{
    "frame_id": 42,
    "timestamp": 1.400,
    "depth_map": np.ndarray,           # (480, 640) float32
    "imgsz_used": 384
}
```

### 📊 Qu'est-ce qu'une depth map ?

C'est une **image en niveaux de gris** où chaque pixel indique **la distance** entre la caméra et ce point.

```
Image RGB                    Depth map
┌─────────────┐             ┌─────────────┐
│ 🌳  🏠  ☁️  │             │ 0.8 0.5 0.2 │  ← 0.2 = proche
│             │  ──────►    │ 0.9 0.3 0.1 │  ← 0.9 = loin
│  🚗   👤   │             │ 0.2 0.1 0.6 │
└─────────────┘             └─────────────┘
```

**Important** : la profondeur est **relative** (entre 0.0 et 1.0), pas en mètres.

- **0.0** = très proche de la caméra
- **1.0** = très loin

###  Détail des champs

| Champ | Type | Obligatoire | Pourquoi existe-t-il ? |
|---|---|---|---|
| `frame_id` | `int` | ✅ | **Identifier** la frame. Sans lui, impossible de savoir à quelle image appartient cette depth |
| `timestamp` | `float` | ✅ | **Horodater**. Sert à mesurer la latence |
| `depth_map` | `np.ndarray` | ✅ | **La donnée principale** : la carte de profondeur |
| `imgsz_used` | `int` | ✅ | **Quelle résolution** d'inférence a été utilisée. Important pour optimisation |

###  Format attendu

```python
# Forme : (H, W) — 2 dimensions seulement
# PAS de canal !
depth_map.shape == (480, 640)   # ✅ Correct
depth_map.shape == (480, 640, 1) # ❌ Faux

# Type : float32 strict
depth_map.dtype == np.float32   # ✅ Correct
depth_map.dtype == np.float64   # ❌ Faux

# Plage : entre 0.0 et 1.0
0.0 <= depth_map.min()          # ✅
depth_map.max() <= 1.0          # ✅
```

### Exemple concret

Une frame où on voit :
- Un mur au fond (loin) → depth ≈ 0.9
- Une personne au milieu → depth ≈ 0.5
- Un objet devant la caméra → depth ≈ 0.1

---

## 5. Sortie 2 — Normales

### 🎯 Qui la produit ?

**Rôle 2** (l'équipe normales)

### 🎯 Que reçoit-il ?

**Entrée** : la **depth map** produite par Rôle 1

```
Forme : (480, 640)
Type  : np.float32
Plage : 0.0 à 1.0
```

###  Que produit-il ?

**Sortie** : une **normal map** + matrice K

```python
{
    "frame_id": 42,
    "timestamp": 1.425,
    "normal_map": np.ndarray,   # (480, 640, 3) float32
    "K": np.ndarray             # (3, 3) float32
}
```

### 📊 Qu'est-ce qu'une normal map ?

C'est une **image 3 canaux** où chaque pixel indique la **direction 3D** de la surface à cet endroit.

**Analogie simple** :
- Imagine une **colline**. À chaque point, la surface est **penchée** dans une direction.
- Le **vecteur normal** est la direction **perpendiculaire** à la surface.
- Il **pointe** dans la direction "qui sort de la surface".

```
Vue de profil d'une colline :

        N (normale)
        ↑
        │
        │
    ╱───┴───╲
   ╱         ╲    ← Surface de la colline
  ╱           ╲
```

### 📋 Détail des champs

| Champ | Type | Obligatoire | Pourquoi existe-t-il ? |
|---|---|---|---|
| `frame_id` | `int` | ✅ | **Doit être identique** à la sortie 1 → relie les données |
| `timestamp` | `float` | ✅ | Postérieur à la sortie 1 → mesure la latence |
| `normal_map` | `np.ndarray` | ✅ | **La donnée principale** : les normales |
| `K` | `np.ndarray` | ✅ | **Matrice caméra** utilisée → permet de reproduire le calcul |

### 📐 Format attendu

```python
# Forme : (H, W, 3) — 3 canaux !
normal_map.shape == (480, 640, 3)   # ✅

# Type : float32
normal_map.dtype == np.float32

# Plage : entre -1.0 et 1.0
-1.0 <= normal_map.min()
normal_map.max() <= 1.0

# Chaque vecteur est NORMALISÉ (norme = 1)
norms = np.linalg.norm(normal_map, axis=-1)
np.allclose(norms, 1.0)   # ✅ Toutes les normes valent 1.0
```

### 🎨 Visualisation

Quand on encode les normales en couleur RGB :

```python
R = (Nx + 1) / 2 × 255
G = (Ny + 1) / 2 × 255
B = (Nz + 1) / 2 × 255
```

**Résultats typiques** :

| Surface | Normale | Couleur RGB |
|---|---|---|
| **Face caméra** | (0, 0, 1) | 🔵 Bleu clair (128, 128, 255) |
| **Sol plat** | (0, -1, 0) | 🟣 Violet (128, 0, 128) |
| **Mur à gauche** | (1, 0, 0) | 🔴 Rouge (255, 128, 128) |
| **Mur à droite** | (-1, 0, 0) | 🟢 Vert (0, 128, 128) |

### 🎯 Exemple concret

Pour une scène avec :
- Un **sol** : normales ≈ (0, -1, 0) → violet
- Un **mur face caméra** : normales ≈ (0, 0, 1) → bleu clair
- Un **objet sphérique** : normales variées → toutes les couleurs

---

## 6. Sortie 3 — Visualisation

###  Qui la produit ?

###  Que reçoit-il ?

**Entrées** :
- La frame RGB (de la caméra)
- La depth map (de Rôle 1)
- La normal map (de Rôle 2)

###  Que produit-il ?

**Sortie** : une **image composite** + infos de performance

```python
{
    "frame_id": 42,
    "composite": np.ndarray,        # (480, 1920, 3) uint8
    "normal_map_rgb": np.ndarray,   # (480, 640, 3) uint8
    "info": {
        "fps": 25.3,
        "latency_ms": 39.5,
        "mode": "normal",
        "depth_ms": 32.1,
        "normals_ms": 7.4
    }
}
```

###  Qu'est-ce que le composite ?

Une image qui montre **3 vues côte-à-côte** :

```
┌─────────────┬─────────────┬─────────────┐
│             │             │             │
│   RGB       │   Depth     │  Normales   │
│  (original) │  (gris)     │  (couleur)  │
│             │             │             │
│  640×480    │  640×480    │  640×480    │
└─────────────┴─────────────┴─────────────┘
             Total : 1920×480
```

**Pourquoi 3 vues ?** Pour que le jury voie **l'entrée** (RGB) et **les 2 sorties** (depth + normales) en même temps. C'est visuellement impressionnant.

### 📋 Détail des champs

| Champ | Type | Pourquoi existe-t-il ? |
|---|---|---|
| `composite` | `np.ndarray` | L'image finale pour l'affichage |
| `normal_map_rgb` | `np.ndarray` | La normal map encodée en RGB (pour debug) |
| `info.fps` | `float` | **Critère du challenge** : FPS mesuré |
| `info.latency_ms` | `float` | **Critère du challenge** : latence |
| `info.mode` | `str` | Mode actuel (normal/degraded/minimal) |
| `info.depth_ms` | `float` | **Temps du depth** → pour optimisation |
| `info.normals_ms` | `float` | **Temps des normales** → pour optimisation |

###  Pourquoi les temps séparés ?

Si `depth_ms = 80` et `normals_ms = 5` :
- On sait que le **goulot est le depth**
- On peut **cibler l'optimisation**

Sans ces temps, on saurait juste "le pipeline prend 85 ms" et on ne saurait pas où agir.

---

## 7. Cas limites


###  Catégorie 1 — Aucune inférence possible

**Cas** : la caméra renvoie une frame noire, ou le modèle plante.

**Comportement attendu** :

```python
depth_map = np.full((480, 640), 0.5, dtype=np.float32)
# → depth map uniforme à 0.5 (valeur "neutre")
```

**Pourquoi 0.5 ?** Parce que c'est la **valeur médiane** entre proche (0) et loin (1). Ni trop proche, ni trop loin.

###  Catégorie 2 — Normales dégénérées

**Cas** : la depth map est **uniforme** (tous les pixels à 0.5).

**Comportement** :

```python
normal_map = np.zeros((480, 640, 3), dtype=np.float32)
normal_map[:, :, 2] = 1.0
# → toutes les normales pointent vers la caméra (0, 0, 1)
```

**Pourquoi ?** Parce que si toutes les profondeurs sont égales, la surface est "plate face caméra".

###  Catégorie 3 — NaN / Inf

**Cas** : la depth map contient des `NaN` ou `Inf` (bug de calcul).

**Comportement** :

```python
# Remplacer les valeurs invalides AVANT tout traitement
depth_map = np.nan_to_num(depth_map, nan=0.5, posinf=1.0, neginf=0.0)
```

**Règle absolue** : **jamais de NaN ou Inf dans les sorties.**

###  Catégorie 4 — Division par zéro

**Cas** : calcul de norme pour normaliser une normale.

**Comportement** :

```python
# Clamp la norme à une valeur minimale
norms = np.linalg.norm(N, axis=-1, keepdims=True)
norms = np.maximum(norms, 1e-8)   # ← évite division par 0
N = N / norms
```

###  Catégorie 5 — Frame sautée

**Cas** : mode dégradé actif → frame 43 sautée.

**Comportement** : `frame_id` a des **trous**.

```
Frame 42 : {...}
Frame 44 : {...}   ← 43 sautée
```

**Règle** : **ne PAS renuméroter** les frames. Le `frame_id` reflète la **vraie** frame.

###  Catégorie 6 — Exception dans un module

**Cas** : une erreur inattendue dans un module.

**Règle** : **AUCUNE exception ne doit sortir d'un module**.

```python
try:
    result = process(frame)
except Exception as e:
    log_error(e)
    result = fallback()   # ← retourner quelque chose de valide
```

**Pourquoi ?** Parce qu'une exception non attrapée **tue tout le pipeline**.

---

## 8. Budget de performance

###  Cible globale

| Métrique | Cible | Minimum acceptable |
|---|---|---|
| **FPS** | ≥ 15 | ≥ 10 |
| **Latence moyenne** | < 100 ms | < 150 ms |
| **Latence p95** | < 130 ms | < 200 ms |
| **Latence p99** | < 200 ms | < 300 ms |

###  Budget par module

| Module | Budget | Responsable |
|---|---|---|
| **Depth estimation** | < 80 ms | Rôle 1 |
| **Calcul normales** | < 15 ms | Rôle 2 |
| **Visualisation** | < 10 ms | Rôle 4 |
| **Marge** | ~5 ms | — |
| **TOTAL** | **< 110 ms** | — |

###  Le goulot principal

Le **depth estimation** est le goulot. Il prend ~80% du temps total.

**Conséquence** : c'est **lui** qu'il faut optimiser en priorité :
- Utiliser un modèle **small** (pas large)
- Réduire la **résolution** (384 → 256)
- Exporter en **ONNX** si possible

---

## 9. Décisions verrouillées

 **Ces décisions NE DOIVENT PAS changer** sans prévenir toute l'équipe.

| Décision | Valeur | Raison |
|---|---|---|
| **Type depth** | `np.float32` | Standard ML |
| **Plage depth** | [0.0, 1.0] | Normalisé |
| **Type normales** | `np.float32` | Standard ML |
| **Plage normales** | [-1.0, 1.0] | Convention math |
| **Norme normales** | 1.0 | Vecteurs unitaires |
| **Système coords** | OpenCV (X→droite, Y→bas, Z→avant) | Standard vision |
| **Résolution** | 640 × 480 | Standard hackathon |
| **Fréquence** | 1 sortie par frame | Simple |
| **Encodage RGB** | `(N + 1) / 2 × 255` | Standard normal map |
| **K par défaut** | fx=fy=500, cx=320, cy=240 | Approximation caméra standard |

---

