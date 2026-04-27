# COLDORG RAG — Assistant Technicien CVC

Prototype de système RAG (Retrieval-Augmented Generation) permettant aux techniciens de diagnostiquer des pannes à partir de l'historique d'interventions et de la documentation technique.

---

## Stack technique

| Composant | Choix | Justification |
|-----------|-------|---------------|
| **LLM** | `mistral-small-latest` | Bon rapport qualité/coût, excellent français |
| **Embeddings** | `mistral-embed` (1024 dim) | Meilleur embedding Mistral disponible |
| **Base vectorielle** | ChromaDB (persistant) | Embarqué, filtrage natif par métadonnées, pas de serveur |
| **Framework RAG** | From scratch | Code transparent, pas de boîte noire |

J'ai choisi d'utiliser mistral, car il ne stocke pas les données et j'ai l'habitude 
d'utiliser leurs solutions.

---

## Architecture du projet

```
coldorg-rag/
├── src/
│   ├── ingestion.py       # Chargement + chunking (JSON & TXT)
│   ├── embeddings.py      # Wrapper mistral-embed (batch + retry)
│   ├── vector_store.py    # Interface ChromaDB (2 collections)
│   ├── retriever.py       # Retrieval + extraction entités + re-ranking
│   ├── generator.py       # Prompt engineering + appel Mistral
│   └── rag_pipeline.py    # Orchestrateur principal
├── data/                  # Données fournies (inchangées)
├── chroma_db/             # Index vectoriel persisté (gitignored)
├── evaluate.py            # Test sur les 5 questions
├── main.py                # CLI interactive multi-turn
├── requirements.txt
└── .env.example
```

---

## Lancement

### 1. Installation

```bash
cd coldorg-rag
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

### 2. Configuration de la clé API

```bash
cp .env .env
# Édite .env et renseigne ta clé MISTRAL_API_KEY
```

### 3. Évaluation sur les 5 questions de test

```bash
python evaluate.py
```

> La première exécution construit l'index (~2 min selon la connexion). Les suivantes sont instantanées grâce au cache ChromaDB.

### 4. CLI interactive (bonus multi-turn)

```bash
python main.py
```

---

## Choix de conception

### Chunking

**Interventions JSON → 1 document par intervention**
Les fiches d'interventions sont déjà des unités atomiques (~150 tokens). Les fragmenter briserait le lien entre symptôme, diagnostic et solution. Chaque document hérite des métadonnées : `marque`, `type_equipement`, `code_erreur`, `date`, `difficulte`.

**Fiches techniques TXT → découpage par sections**
Les fiches sont découpées sur les titres en majuscules et les séparateurs. Chunk size ~400 tokens avec 50 tokens d'overlap pour éviter de couper des procédures importantes. Chaque chunk hérite des métadonnées de la fiche (marque, type d'équipement).

Ces deux découpages permet au model de recouper les informations et de completer sa reponse afin d'éviter l'hallucination.

**Deux collections ChromaDB séparées**
Interventions et docs techniques dans deux collections distinctes, pour permettre un filtrage indépendant et un meilleur contrôle de la diversité des sources retournées.

### Gestion des métadonnées

Chaque document indexé porte les champs :

| Métadonnée | Interventions | Docs techniques |
|-----------|---------------|-----------------|
| `marque` | ✓ (ex: "Frisquet") | ✓ (ex: "Daikin") |
| `type_equipement` | ✓ (ex: "pac_air_eau") | ✓ |
| `code_erreur` | ✓ (ex: "E133") | — |
| `source` | "intervention" | "doc_technique" |
| `date` | ✓ | — |
| `difficulte` | ✓ | — |

Le filtrage par marque est activé automatiquement quand la question mentionne une marque connue. Cela divise l'espace de recherche et améliore la précision.

### Pipeline de retrieval

```
Question → Extraction entités (regex) → Filtrage métadonnées ChromaDB
       → Recherche vectorielle (top-8) → Re-ranking hybride → top-5
```

**Re-ranking hybride (amélioration implémentée)** :
- Score de base : cosine similarity (embedding)
- Bonus +0.15 : si le code erreur correspond exactement
- Bonus +0.05 : si la marque correspond

Cela améliore significativement Q4 (F28 récurrent sur Saunier Duval) où le match exact du code erreur est plus discriminant que la similarité sémantique seule.

---

## Ce qu'on ferait avec 10 000 interventions

| Aspect | Actuel (30) | À 10 000 interventions |
|--------|-------------|----------------------|
| **Stockage** | ChromaDB local | ChromaDB server |
| **Ingestion** | Script one-shot | Pipeline batch + ingestion incrémentale (CDC) |
| **Chunking** | Taille fixe | Chunking adaptatif selon longueur et structure |
| **Retrieval** | top-5 flat | Retrieval hiérarchique : filtrer d'abord par marque/type, puis vectoriel |
| **Évaluation** | 5 questions manuelles | Benchmark RAGAS automatisé (faithfulness, answer relevance) |
| **Monitoring** | Aucun | Logging des requêtes, latences, scores de retrieval |
| **Déduplication** | N/A | Near-duplicate detection avant indexation |
| **Mise à jour** | Rebuild complet | Upsert incrémental par ID |

---

## Résultat
### Question 1 :

Documents :

INT-011

INT-001

INT-029

DOC-fiche_frisquet_prestige

INT-007

Causes possibles :

1 Problème d'arrivée de gaz :
• Robinet gaz fermé (sous la chaudière ou au compteur).
• Pression gaz insuffisante (moins de 20 mbar pour G20).
• Compteur gaz bloqué ou compteur électronique en défaut.

2 Électrode d'allumage :
• Encrassée ou écart incorrect (doit être à 3 mm).
• Fil de masse desserré sur le boîtier de contrôle.

3 Problème d'ionisation :
• Courant d'ionisation < 1 microA (vérifier avec multimètre).

4 Carte électronique défaillante (rare, à vérifier en dernier recours).

Étapes de diagnostic :

1 Vérifier l'arrivée de gaz :
• Contrôler que le robinet gaz sous la chaudière est ouvert.
• Vérifier que le compteur gaz n'est pas bloqué (compteur mécanique ou électronique).
• Mesurer la pression gaz en entrée (doit être ≥ 20 mbar pour G20). Si < 20 mbar, contacter GRDF.

2 Inspecter l'électrode d'allumage :
• Nettoyer l'électrode avec un chiffon sec ou une brosse douce.
• Vérifier l'écart entre l'électrode et la veilleuse (doit être à 3 mm).
• Contrôler le fil de masse sur le boîtier de contrôle (resserrer si nécessaire).

3 Mesurer le courant d'ionisation :
• Utiliser un multimètre en mode microampères.
• Brancher sur la sonde d'ionisation (doit afficher > 1 µA).

4 Tester la carte électronique :
• Si toutes les étapes précédentes sont OK, suspecter la carte (remplacement si disponible).

Pièces à prévoir :

• Chiffon microfibre / brosse douce (pour nettoyer l'électrode).
• Clé plate (pour resserrer le fil de masse).
• Manomètre gaz (pour vérifier la pression).
• Multimètre (pour mesurer le courant d'ionisation).
• Électrode d'allumage de rechange (si doute sur son état).

Interventions similaires dans l'historique :

• INT-011 (pression gaz trop basse, contact GRDF).
• INT-001 (électrode encrassée et fil de masse desserré).
• INT-029 (robinet gaz fermé).       

### Question 2
Documents :

DOC-fiche_atlantic_climatisation

INT-003

INT-013

INT-024

Causes possibles :

1 Bac de condensat obstrué (algues, poussière)
2 Tuyau d'évacuation bouché ou mal penté
3 Pompe de relevage en panne (si installée)
4 Unité intérieure non de niveau (débordement du bac)

Étapes de diagnostic :

1 Vérifier visuellement l'unité intérieure : est-elle de niveau ? (utiliser un niveau à bulle)
2 Inspecter le bac de condensat : enlever la grille frontale et vérifier l'état du bac (présence d'algues, de poussière ou d'eau stagnante)
3 Contrôler le tuyau d'évacuation :
• Débrancher le tuyau et souffler dedans pour vérifier l'obstruction
• Vérifier la pente (minimum 1% en direction de l'évacuation)
4 Tester la pompe de relevage (si présente) : vérifier son fonctionnement et son alimentation électrique
5 Ajouter une pastille anti-algues dans le bac de condensat si nécessaire

Pièces à prévoir :

• Pastille anti-algues (référence standard pour Atlantic Idéa)
• Pompe de relevage de rechange (si nécessaire, vérifier le modèle)
• Tuyau d'évacuation flexible (si remplacement nécessaire)
• Kit de nettoyage pour bac de condensat (brosse douce, produit désinfectant)

Interventions similaires dans l'historique :

• INT-003 (Fuite d'eau unité intérieure : bac de condensat obstrué, tuyau bouché, mauvaise pente)

### Question 3
Documents :

INT-002

DOC-fiche_daikin_altherma

INT-017

INT-005

INT-026


Causes possibles :

1 Câble de communication coupé, endommagé ou mal branché (rongeurs, écrasement, etc.)
2 Bornier de connexion oxydé sur l'unité intérieure ou extérieure
3 Carte électronique (intérieure ou extérieure) défaillante

Étapes de diagnostic :

1 Vérifier le câble de communication :
• Contrôler visuellement l'état du câble (4 fils) dans la gaine technique (rechercher traces de rongeurs, coupures, écrasements).
• Tester la continuité des 4 fils avec un multimètre (résistance < 1Ω par fil).
• Vérifier les connexions aux borniers des deux unités (serrage, oxydation).
2 Tester la tension de communication :
• Mesurer la tension entre les bornes de communication (bornes désignées dans la doc technique, env. 30V DC).
• Si tension absente ou anormale → câble ou carte défectueuse.
3 Inspecter les borniers :
• Nettoyer les contacts oxydés avec une brosse métallique ou contact cleaner.
4 Tester les cartes électroniques :
• Si câble et borniers OK, tester en inversant les cartes (si possible) pour isoler la défaillance.

Pièces à prévoir :

• Câble de communication 4x0.75mm² (longueur adaptée à l'installation, ex: 8m si similaire à INT-002).
• Gaine anti-rongeurs (type ICTA 20mm ou équivalent).
• Optionnel :
• Borniers de rechange (si oxydation sévère).
• Carte électronique intérieure/extérieure (références à vérifier selon modèle exact, ex: carte INT-EXT Daikin réf. 50XXXX).

Interventions similaires dans l'historique :

• INT-002 (code U4, solution câble + gaine anti-rongeurs).
• INT-005 (code E7, moteur ventilateur grippé) → non pertinent ici.
• INT-017 (code AH, pompe ECS) → non pertinent ici.

Remarque : Si le câble est intact et les borniers propres, prévoir aussi un test de tension de communication (30V DC) pour confirmer la défaillance carte. Dans ce cas, commander la carte concernée (intérieure ou extérieure) en fonction du diagnostic.

### Question 4 
Documents :

INT-023

INT-004

DOC-fiche_saunier_duval_themaplus

INT-016

Causes possibles :

1 Carte électronique défaillante (relais d'allumage intermittent) – déjà identifié dans INT-023.
2 Électrovanne gaz fatiguée (courant de commande faible, ionisation instable) – déjà identifié dans INT-004.
3 Électrode d'allumage/ionisation usée ou encrassée (courant d'ionisation < 1 µA).
4 Câblage de l'électrode défectueux (oxydation, faux contact).
5 Problème d'arrivée de gaz (pression insuffisante, robinet partiellement fermé).
6 Oxydation des connecteurs (carte électronique ou électrovanne).

Étapes de diagnostic :

1 Vérifier l'arrivée de gaz :
• Contrôler la pression (doit être ≥ 20 mbar).
• Vérifier que le robinet gaz est complètement ouvert.
2 Inspecter l'électrode d'allumage/ionisation :
• Nettoyer l'électrode (encrassement possible).
• Vérifier la position (écart idéal : 3-4 mm de la veilleuse).
• Mesurer le courant d'ionisation (doit être > 1 µA).
3 Tester l'électrovanne gaz :
• Vérifier la tension de commande (230V).
• Contrôler le courant de commande (doit être stable).
4 Vérifier les connecteurs :
• Nettoyer les connecteurs de la carte électronique et de l'électrovanne (traces d'oxydation).
5 Remplacer la carte électronique si les tests précédents sont OK (comme dans INT-023).

Pièces à prévoir :

• Carte électronique (réf. 0020049194) – si défaut persistant après nettoyage.
• Électrovanne gaz combinée (réf. 05743600) – si courant d'ionisation instable.
• Électrode d'allumage/ionisation (pièce d'usure, à remplacer si usée).
• Multimètre (pour mesurer courant d'ionisation et tension).
• Nettoyant contact électrique (pour oxydation des connecteurs).

Interventions similaires dans l'historique :

• INT-023 (carte électronique défaillante).
• INT-004 (électrovanne gaz fatiguée).

### Question 5
Documents :

DOC-fiche_daikin_altherma

INT-002

INT-026

INT-005

INT-017



Causes possibles :

1 Déséquilibre hydraulique (débit d’eau insuffisant dans les radiateurs/plancher chauffant).
2 Température de consigne trop basse (réglage incorrect sur la télécommande ou le thermostat).
3 Pompe à eau en mode dégradé (bruit anormal, vibration faible, mais pas de code erreur affiché).
4 Vanne mélangeuse bloquée (si PAC avec gestion de zones).
5 Air dans le circuit hydraulique (purge nécessaire).
6 Sonde de température défectueuse (unité intérieure ou extérieure).
7 Problème de régulation (carte électronique intérieure défaillante sans déclencher de code).

Étapes de diagnostic :

1 Vérifier la température de consigne :
• Sur la télécommande de la PAC, confirmer que la consigne est bien réglée (ex: 20°C minimum pour un chauffage confort).
• Vérifier que le mode "Chauffage" est activé (pas en mode "Éco" ou "Absence").
2 Contrôler le débit d’eau :
• Mesurer la température de départ/retour de l’eau sur l’unité intérieure (différence normale : 5-10°C).
• Si différence < 3°C → débit insuffisant (pompe à vérifier, filtre colmaté).
• Si différence > 15°C → débit trop élevé (vanne de réglage à ajuster).
3 Inspecter la pompe à eau :
• Écouter si la pompe tourne (bruit de vibration normal).
• Si silence ou bruit anormal → pompe bloquée (calcaire) ou carte défaillante.
• Référence utile : Voir INT-017 (AH) pour procédure de nettoyage/déblocage.
4 Vérifier l’équilibrage du circuit :
• Si PAC avec plusieurs zones : s’assurer que les vannes sont ouvertes et réglées.
• Contrôler la pression du circuit (manomètre sur l’unité intérieure, normale : 1-2 bars à froid).
5 Purger l’air du circuit :
• Purger les radiateurs/plancher chauffant (même sans code erreur, l’air peut bloquer la circulation).
• Vérifier le purgeur automatique (si présent) sur l’unité intérieure.
6 Tester les sondes de température :
• Mesurer la résistance des sondes (références dans la doc technique Daikin) ou les remplacer si doute.
• Vérifier que les valeurs affichées sur la télécommande correspondent aux mesures réelles.
7 Contrôler la vanne mélangeuse (si applicable) :
• Si la PAC gère plusieurs zones, vérifier que la vanne n’est pas bloquée en position fermée.
8 Vérifier la régulation :
• Redémarrer la PAC (coupure secteur 30 sec) pour réinitialiser la carte électronique.
• Si le problème persiste, tester avec une autre télécommande (si disponible).

Pièces à prévoir :

• Pompe à eau de rechange (réf. Daikin selon modèle, ex: 5018003 pour 8kW).
• Filtre à eau (si colmatage probable, réf. 5007955).
• Joint de bride (pour accès pompe/ballon, réf. 5007954).
• Clé de purge (pour radiateurs/plancher).
• Thermomètre infrarouge (pour mesurer les températures).

Interventions similaires dans l'historique :

• INT-017 (AH) : Défaut pompe ECS → utile pour la procédure de nettoyage.
• INT-002 (U4) : Problème de communication → si suspicion de carte électronique défaillante (même sans code erreur).

## Pistes d'amélioration

### Implémentée : Re-ranking hybride
Score vectoriel + bonus sur code erreur exact → meilleure précision sur les pannes spécifiques.

### Piste 2 : Filtrage par plage de dates
Pour les équipements récents, prioriser les interventions récentes. Utile quand les équipements évoluent (nouveaux firmwares, nouveaux composants).

### Piste 3 : Reformulation de requête (HyDE)
Générer une "réponse hypothétique" avant l'embedding pour améliorer le retrieval sur des questions ambiguës. Efficace sur Q5 (aucun code erreur, symptôme vague).
