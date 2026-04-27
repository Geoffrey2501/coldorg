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

## Pistes d'amélioration

### Implémentée : Re-ranking hybride
Score vectoriel + bonus sur code erreur exact → meilleure précision sur les pannes spécifiques.

### Piste 2 : Filtrage par plage de dates
Pour les équipements récents, prioriser les interventions récentes. Utile quand les équipements évoluent (nouveaux firmwares, nouveaux composants).

### Piste 3 : Reformulation de requête (HyDE)
Générer une "réponse hypothétique" avant l'embedding pour améliorer le retrieval sur des questions ambiguës. Efficace sur Q5 (aucun code erreur, symptôme vague).
