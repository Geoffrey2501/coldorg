"""
retriever.py — Logique de recherche (retrieval) avec extraction d'entités et re-ranking.
"""

from __future__ import annotations  # Indispensable pour utiliser chromadb dans les types sans NameError

import re
from typing import TYPE_CHECKING, List, Dict, Tuple, Optional

# Imports utilisés uniquement pour le typage (statique)
if TYPE_CHECKING:
    from mistralai.client import Mistral
    import chromadb

from .embeddings import embed_query
from .vector_store import search


# ──────────────────────────────────────────────
# Configuration & Extraction d'entités
# ──────────────────────────────────────────────

# Liste étendue des marques pour couvrir le dataset
KNOWN_BRANDS = [
    "Frisquet", "Daikin", "Saunier Duval", "Atlantic",
    "Mitsubishi", "Viessmann", "Aldes", "Thermor", "Chaffoteaux", "Viessmann"
]

# Regex plus robuste pour les codes erreur (ex: E133, F.28, U4, AL05)
# Supporte les formats : Lettre+Chiffres, Chiffre+Lettre, ou Lettre+Point+Chiffre
_ERROR_CODE_RE = re.compile(r'\b([A-Z][0-9]{1,3}|[0-9][A-Z]|[A-Z]\.[0-9]{1,2})\b', re.IGNORECASE)


def extract_entities(question: str) -> Dict[str, Optional[str]]:
    """
    Extrait la marque et le code erreur de la question de manière plus précise.
    """
    found_brand = None
    # Recherche de marque avec frontières de mots pour éviter les faux positifs (ex: "Al" dans "Alarme")
    for brand in KNOWN_BRANDS:
        if re.search(rf"\b{re.escape(brand)}\b", question, re.IGNORECASE):
            found_brand = brand
            break

    found_code = None
    match = _ERROR_CODE_RE.search(question)
    if match:
        found_code = match.group(1).upper()

    return {
        "marque": found_brand,
        "code_erreur": found_code
    }


# ──────────────────────────────────────────────
# Re-ranking Hybride (Vecteur + Métadonnées)
# ──────────────────────────────────────────────

def rerank(results: List[Dict], entities: Dict[str, str], top_k: int = 5) -> List[Dict]:
    """
    Ajuste le score des résultats basés sur la similarité cosinus avec des bonus métier.

    Bonus :
    - +0.20 : Match exact du code erreur (priorité absolue au dépannage)
    - +0.10 : Match de la marque
    """
    target_code = (entities.get("code_erreur") or "").upper()
    target_brand = (entities.get("marque") or "").lower()

    for r in results:
        meta = r.get("metadata", {})
        score = r.get("score", 0.0)
        bonus = 0.0

        # Vérification du code erreur dans les métadonnées
        doc_code = str(meta.get("code_erreur", "")).upper()
        if target_code and target_code == doc_code:
            bonus += 0.20

        # Vérification de la marque
        doc_brand = str(meta.get("marque", "")).lower()
        if target_brand and target_brand == doc_brand:
            bonus += 0.10

        r["rerank_score"] = score + bonus

    # Tri par nouveau score décroissant
    results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return results[:top_k]


# ──────────────────────────────────────────────
# Fonction principale de Retrieval
# ──────────────────────────────────────────────

def retrieve(
    question: str,
    chroma_client: chromadb.PersistentClient,
    mistral_client: Mistral,
    n_candidates: int = 10,
    top_k: int = 5,
    use_metadata_filter: bool = True,
) -> Tuple[List[Dict], Dict]:
    """
    Exécute le pipeline complet : Extraction -> Recherche Vectorielle -> Re-ranking.
    """
    # 1. Analyse de la question
    entities = extract_entities(question)

    # 2. Vectorisation via Mistral
    query_vec = embed_query(question, client=mistral_client)

    # 3. Préparation du filtre ChromaDB
    where_filter = None
    if use_metadata_filter and entities["marque"]:
        where_filter = {"marque": entities["marque"]}

    # 4. Recherche dans les collections (interventions + docs)
    results = search(
        query_embedding=query_vec,
        chroma_client=chroma_client,
        n_results=n_candidates,
        where_filter=where_filter,
    )

    # 5. Stratégie de repli : si le filtre marque est trop restrictif, on élargit la recherche
    if where_filter and len(results) < 2:
        results = search(
            query_embedding=query_vec,
            chroma_client=chroma_client,
            n_results=n_candidates,
            where_filter=None,
        )

    # 6. Re-ranking final pour privilégier la précision technique
    ranked_results = rerank(results, entities, top_k=top_k)

    return ranked_results, entities