"""
vector_store.py — Interface ChromaDB pour l'indexation et la recherche.

Deux collections :
  - "interventions" : les 30 fiches d'interventions
  - "docs_techniques" : les chunks des fiches PDF

ChromaDB est configuré en mode persistant (stockage sur disque dans chroma_db/).
Imports ChromaDB sont lazy pour accélérer le démarrage.
"""

from pathlib import Path
from .ingestion import Document


COLLECTION_INTERVENTIONS = "interventions"
COLLECTION_DOCS = "docs_techniques"


def get_chroma_client(persist_dir: Path):
    """Crée un client ChromaDB persistant. Import lazy."""
    import chromadb
    from chromadb.config import Settings
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_or_create_collection(client, name: str):
    """Récupère ou crée une collection ChromaDB."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},   # distance cosinus (mieux pour embeddings NLP)
    )


# Indexation

def index_documents(
    documents: list[Document],
    embeddings: list[list[float]],
    chroma_client,
) -> None:
    """
    Indexe les documents dans ChromaDB.
    Sépare automatiquement interventions et docs techniques dans deux collections.
    """
    # Séparer par source
    interventions = [(d, e) for d, e in zip(documents, embeddings) if d.source == "intervention"]
    docs_tech = [(d, e) for d, e in zip(documents, embeddings) if d.source == "doc_technique"]

    _upsert_to_collection(chroma_client, COLLECTION_INTERVENTIONS, interventions)
    _upsert_to_collection(chroma_client, COLLECTION_DOCS, docs_tech)

    print(f"[vector_store] {len(interventions)} interventions indexées")
    print(f"[vector_store] {len(docs_tech)} chunks techniques indexés")


def _upsert_to_collection(client, name: str, pairs: list[tuple[Document, list[float]]]) -> None:
    if not pairs:
        return
    collection = _get_or_create_collection(client, name)
    collection.upsert(
        ids=[d.doc_id for d, _ in pairs],
        embeddings=[e for _, e in pairs],
        documents=[d.content for d, _ in pairs],
        metadatas=[d.metadata for d, _ in pairs],
    )


# ──────────────────────────────────────────────
# Recherche
# ──────────────────────────────────────────────

def search(
    query_embedding: list[float],
    chroma_client,
    n_results: int = 5,
    where_filter: dict | None = None,
) -> list[dict]:
    """
    Recherche dans les deux collections et fusionne les résultats.

    Args:
        query_embedding : vecteur de la question
        n_results       : nombre de résultats par collection
        where_filter    : filtre ChromaDB sur les métadonnées, ex: {"marque": "Frisquet"}

    Returns:
        Liste de dicts triés par distance croissante (plus proche = plus pertinent).
    """
    results = []

    for collection_name in [COLLECTION_INTERVENTIONS, COLLECTION_DOCS]:
        collection = _get_or_create_collection(chroma_client, collection_name)

        # Vérifier qu'il y a des documents dans la collection
        if collection.count() == 0:
            continue

        kwargs = dict(
            query_embeddings=[query_embedding],
            n_results=min(n_results, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        # ChromaDB n'accepte where que si non vide
        if where_filter:
            kwargs["where"] = where_filter

        res = collection.query(**kwargs)

        for doc, meta, dist in zip(
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
        ):
            results.append({
                "content": doc,
                "metadata": meta,
                "distance": dist,
                "score": 1.0 - dist,   # cosine similarity (0→1, plus haut = mieux)
            })

    # Trier par score décroissant (mieux en premier)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def collection_exists_and_populated(chroma_client) -> bool:
    """Vérifie si l'index est déjà construit (pour éviter de re-embedder)."""
    try:
        col = chroma_client.get_collection(COLLECTION_INTERVENTIONS)
        return col.count() > 0
    except Exception:
        return False
