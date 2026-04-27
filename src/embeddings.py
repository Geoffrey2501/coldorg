"""
embeddings.py — Génération d'embeddings via l'API Mistral (mistral-embed).

Fonctionnalités :
  - Embedding par batch (limite API : 2048 tokens par requête, on envoie par lots)
  - Dimensions : 1024 (mistral-embed)
  - Imports lourds (mistralai) chargés à la première utilisation
"""

import os
import time

EMBED_MODEL = "mistral-embed"
BATCH_SIZE = 32   # nombre de textes envoyés par appel API


def get_client():
    """Crée et retourne un client Mistral. Import lazy."""
    from mistralai.client import Mistral
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Variable MISTRAL_API_KEY non définie. "
            "Copie .env en .env et renseigne ta clé."
        )
    return Mistral(api_key=api_key)


def embed_texts(texts: list[str], client=None) -> list[list[float]]:
    """
    Génère les embeddings pour une liste de textes.
    Retourne une liste de vecteurs float (1024 dimensions).
    Envoie les requêtes par batch pour respecter les limites API.
    """
    if client is None:
        client = get_client()

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i: i + BATCH_SIZE]

        # Retry simple en cas d'erreur rate-limit
        for attempt in range(3):
            try:
                response = client.embeddings.create(
                    model=EMBED_MODEL,
                    inputs=batch,
                )
                all_embeddings.extend([e.embedding for e in response.data])
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                print(f"[embeddings] Erreur API (tentative {attempt+1}/3) : {exc}")
                time.sleep(2 ** attempt)

        # Petite pause entre les batches pour éviter le rate-limiting
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.2)

    return all_embeddings


def embed_query(query: str, client=None) -> list[float]:
    """Embedding d'une question unique."""
    return embed_texts([query], client)[0]
