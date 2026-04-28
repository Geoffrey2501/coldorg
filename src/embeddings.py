"""
embeddings.py — Génération d'embeddings via l'API Mistral (mistral-embed).

Fonctionnalités :
  - Embedding par batch (limite API : 2048 tokens par requête, on envoie par lots)
  - Dimensions : 1024 (mistral-embed)
  - Imports lourds (mistralai) chargés à la première utilisation
"""

import os
import time
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "mistral-embed"
LOCAL_EMBED_MODEL = "all-MiniLM-L6-v2"  # Modèle léger et performant
BATCH_SIZE = 32


def get_client():
	"""Retourne le client Mistral ou None si la clé est absente."""
	api_key = os.environ.get("MISTRAL_API_KEY")
	if not api_key:
		print("[embeddings] MISTRAL_API_KEY non trouvée. Utilisation du mode local.")
		return None

	from mistralai.client import Mistral
	return Mistral(api_key=api_key)


def embed_texts(texts: list[str], client=None) -> list[list[float]]:
	"""Génère les embeddings via Mistral ou Sentence-Transformers en local."""
	if client is None:
		client = get_client()

	# MODE LOCAL (Fallback)
	if client is None:
		model = SentenceTransformer(LOCAL_EMBED_MODEL)
		embeddings = model.encode(texts)
		return embeddings.tolist()

	# MODE API MISTRAL
	all_embeddings: list[list[float]] = []
	for i in range(0, len(texts), BATCH_SIZE):
		batch = texts[i: i + BATCH_SIZE]
		for attempt in range(3):
			try:
				response = client.embeddings.create(model=EMBED_MODEL, inputs=batch)
				all_embeddings.extend([e.embedding for e in response.data])
				break
			except Exception as exc:
				if attempt == 2: raise
				time.sleep(2 ** attempt)
		if i + BATCH_SIZE < len(texts):
			time.sleep(0.2)
	return all_embeddings


def embed_query(query: str, client=None) -> list[float]:
	return embed_texts([query], client)[0]