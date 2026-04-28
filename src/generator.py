"""
generator.py — Génération de la réponse via mistral-small-latest.

Le prompt système positionne le LLM comme un assistant technicien CVC expert.
La réponse est demandée en format structuré Markdown.
"""

SYSTEM_PROMPT = """Tu es un assistant expert en maintenance CVC (Chauffage, Ventilation, Climatisation) pour la société COLDORG.
Tu aides les techniciens de terrain à diagnostiquer et résoudre des pannes.

Tes réponses doivent être :
- Pratiques et directement exploitables sur le terrain
- Structurées et claires
- Basées UNIQUEMENT sur les documents fournis dans le contexte
- En français

Si le contexte ne contient pas suffisamment d'informations, dis-le explicitement plutôt que d'inventer.

Format de réponse attendu :
**Causes possibles :**
1. ...

**Étapes de diagnostic :**
1. ...

**Pièces à prévoir :**
- ...

**Interventions similaires dans l'historique :** (IDs des fiches pertinentes, ex: INT-001, INT-011)
"""
import os

GENERATION_MODEL = "mistral-small-latest"
FALLBACK_MODEL = "llama3"  # Modèle pour Ollama
"""
   Génère une réponse RAG structurée.

   Args:
       question          : question du technicien
       retrieved_docs    : documents récupérés par le retriever
       mistral_client    : client Mistral
       conversation_history : historique de conversation pour le multi-turn
                             (liste de {"role": "user"|"assistant", "content": "..."})

   Returns:
       Réponse générée sous forme de texte Markdown.
   """

def generate_answer(
		question: str,
		retrieved_docs: list[dict],
		mistral_client,
		conversation_history: list[dict] | None = None,
) -> str:
	"""Génère une réponse avec fallback local via Ollama."""
	context = build_context(retrieved_docs)
	user_message = f"CONTEXTE :\n{context}\n\nQUESTION : {question}"

	messages = [{"role": "system", "content": SYSTEM_PROMPT}]
	if conversation_history:
		messages.extend(conversation_history[-4:])
	messages.append({"role": "user", "content": user_message})

	# Tentative avec Mistral API
	if mistral_client is not None:
		try:
			response = mistral_client.chat.complete(
				model=GENERATION_MODEL,
				messages=messages,
				temperature=0.1
			)
			return response.choices[0].message.content
		except Exception as e:
			print(f"[generator] Erreur API Mistral : {e}. Tentative de fallback...")

	# FALLBACK : Utilisation de Ollama (Local & Gratuit)
	try:
		import ollama
		response = ollama.chat(
			model=FALLBACK_MODEL,
			messages=messages,
			options={"temperature": 0.1}
		)
		return response['message']['content']
	except Exception as e:
		return f"Erreur : Impossible de générer une réponse (API indisponible et Ollama non configuré). {e}"

def build_context(retrieved_docs: list[dict]) -> str:
    """
    Construit le bloc de contexte à injecter dans le prompt.
    Chaque document est présenté avec sa source et son score.
    """
    parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        meta = doc["metadata"]
        source_label = (
            f"Intervention {meta['id']}"
            if meta.get("source") == "intervention"
            else f"Documentation technique ({meta.get('filename', meta.get('id', ''))})"
        )
        score_pct = int(doc.get("rerank_score", doc["score"]) * 100)
        parts.append(
            f"--- Document {i} | {source_label} | Pertinence : {score_pct}% ---\n"
            f"{doc['content']}"
        )

    return "\n\n".join(parts)