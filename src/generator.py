"""
generator.py — Génération de la réponse via mistral-small-latest.

Le prompt système positionne le LLM comme un assistant technicien CVC expert.
La réponse est demandée en format structuré Markdown.
"""


GENERATION_MODEL = "mistral-small-latest"

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


def generate_answer(
    question: str,
    retrieved_docs: list[dict],
    mistral_client,
    conversation_history: list[dict] | None = None,
) -> str:
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
    context = build_context(retrieved_docs)

    user_message = (
        f"CONTEXTE (documents récupérés depuis la base COLDORG) :\n\n"
        f"{context}\n\n"
        f"---\n\n"
        f"QUESTION DU TECHNICIEN : {question}"
    )

    # Construction des messages avec historique optionnel (multi-turn)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        # On ajoute les tours précédents mais sans les re-contextes (trop long)
        # On garde seulement les paires Question/Réponse pour la mémoire de session
        for turn in conversation_history[-4:]:   # max 4 tours en mémoire
            messages.append(turn)

    messages.append({"role": "user", "content": user_message})

    response = mistral_client.chat.complete(
        model=GENERATION_MODEL,
        messages=messages,
        temperature=0.1,      # réponses déterministes et factuelles
        max_tokens=1024,
    )

    return response.choices[0].message.content
