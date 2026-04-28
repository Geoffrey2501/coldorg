"""
rag_pipeline.py — Orchestrateur du pipeline RAG complet.

Responsabilités :
  - Initialiser les clients Mistral et ChromaDB
  - Construire l'index (embedding + stockage) si nécessaire
  - Exposer une interface simple : RAGPipeline.query(question) → réponse

Imports lourds (mistralai, chromadb) sont lazy — chargés à la première utilisation.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from .ingestion import load_all_documents
from .embeddings import get_client as get_mistral_client, embed_texts
from .vector_store import get_chroma_client, index_documents, collection_exists_and_populated
from .retriever import retrieve
from .generator import generate_answer


# Chemins par défaut (relatifs à la racine du projet)
_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = _ROOT / "data"
CHROMA_DIR = _ROOT / "chroma_db"


class RAGPipeline:
    """
    Façade principale du système RAG COLDORG.

    Usage :
        pipeline = RAGPipeline()
        pipeline.build_index()          # à faire une seule fois
        response = pipeline.query("Code erreur E133 sur Frisquet...")
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        chroma_dir: Path = CHROMA_DIR,
    ):
        load_dotenv()

        self.data_dir = data_dir
        self._chroma_dir = chroma_dir
        self._mistral = None     # lazy init
        self._chroma = None      # lazy init
        self._conversation_history: list[dict] = []
        self.use_api = os.environ.get("MISTRAL_API_KEY") is not None
        if not self.use_api:
            print("[pipeline] ATTENTION : Pas de clé API. Mode LOCAL activé.")

    @property
    def mistral(self):
        if self._mistral is None:
            self._mistral = get_mistral_client()
        return self._mistral

    @property
    def chroma(self):
        if self._chroma is None:
            self._chroma = get_chroma_client(self._chroma_dir)
        return self._chroma

    # ──────────────────────────────────────────
    # Construction de l'index
    # ──────────────────────────────────────────

    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Charge les documents, génère les embeddings et les stocke dans ChromaDB.
        Si l'index existe déjà, cette étape est sautée (sauf si force_rebuild=True).
        """
        if not force_rebuild and collection_exists_and_populated(self.chroma):
            print("[pipeline] Index déjà construit — utilisation du cache ChromaDB.")
            return

        print("[pipeline] Construction de l'index...")

        # 1. Ingestion
        documents = load_all_documents(self.data_dir)

        # 2. Embedding (par batch)
        print(f"[pipeline] Génération des embeddings pour {len(documents)} documents...")
        texts = [d.content for d in documents]
        embeddings = embed_texts(texts, client=self.mistral)

        # 3. Indexation dans ChromaDB
        index_documents(documents, embeddings, self.chroma)

        print("[pipeline] ✓ Index construit avec succès.")

    # ──────────────────────────────────────────
    # Requête
    # ──────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_metadata_filter: bool = True,
        multi_turn: bool = False,
    ) -> dict:
        """
        Répond à une question en utilisant le pipeline RAG complet.

        Args:
            question            : question du technicien
            top_k               : nombre de documents à récupérer
            use_metadata_filter : activer le filtrage par marque détectée
            multi_turn          : conserver l'historique de conversation

        Returns:
            dict avec :
              - "answer"    : réponse générée (str)
              - "sources"   : liste des documents sources utilisés
              - "entities"  : entités extraites de la question (marque, code erreur)
        """
        # Retrieval
        retrieved_docs, entities = retrieve(
            question=question,
            chroma_client=self.chroma,
            mistral_client=self.mistral,
            top_k=top_k,
            use_metadata_filter=use_metadata_filter,
        )

        # Génération
        history = self._conversation_history if multi_turn else None
        answer = generate_answer(
            question=question,
            retrieved_docs=retrieved_docs,
            mistral_client=self.mistral,
            conversation_history=history,
        )

        # Mise à jour de l'historique de conversation (multi-turn)
        if multi_turn:
            self._conversation_history.append({"role": "user", "content": question})
            self._conversation_history.append({"role": "assistant", "content": answer})

        return {
            "answer": answer,
            "sources": retrieved_docs,
            "entities": entities,
        }

    def reset_conversation(self) -> None:
        """Efface l'historique de conversation (nouvelle session)."""
        self._conversation_history = []
