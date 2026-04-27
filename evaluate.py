"""
evaluate.py — Évaluation du système RAG sur les 5 questions de test.

Lance : python evaluate.py
"""

import json
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich import box

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline

# Initialisation de la console sans système de couleur
console = Console(color_system=None)

QUESTIONS_FILE = Path(__file__).parent / "data" / "questions_test.json"


def run_evaluation():
    console.rule("COLDORG RAG — Évaluation sur 5 questions de test")

    # Initialisation du pipeline
    console.print("\nInitialisation du pipeline RAG...")
    pipeline = RAGPipeline()
    pipeline.build_index()

    # Chargement des questions
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    console.print(f"\n✓ Pipeline prêt. {len(questions)} questions à évaluer.\n")

    for q in questions:
        qid = q["id"]
        question = q["question"]

        console.rule(f"{qid}")
        console.print(Panel(question, title="Question du technicien"))

        # Requête au pipeline
        with console.status("Recherche et génération en cours..."):
            result = pipeline.query(question)

        # Entités détectées
        entities = result["entities"]
        console.print(
            f"Entités détectées : "
            f"Marque={entities['marque'] or 'N/A'}  "
            f"Code erreur={entities['code_erreur'] or 'N/A'}"
        )

        # Tableau des sources
        table = Table(
            title="Documents sources récupérés",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        table.add_column("ID", width=20)
        table.add_column("Source", width=16)
        table.add_column("Marque", width=16)
        table.add_column("Code erreur", width=12)
        table.add_column("Score", justify="right", width=8)

        for doc in result["sources"]:
            meta = doc["metadata"]
            score = doc.get("rerank_score", doc["score"])
            table.add_row(
                meta.get("id", "—"),
                meta.get("source", "—"),
                meta.get("marque", "—"),
                meta.get("code_erreur") or "—",
                f"{score:.2f}",
            )

        console.print(table)

        # Réponse générée
        console.print(Panel(
            Markdown(result["answer"]),
            title="Réponse générée",
        ))

        console.print()


if __name__ == "__main__":
    run_evaluation()