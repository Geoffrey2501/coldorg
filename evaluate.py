"""
evaluate.py — Évaluation du système RAG sur les 5 questions de test.

Lance : python evaluate.py

Affiche pour chaque question :
  - La question
  - Les entités détectées (marque, code erreur)
  - Les documents sources récupérés (ID + score)
  - La réponse générée
  - Un séparateur pour la lisibilité
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

console = Console()

QUESTIONS_FILE = Path(__file__).parent / "data" / "questions_test.json"


def run_evaluation():
    console.rule("[bold blue]COLDORG RAG — Évaluation sur 5 questions de test")

    # Initialisation du pipeline
    console.print("\n[dim]Initialisation du pipeline RAG...[/dim]")
    pipeline = RAGPipeline()
    pipeline.build_index()

    # Chargement des questions
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    console.print(f"\n[bold green]✓ Pipeline prêt. {len(questions)} questions à évaluer.[/bold green]\n")

    for q in questions:
        qid = q["id"]
        question = q["question"]

        console.rule(f"[bold yellow]{qid}")
        console.print(Panel(question, title="[bold]Question du technicien", border_style="yellow"))

        # Requête au pipeline
        with console.status(f"[dim]Recherche et génération en cours...[/dim]"):
            result = pipeline.query(question)

        # Entités détectées
        entities = result["entities"]
        console.print(
            f"[dim]Entités détectées :[/dim] "
            f"Marque=[cyan]{entities['marque'] or 'N/A'}[/cyan]  "
            f"Code erreur=[cyan]{entities['code_erreur'] or 'N/A'}[/cyan]"
        )

        # Tableau des sources
        table = Table(
            title="Documents sources récupérés",
            box=box.SIMPLE_HEAVY,
            show_lines=True,
        )
        table.add_column("ID", style="cyan", width=20)
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
            title="[bold green]Réponse générée",
            border_style="green",
        ))

        console.print()


if __name__ == "__main__":
    run_evaluation()
