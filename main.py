import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt
from rich import box

sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline

console = Console()

HELP_TEXT = (
    "— Nouvelle conversation\n"
    "— Reconstruire l'index (force)\n"
    "— Afficher cette aide\n"
    "— Quitter"
)


def main():
    console.print(Panel(
        "Assistant technique CVC — Powered by Mistral AI + ChromaDB\n\n"
        + HELP_TEXT
    ))

    # Initialisation
    console.print("\nInitialisation du pipeline...")
    pipeline = RAGPipeline()

    try:
        pipeline.build_index()
    except EnvironmentError as e:
        console.print(f"[Erreur : {e}")
        console.print("Crée un fichier .env à partir de .env et renseigne ta clé Mistral.")
        sys.exit(1)

    console.print("✓ Prêt ! Pose ta question.\n")

    turn = 0

    while True:
        try:
            user_input = Prompt.ask(f"[Tour {turn + 1}] Technicien").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\nAu revoir !")
            break

        if not user_input:
            continue

        # Commandes spéciales
        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("Session terminée. Au revoir !")
            break

        if user_input.lower() == "/reset":
            pipeline.reset_conversation()
            turn = 0
            console.print("Historique effacé. Nouvelle conversation.")
            continue

        if user_input.lower() == "/index":
            console.print("Reconstruction de l'index...")
            pipeline.build_index(force_rebuild=True)
            console.print("Index reconstruit.")
            continue

        if user_input.lower() == "/help":
            console.print(Panel(HELP_TEXT, title="Aide", border_style="cyan"))
            continue

        # Requête RAG
        with console.status("Recherche en cours..."):
            result = pipeline.query(user_input, multi_turn=True)

        # Affichage des sources (compact)
        sources_summary = ", ".join(
            doc["metadata"].get("id", "?") for doc in result["sources"]
        )
        entities = result["entities"]
        console.print(
            f"Sources : {sources_summary} | "
            f"Marque : {entities['marque'] or '—'} | "
            f"Code : {entities['code_erreur'] or '—'}"
        )

        # Affichage de la réponse
        console.print(Panel(
            Markdown(result["answer"]),
            title="Assistant COLDORG",
        ))

        turn += 1
        console.print()


if __name__ == "__main__":
    main()
