"""
ingestion.py — Chargement et chunking des deux sources de données.

Sources :
  - interventions.json  → 1 intervention = 1 document (déjà atomiques)
  - docs/*.txt          → découpage par sections avec overlap
"""

import json
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """Unité de base du pipeline RAG."""
    doc_id: str
    content: str
    source: str                        # "intervention" | "doc_technique"
    marque: str = ""
    type_equipement: str = ""
    code_erreur: str = ""
    difficulte: str = ""
    date: str = ""
    metadata: dict = field(default_factory=dict)


# Chargement des interventions JSON

def load_interventions(path: Path) -> list[Document]:
    """
    Transforme chaque fiche d'intervention en Document narratif.
    1 intervention → 1 document (les fiches sont déjà bien délimitées ~150 tokens).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        pieces = ", ".join(item.get("pieces_remplacees") or []) or "Aucune"
        content = (
            f"Équipement : {item['equipement']}\n"
            f"Date : {item['date']}\n"
            f"Code erreur : {item.get('code_erreur') or 'N/A'}\n"
            f"Symptôme : {item['symptome']}\n"
            f"Diagnostic : {item['diagnostic']}\n"
            f"Solution : {item['solution']}\n"
            f"Pièces remplacées : {pieces}\n"
            f"Durée : {item['temps_intervention_min']} min | "
            f"Difficulté : {item['difficulte']}"
        )

        docs.append(Document(
            doc_id=item["id"],
            content=content,
            source="intervention",
            marque=item.get("marque", ""),
            type_equipement=item.get("type_equipement", ""),
            code_erreur=item.get("code_erreur") or "",
            difficulte=item.get("difficulte", ""),
            date=item.get("date", ""),
            metadata={
                "id": item["id"],
                "marque": item.get("marque", ""),
                "type_equipement": item.get("type_equipement", ""),
                "code_erreur": item.get("code_erreur") or "",
                "difficulte": item.get("difficulte", ""),
                "date": item.get("date", ""),
                "source": "intervention",
            }
        ))
    return docs


# Chargement et chunking des fiches techniques TXT


# Correspondances nom de fichier → métadonnées
DOC_METADATA = {
    "fiche_frisquet_prestige.txt": {
        "marque": "Frisquet",
        "type_equipement": "chaudiere_gaz",
    },
    "fiche_daikin_altherma.txt": {
        "marque": "Daikin",
        "type_equipement": "pac_air_eau",
    },
    "fiche_saunier_duval_themaplus.txt": {
        "marque": "Saunier Duval",
        "type_equipement": "chaudiere_gaz",
    },
    "fiche_atlantic_climatisation.txt": {
        "marque": "Atlantic",
        "type_equipement": "climatisation",
    },
}

# Séparateur de sections : ligne vide + ligne en majuscules ou commençant par '##' / '---'
_SECTION_RE = re.compile(
    r'(?=\n(?:[A-ZÀÂÉÈÊÎÔÙÛÜ\s]{5,}|#{1,3} |\-{3,})\n)',
    re.MULTILINE
)

CHUNK_SIZE = 400      # tokens approximatifs (1 token ≈ 4 chars)
OVERLAP_CHARS = 200   # overlap entre chunks consécutifs


def _split_into_chunks(text: str, chunk_size_chars: int = CHUNK_SIZE * 4,
                        overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    Découpe un texte en chunks de taille max avec overlap.
    Priorité aux coupures sur saut de ligne.
    """
    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size_chars, length)

        # Essayer de couper sur un saut de ligne proche de la fin
        if end < length:
            newline_pos = text.rfind('\n', start, end)
            if newline_pos > start + chunk_size_chars // 2:
                end = newline_pos

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Si on a atteint la fin du texte, on s'arrête
        if end >= length:
            break

        next_start = end - overlap
        # Guard anti-boucle infinie : si on ne progresse pas, on avance de force
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def load_technical_docs(docs_dir: Path) -> list[Document]:
    """
    Charge et découpe les fiches techniques TXT.
    Stratégie :
      1. Découper d'abord par grandes sections (titres/séparateurs)
      2. Si une section dépasse chunk_size, la re-découper avec overlap
    """
    all_docs = []

    for txt_file in sorted(docs_dir.glob("*.txt")):
        meta = DOC_METADATA.get(txt_file.name, {})
        marque = meta.get("marque", "")
        type_eq = meta.get("type_equipement", "")

        text = txt_file.read_text(encoding="utf-8")

        # Découpage par sections
        sections = _SECTION_RE.split(text)
        sections = [s.strip() for s in sections if s.strip()]

        chunk_idx = 0
        for section in sections:
            # Si la section est assez courte, on la garde entière
            if len(section) <= CHUNK_SIZE * 4:
                sub_chunks = [section]
            else:
                sub_chunks = _split_into_chunks(section)

            for chunk_text in sub_chunks:
                doc_id = f"DOC-{txt_file.stem}-{chunk_idx:03d}"
                all_docs.append(Document(
                    doc_id=doc_id,
                    content=chunk_text,
                    source="doc_technique",
                    marque=marque,
                    type_equipement=type_eq,
                    metadata={
                        "id": doc_id,
                        "marque": marque,
                        "type_equipement": type_eq,
                        "code_erreur": "",
                        "difficulte": "",
                        "date": "",
                        "source": "doc_technique",
                        "filename": txt_file.name,
                    }
                ))
                chunk_idx += 1

    return all_docs


# Point d'entrée

def load_all_documents(data_dir: Path) -> list[Document]:
    """Charge et retourne l'ensemble des documents (interventions + docs techniques)."""
    interventions = load_interventions(data_dir / "interventions.json")
    tech_docs = load_technical_docs(data_dir / "docs")
    print(f"[ingestion] {len(interventions)} interventions chargées")
    print(f"[ingestion] {len(tech_docs)} chunks de docs techniques chargés")
    return interventions + tech_docs
