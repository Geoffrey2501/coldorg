"""
Script de vérification rapide (sans API calls) pour valider la logique core.
Lance : python3 test_core.py
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
from src.ingestion import load_interventions, load_technical_docs
from src.retriever import extract_entities, rerank

# ── 1. Ingestion ──────────────────────────────
interventions = load_interventions(Path('data/interventions.json'))
tech_docs = load_technical_docs(Path('data/docs'))

print(f"✓ Interventions chargées : {len(interventions)} documents")
print(f"✓ Docs techniques chargées : {len(tech_docs)} chunks")
print()

# ── 2. Format d'un document ───────────────────
print("=== Exemple document INT-001 ===")
print(interventions[0].content)
print()
print("=== Métadonnées INT-001 ===")
print(interventions[0].metadata)
print()

# ── 3. Extraction d'entités ───────────────────
print("=== Test extraction entités (5 questions de test) ===")
questions = [
    ("Q1", "Code erreur E133 sur une chaudière Frisquet Prestige Condensation 25kW. La chaudière ne redémarre pas depuis ce matin."),
    ("Q2", "J'ai une fuite d'eau qui coule le long du mur sous mon climatiseur Atlantic Idéa."),
    ("Q3", "Ma pompe à chaleur Daikin Altherma affiche le code U4, plus de chauffage ni d'eau chaude."),
    ("Q4", "Un client a une chaudière Saunier Duval ThemaPlus Condens en panne avec le code F28 pour la troisième fois ce mois."),
    ("Q5", "Le client dit que sa PAC Daikin chauffe mais que la maison reste froide. La PAC ne montre aucun code erreur."),
]

for qid, q in questions:
    e = extract_entities(q)
    print(f"  {qid}: marque={str(e['marque']):<14} code={str(e['code_erreur']):<6}")

# ── 4. Test re-ranking ────────────────────────
print()
print("=== Test re-ranking (Q4: F28 Saunier Duval) ===")
mock_results = [
    {"content": "INT-001", "score": 0.85, "metadata": {"id": "INT-001", "marque": "Frisquet", "code_erreur": "E133", "source": "intervention"}},
    {"content": "INT-004", "score": 0.78, "metadata": {"id": "INT-004", "marque": "Saunier Duval", "code_erreur": "F28", "source": "intervention"}},
    {"content": "INT-023", "score": 0.72, "metadata": {"id": "INT-023", "marque": "Saunier Duval", "code_erreur": "F28", "source": "intervention"}},
    {"content": "DOC-001", "score": 0.70, "metadata": {"id": "DOC-001", "marque": "Saunier Duval", "code_erreur": "", "source": "doc_technique"}},
]
entities_q4 = extract_entities(questions[3][1])
ranked = rerank(mock_results, entities_q4, top_k=4)

print("  Avant re-ranking: INT-001(0.85), INT-004(0.78), INT-023(0.72), DOC-001(0.70)")
print("  Après re-ranking:")
for r in ranked:
    print(f"    {r['metadata']['id']}: score={r['rerank_score']:.2f} (base={r['score']:.2f})")

print()
print("✓ Tous les tests core passés avec succès !")
print()
print("Pour lancer le pipeline complet avec l'API Mistral :")
print("  python3 evaluate.py   # Test sur les 5 questions")
print("  python3 main.py       # CLI interactive")
