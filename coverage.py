"""
Test de couverture + évaluation LLM du RAG.
Charge les questions depuis coverage.csv, interroge le RAG pour chacune,
puis demande à un LLM juge (Claude) de noter chaque réponse de 0 à 10.

Critères de notation :
  10   : réponse correcte et complète
  7-9  : réponse correcte mais incomplète ou légèrement imprécise
  4-6  : le RAG dit "les documents ne me permettent pas de répondre complètement"
         mais fournit quand même des éléments pertinents
  1-3  : réponse très partielle ou hors sujet
  0    : le RAG n'a pas pu répondre du tout / question hors domaine

Lancement :
    python coverage.py
"""

import csv
import json
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

CSV_FILE = Path("coverage.csv")
JUDGE_MODEL = "claude-opus-4-8"

JUDGE_PROMPT = """Tu es un évaluateur impartial d'un système RAG spécialisé sur l'IA et la désinformation.

Question posée au RAG : {question}
Réponse du RAG : {response}
Sources (extraits de documents) utilisées par le RAG pour répondre :
{sources}

Attribue une note de 0 à 10 selon ces critères :
- 10 : réponse correcte et complète, bien sourcée et fidèle aux extraits fournis
- 7-9 : réponse correcte mais incomplète ou légèrement imprécise par rapport aux extraits
- 4-6 : le RAG indique que ses documents ne lui permettent pas de répondre complètement, mais fournit quand même des éléments pertinents
- 1-3 : réponse très partielle, confuse, hors sujet par rapport aux extraits, ou question totalement hors domaine
- 0 : le RAG n'a pas pu répondre du tout, ou la question était complètement hors du domaine de ses documents

Réponds UNIQUEMENT avec un JSON valide (sans bloc markdown), de la forme :
{{"note": <entier de 0 à 10>, "justification": "<une phrase courte>"}}"""


def load_questions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def format_sources(docs) -> str:
    if not docs:
        return "(aucune source récupérée)"
    parts = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Article")
        source = doc.metadata.get("source", "")
        parts.append(f"[{i} — {title}]({source})\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def judge_response(client: anthropic.Anthropic, question: str, response: str, sources: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, response=response, sources=sources)
    result = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    text = result.content[0].text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"note": -1, "justification": f"Erreur de parsing du juge : {text[:100]}"}


def main():
    from rag import VECTOR_STORE_PATH, RAGManager
    from llm import LLMManager

    rows = load_questions(CSV_FILE)
    print(f"  {len(rows)} question(s) chargée(s) depuis '{CSV_FILE}'.\n")

    rag = RAGManager()
    vectorstore = rag.load_existing() if Path(VECTOR_STORE_PATH).exists() else rag.index_from_db()
    retriever = rag.build_retriever(vectorstore)
    chain = LLMManager().build_chain(retriever)
    judge = anthropic.Anthropic()

    results = []
    for i, row in enumerate(rows, 1):
        question = row["question"]
        categorie = row["categorie"]
        print(f"[{i}/{len(rows)}] {question}")
        docs = retriever.invoke(question)
        sources = format_sources(docs)
        response = chain.invoke(question)
        print(f"  RAG : {response[:150].replace(chr(10), ' ')}...")
        evaluation = judge_response(judge, question, response, sources)
        note = evaluation.get("note", -1)
        justification = evaluation.get("justification", "")
        print(f"  Note : {note}/10 — {justification}\n")
        results.append({"categorie": categorie, "question": question, "reponse": response, "note": note})

    with CSV_FILE.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["categorie", "question", "reponse", "note"])
        writer.writeheader()
        writer.writerows(results)
    print(f"  Résultats sauvegardés dans '{CSV_FILE}'.")

    notes = [r["note"] for r in results if r["note"] >= 0]
    if notes:
        print("=" * 60)
        print(f"  Moyenne : {sum(notes) / len(notes):.1f}/10  ({len(notes)} questions évaluées)")
        print(f"  Min : {min(notes)}/10  |  Max : {max(notes)}/10")

    return results


if __name__ == "__main__":
    main()
