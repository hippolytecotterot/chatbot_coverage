"""
Console interface for debugging the MindTheBot RAG pipeline.
Run with: python console.py
"""

from pathlib import Path

from dotenv import load_dotenv

from llm import LLMManager
from rag import RAGManager

load_dotenv(Path(__file__).parent / ".env")


def main() -> None:
    print("=== MindTheBot RAG — Console debug ===")
    print("Initialisation du RAG...")

    rag = RAGManager()
    llm = LLMManager()

    vs_path = Path(rag.vector_store_path)
    if vs_path.exists() and any(vs_path.iterdir()):
        print(f"  Chargement du vector store existant depuis '{vs_path}'")
        vs = rag.load_existing()
    else:
        print("  Aucun vector store trouvé, indexation depuis la base de données...")
        vs = rag.index_from_db()

    retriever = rag.build_retriever(vs)
    chain = llm.build_chain(retriever)

    print("\nPrêt. Tape 'exit' ou Ctrl+C pour quitter.\n")

    while True:
        try:
            question = input("Question > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAu revoir.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Au revoir.")
            break

        docs = retriever.invoke(question)
        source_urls = rag.get_source_urls(docs)

        print(f"\n[{len(docs)} chunk(s) récupéré(s)]", end="")
        if source_urls:
            print(f" — Sources : {', '.join(source_urls)}", end="")
        print("\n")

        print("Réponse : ", end="", flush=True)
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
