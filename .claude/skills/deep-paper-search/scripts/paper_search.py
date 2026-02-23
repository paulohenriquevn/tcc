#!/usr/bin/env python3
"""Deep Paper Search — Busca, indexação e consulta de papers científicos.

Usa LlamaIndex com ArxivReader e PubmedReader para buscar papers,
indexá-los localmente com embeddings HuggingFace, e fazer retrieval
semântico (sem LLM — o agente Claude sintetiza os resultados).

Modos:
    search  — Busca papers no Arxiv + PubMed e salva localmente
    index   — Indexa papers baixados em vector store local
    query   — Busca semântica no índice (retorna chunks com score)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Diretórios padrão (relativos à raiz do projeto TCC)
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # TCC/claude/skills/deep-paper-search/scripts -> TCC/
PAPERS_DIR = PROJECT_ROOT / "pesquisas" / ".papers"
INDEX_DIR = PROJECT_ROOT / "pesquisas" / ".paper_index"


def search_papers(
    query: str,
    max_results: int = 10,
    sources: str = "all",
    papers_dir: Optional[Path] = None,
) -> None:
    """Busca papers no Arxiv e/ou PubMed, salva localmente e imprime resumos."""
    if papers_dir is None:
        papers_dir = PAPERS_DIR
    papers_dir.mkdir(parents=True, exist_ok=True)

    all_papers = []
    all_abstracts = []

    # --- Arxiv ---
    if sources in ("all", "arxiv"):
        print(f"\n{'='*60}")
        print(f"ARXIV — Buscando: {query}")
        print(f"{'='*60}")
        try:
            from llama_index.readers.papers import ArxivReader

            reader = ArxivReader()
            papers, abstracts = reader.load_papers_and_abstracts(
                search_query=query,
                papers_dir=str(papers_dir),
                max_results=max_results,
            )
            all_papers.extend(papers)
            all_abstracts.extend(abstracts)

            for i, abstract_doc in enumerate(abstracts, 1):
                meta = abstract_doc.metadata
                print(f"\n--- Paper {i} ---")
                print(f"Title: {meta.get('Title of this paper', 'N/A')}")
                print(f"URL: {meta.get('URL', 'N/A')}")
                print(f"Abstract: {abstract_doc.text[:500]}...")
                print()

            print(f"[Arxiv] {len(papers)} papers baixados em {papers_dir}")
        except ImportError:
            print("[ERRO] llama-index-readers-papers não instalado.")
            print("  pip install llama-index-readers-papers")
            sys.exit(1)
        except Exception as e:
            print(f"[ERRO Arxiv] {type(e).__name__}: {e}")

    # --- PubMed ---
    if sources in ("all", "pubmed"):
        print(f"\n{'='*60}")
        print(f"PUBMED — Buscando: {query}")
        print(f"{'='*60}")
        try:
            from llama_index.readers.papers import PubmedReader

            reader = PubmedReader()
            pubmed_docs = reader.load_data(
                search_query=query,
                max_results=max_results,
            )
            all_papers.extend(pubmed_docs)

            for i, doc in enumerate(pubmed_docs, 1):
                meta = doc.metadata
                print(f"\n--- Paper {i} ---")
                print(f"Title: {meta.get('Title of this paper', 'N/A')}")
                print(f"Journal: {meta.get('Journal it was published in:', 'N/A')}")
                print(f"URL: {meta.get('URL', 'N/A')}")
                print(f"Text preview: {doc.text[:300]}...")
                print()

            print(f"[PubMed] {len(pubmed_docs)} papers encontrados")
        except ImportError:
            print("[ERRO] llama-index-readers-papers não instalado.")
            print("  pip install llama-index-readers-papers")
            sys.exit(1)
        except Exception as e:
            print(f"[ERRO PubMed] {type(e).__name__}: {e}")

    # --- Resumo ---
    print(f"\n{'='*60}")
    print(f"RESUMO")
    print(f"{'='*60}")
    print(f"Total de papers: {len(all_papers)}")
    print(f"Total de abstracts: {len(all_abstracts)}")
    print(f"Papers salvos em: {papers_dir}")


def build_index(papers_dir: Optional[Path] = None, index_dir: Optional[Path] = None) -> None:
    """Indexa todos os papers baixados em um VectorStoreIndex local."""
    if papers_dir is None:
        papers_dir = PAPERS_DIR
    if index_dir is None:
        index_dir = INDEX_DIR

    if not papers_dir.exists():
        print(f"[ERRO] Diretório de papers não encontrado: {papers_dir}")
        print("  Execute 'search' primeiro para baixar papers.")
        sys.exit(1)

    pdf_files = list(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"[ERRO] Nenhum PDF encontrado em {papers_dir}")
        print("  Execute 'search' primeiro para baixar papers.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"INDEXAÇÃO — {len(pdf_files)} PDFs encontrados")
    print(f"{'='*60}")

    try:
        from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        print(f"[ERRO] Dependência não instalada: {e}")
        print("  pip install llama-index-core llama-index-embeddings-huggingface")
        sys.exit(1)

    print("Carregando modelo de embeddings (all-MiniLM-L6-v2)...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    Settings.llm = None  # Sem LLM — retrieval only

    print(f"Lendo PDFs de {papers_dir}...")
    reader = SimpleDirectoryReader(input_dir=str(papers_dir), required_exts=[".pdf"])
    documents = reader.load_data()
    print(f"  {len(documents)} documentos/chunks carregados")

    print("Construindo índice vetorial...")
    index = VectorStoreIndex.from_documents(documents)

    index_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(index_dir))
    print(f"Índice salvo em: {index_dir}")

    print(f"\n{'='*60}")
    print(f"INDEXAÇÃO COMPLETA")
    print(f"{'='*60}")
    print(f"Documentos indexados: {len(documents)}")
    print(f"Índice em: {index_dir}")


def query_index(
    question: str,
    top_k: int = 5,
    index_dir: Optional[Path] = None,
) -> None:
    """Busca semântica no índice — retorna chunks relevantes sem LLM."""
    if index_dir is None:
        index_dir = INDEX_DIR

    if not index_dir.exists():
        print(f"[ERRO] Índice não encontrado em: {index_dir}")
        print("  Execute 'index' primeiro para construir o índice.")
        sys.exit(1)

    try:
        from llama_index.core import Settings, StorageContext, load_index_from_storage
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    except ImportError as e:
        print(f"[ERRO] Dependência não instalada: {e}")
        print("  pip install llama-index-core llama-index-embeddings-huggingface")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"QUERY — {question}")
    print(f"{'='*60}")

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    Settings.llm = None

    print("Carregando índice...")
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)

    retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
    nodes = retriever.retrieve(question)

    print(f"\nTop {len(nodes)} chunks relevantes:\n")
    for i, node in enumerate(nodes, 1):
        score = node.score if node.score is not None else 0.0
        meta = node.metadata
        file_name = meta.get("file_name", "N/A")
        page = meta.get("page_label", meta.get("page", "N/A"))

        print(f"--- Chunk {i} (score: {score:.4f}) ---")
        print(f"Source: {file_name} | Page: {page}")
        print(f"Text:\n{node.text[:800]}")
        print()

    print(f"{'='*60}")
    print(f"Total: {len(nodes)} chunks retornados para top_k={top_k}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Deep Paper Search — Busca, indexação e consulta de papers científicos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  %(prog)s search --query "accent control TTS LoRA" --max-results 5
  %(prog)s search --query "speaker disentanglement" --sources arxiv
  %(prog)s index
  %(prog)s query --question "How do LoRA adapters affect prosody in TTS?"
  %(prog)s query --question "CORAA dataset characteristics" --top-k 10
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- search ---
    search_parser = subparsers.add_parser("search", help="Busca papers no Arxiv + PubMed")
    search_parser.add_argument("--query", "-q", required=True, help="Termo de busca")
    search_parser.add_argument(
        "--max-results", "-n", type=int, default=10, help="Máximo de papers por fonte (default: 10)"
    )
    search_parser.add_argument(
        "--sources", "-s", choices=["all", "arxiv", "pubmed"], default="all", help="Fontes (default: all)"
    )
    search_parser.add_argument("--papers-dir", type=Path, default=None, help="Diretório para salvar papers")

    # --- index ---
    index_parser = subparsers.add_parser("index", help="Indexa papers baixados em vector store local")
    index_parser.add_argument("--papers-dir", type=Path, default=None, help="Diretório dos papers")
    index_parser.add_argument("--index-dir", type=Path, default=None, help="Diretório para salvar o índice")

    # --- query ---
    query_parser = subparsers.add_parser("query", help="Busca semântica no índice")
    query_parser.add_argument("--question", "-q", required=True, help="Pergunta para buscar")
    query_parser.add_argument("--top-k", "-k", type=int, default=5, help="Número de chunks (default: 5)")
    query_parser.add_argument("--index-dir", type=Path, default=None, help="Diretório do índice")

    return parser.parse_args()


def main() -> None:
    """Entrypoint principal."""
    args = parse_args()

    if args.command == "search":
        search_papers(
            query=args.query,
            max_results=args.max_results,
            sources=args.sources,
            papers_dir=args.papers_dir,
        )
    elif args.command == "index":
        build_index(
            papers_dir=args.papers_dir,
            index_dir=args.index_dir,
        )
    elif args.command == "query":
        query_index(
            question=args.question,
            top_k=args.top_k,
            index_dir=args.index_dir,
        )


if __name__ == "__main__":
    main()
