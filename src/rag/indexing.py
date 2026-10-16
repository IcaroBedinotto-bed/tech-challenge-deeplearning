
from pathlib import Path

import pandas as pd
from langchain_chroma import Chroma

from src.rag.documents import create_documents
from src.rag.embedding import load_embedding_model
from src.rag.vector_store import create_vector_store


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "olist_order_reviews_dataset.csv"
)

CHROMA_DIR = Path("C:/chroma_data/orders_reviews")

COLLECTION_NAME = "orders_reviews"


def build_vector_store() -> Chroma:

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Arquivo de avaliações não encontrado: {RAW_DATA_PATH}"
        )

    print("1. Lendo avaliações...")
    df_reviews = pd.read_csv(RAW_DATA_PATH)

    print(f"   Registros no CSV: {len(df_reviews):,}")

    print("2. Criando documentos...")
    documents = create_documents(df_reviews)

    if not documents:
        raise ValueError("Nenhum documento válido foi gerado.")

    print(f"   Documentos válidos: {len(documents):,}")

    print("3. Carregando modelo de embeddings...")
    embeddings = load_embedding_model()

    print("4. Criando índice vetorial...")
    vector_store = create_vector_store(
        documents=documents,
        embeddings=embeddings,
        persist_directory=CHROMA_DIR,
    )

    print("5. Validando coleção...")
    total = vector_store._collection.count()

    if total != len(documents):
        raise RuntimeError(
            f"Quantidade divergente: {total} registros no Chroma "
            f"e {len(documents)} documentos de origem."
        )

    print(f"   Registros no Chroma: {total:,}")

    return vector_store


if __name__ == "__main__":
    vector_store = build_vector_store()

    print("6. Testando recuperação...")
    results = vector_store.similarity_search(
        "query: atraso na entrega",
        k=5,
    )

    print(f"   Documentos recuperados: {len(results)}")

    for i, doc in enumerate(results, start=1):
        print(f"\nAvaliação {i}")
        print(f"Nota: {doc.metadata.get('review_score')}")
        print(f"Comentário: {doc.page_content[:200]}")

    if len(results) != 5:
        raise RuntimeError(
            "A busca não retornou os cinco documentos esperados."
        )

    print("\nIndexação e teste concluídos.")
