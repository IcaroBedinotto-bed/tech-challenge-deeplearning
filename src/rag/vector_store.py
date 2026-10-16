
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings


def create_vector_store(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_directory: str | Path | None = None,
) -> Chroma:
   
    if not documents:
        raise ValueError("A lista de documentos está vazia.")

    if persist_directory is None:
        project_root = Path(__file__).resolve().parents[2]
        persist_directory = Path("C:/chroma_data/order_reviews")

    persist_directory = Path(persist_directory)
    persist_directory.mkdir(parents=True, exist_ok=True)

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="orders_reviews",
        persist_directory=str(persist_directory),
    )

    return vector_store
