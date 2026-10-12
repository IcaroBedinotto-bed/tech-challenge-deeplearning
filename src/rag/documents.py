
import pandas as pd
from langchain_core.documents import Document


def create_documents(df_reviews: pd.DataFrame) -> list[Document]:
    """
    Transforma avaliações válidas em documentos para o RAG.
    Cada documento recebe o prefixo passage: exigido pelo E5.
    """

    documents = []

    for _, row in df_reviews.iterrows():
        comment = row["review_comment_message"]

        if pd.isna(comment) or not str(comment).strip():
            continue

        document = Document(
            page_content=f"passage: {str(comment).strip()}", #O modelo de embedding escolhido demandam o uso do prefixo passage
            metadata={
                "review_id": str(row["review_id"]),
                "order_id": str(row["order_id"]),
                "review_score": int(row["review_score"]),
            },
        )

        documents.append(document)

    return documents
