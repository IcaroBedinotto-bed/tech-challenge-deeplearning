
from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model() -> HuggingFaceEmbeddings:

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32
        }
    )

    return embeddings
