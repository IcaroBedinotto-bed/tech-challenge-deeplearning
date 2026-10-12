
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma


def load_llm() -> ChatOllama:

    return ChatOllama(
        model="qwen3:4b",
        temperature=0,
    )


def ask_rag(
    question: str,
    vector_store: Chroma,
    llm: ChatOllama,
    k: int = 5,
) -> dict:
    """Recupera avaliações e responde usando as evidências."""

    if not question.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    # 1. Recuperação semântica
    query = f"query: {question}"
    documents = vector_store.similarity_search(query, k=k)

    if not documents:
        return {
            "answer": "Não encontrei avaliações relevantes para responder.",
            "sources": [],
        }

    # 2. Preparação do contexto e das fontes
    sources = []
    context_parts = []

    for i, doc in enumerate(documents, start=1):
        comment = doc.page_content.removeprefix("passage: ")
        review_id = doc.metadata.get("review_id", "não informado")
        order_id = doc.metadata.get("order_id", "não informado")
        score = doc.metadata.get("review_score", "não informado")

        context_parts.append(
            f"[Avaliação {i}]\n"
            f"Comentário: {comment}\n"
            f"Nota: {score}"
        )

        sources.append({
            "review_id": review_id,
            "order_id": order_id,
            "review_score": score,
            "comment": comment,
        })

    context = "\n\n".join(context_parts)

    # 3. Geração da resposta
    prompt = f"""
Você é um analista de experiência do cliente de um e-commerce.

Responda à pergunta usando exclusivamente as avaliações abaixo.

Regras:
- Responda em português brasileiro.
- Não invente fatos, números ou reclamações.
- Diferencie evidências de interpretações.
- Cite as avaliações pelo identificador [Avaliação N].
- Se as evidências forem insuficientes, informe isso claramente.
- Não afirme que um problema é frequente apenas porque aparece
  nos resultados recuperados.
- Se a pergunta não puder ser respondida pelas avaliações,
  explique essa limitação.

Pergunta:
{question}

Avaliações disponíveis:
{context}
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "sources": sources,
    }
