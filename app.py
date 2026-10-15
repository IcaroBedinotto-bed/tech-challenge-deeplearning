
from pathlib import Path

import streamlit as st
from langchain_chroma import Chroma

from src.rag.embedding import load_embedding_model
from src.rag.chain import load_llm, ask_rag


# --------------------------------------------------
# Configuração da aplicação
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_ROOT / "data" / "processed" / "chroma_final"

st.set_page_config(
    page_title="Voice of Customer Intelligence",
    page_icon="💬",
    layout="wide",
)

st.title("💬 Voice of Customer Intelligence")
st.caption(
    "Compreenda as reclamações, elogios e pontos de melhorias apontadas pelos seus clientes"
)


# --------------------------------------------------
# Carregamento dos recursos
# Executado uma vez e reutilizado entre interações
# --------------------------------------------------

@st.cache_resource
def load_resources():
    if not (CHROMA_DIR / "chroma.sqlite3").exists():
        raise FileNotFoundError(
            f"Banco Chroma não encontrado em: {CHROMA_DIR}"
        )

    embeddings = load_embedding_model()

    vector_store = Chroma(
        collection_name="olist_reviews",
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
    )

    llm = load_llm()

    return vector_store, llm


try:
    with st.spinner("Carregando o sistema..."):
        vector_store, llm = load_resources()
except Exception as error:
    st.error(f"Não foi possível inicializar o RAG: {error}")
    st.stop()


# --------------------------------------------------
# Pergunta do usuário
# --------------------------------------------------

with st.form("question_form"):
    question = st.text_input(
        "Sua pergunta",
        placeholder=(
            "Ex.: Quais são as principais reclamações "
            "dos clientes?"
        ),
    )

    k = st.select_slider(
        "Quantidade de avaliações para recuperar",
        options=[3, 5, 10],
        value=5,
    )

    submitted = st.form_submit_button(
        "Consultar avaliações",
        type="primary",
    )


# --------------------------------------------------
# Execução do RAG
# --------------------------------------------------

if submitted:
    if not question.strip():
        st.warning("Digite uma pergunta para continuar.")
    else:
        try:
            with st.spinner("Analisando as avaliações..."):
                result = ask_rag(
                    question=question,
                    vector_store=vector_store,
                    llm=llm,
                    k=k,
                )

            st.subheader("Resposta")
            st.markdown(result["answer"])

            metrics = result["metrics"]

            st.subheader("Desempenho")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Tempo total",
                f'{metrics["total_seconds"]:.2f} s',
            )
            col2.metric(
                "Recuperação",
                f'{metrics["retrieval_seconds"]:.2f} s',
            )
            col3.metric(
                "Geração",
                f'{metrics["generation_seconds"]:.2f} s',
            )

            with st.expander("Consumo de tokens"):
                st.write(
                    "Tokens de entrada:",
                    metrics["input_tokens"],
                )
                st.write(
                    "Tokens de saída:",
                    metrics["output_tokens"],
                )

            st.subheader("Avaliações utilizadas como fontes")

            sources = result["sources"]

            if sources:
                for source in sources:
                    with st.expander(
                        f'Avaliação {source["evidence_id"]} '
                        f'— Nota: {source["review_score"]}'
                    ):
                        st.write("**Comentário**")
                        st.write(source["comment"])

                        st.caption(
                            f'Review ID: {source["review_id"]}'
                        )
                        st.caption(
                            f'Order ID: {source["order_id"]}'
                        )
            else:
                st.info(
                    "Nenhuma fonte foi apresentada para esta resposta."
                )

        except Exception as error:
            st.error(f"Erro ao consultar o RAG: {error}")
