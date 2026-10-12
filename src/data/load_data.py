
from pathlib import Path

import pandas as pd


def load_reviews() -> pd.DataFrame:
    """Dada a exploração dos dados, vamos apenas utilizar o filtro de avaliações válidas, sem necessidade de 
    retirada de duplicadas pelo baixo volume e baixo risco"""

    project_root = Path(__file__).resolve().parents[2]

    file_path = (
        project_root
        / "data"
        / "raw"
        / "olist_order_reviews_dataset.csv"
    )

    df = pd.read_csv(file_path)

    df = df[
        df["review_comment_message"].notna()
        & df["review_comment_message"].str.strip().ne("")
    ].copy()

    # Manter os campos necessários para o RAG
    df = df[
        [
            "review_id",
            "order_id",
            "review_score",
            "review_comment_message",
        ]
    ]

    return df
