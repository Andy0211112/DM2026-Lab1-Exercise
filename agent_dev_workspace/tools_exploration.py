"""Tools for exploring relationships between documents and features."""

from typing import Any, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session: Any) -> list:
    """Factory function that creates and returns exploration tools bound to the session.

    Args:
        session: The shared SessionState instance.

    Returns:
        List of initialized LangChain tools.
    """

    @tool
    def cosine_similarity_tool(doc_index_a: int, doc_index_b: int) -> Dict[str, Any]:
        """Calculates the cosine similarity between two documents in session.dataframe.

        Vectorizes each document's text using the saved count_vectorizer from
        session.artifacts.

        Args:
            doc_index_a: Positional index (row number) of the first document.
            doc_index_b: Positional index (row number) of the second document.

        Returns:
            Dict containing result_id, doc_index_a, doc_index_b, similarity score,
            and document metadata/snippets.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded in session."}

        vectorizer = session.artifacts.get("count_vectorizer")
        if vectorizer is None:
            return {"error": "No vectorizer found in session.artifacts. Run build_dtm_tool first."}

        n_rows = len(session.dataframe)
        if not (0 <= doc_index_a < n_rows) or not (0 <= doc_index_b < n_rows):
            return {
                "error": f"Document index out of range. Valid range is 0 to {n_rows - 1}."
            }

        doc_a_row = session.dataframe.iloc[doc_index_a]
        doc_b_row = session.dataframe.iloc[doc_index_b]

        text_a = str(doc_a_row["text"]) if doc_a_row["text"] is not None else ""
        text_b = str(doc_b_row["text"]) if doc_b_row["text"] is not None else ""

        vec_a = vectorizer.transform([text_a]).toarray()[0]
        vec_b = vectorizer.transform([text_b]).toarray()[0]

        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0.0 or norm_b == 0.0:
            similarity = 0.0
        else:
            similarity = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_a": doc_index_a,
            "doc_index_b": doc_index_b,
            "similarity": similarity,
            "doc_a": {
                "category_name": str(doc_a_row.get("category_name", "")),
                "snippet": text_a[:100],
            },
            "doc_b": {
                "category_name": str(doc_b_row.get("category_name", "")),
                "snippet": text_b[:100],
            },
        }

        session.store_result("cosine_similarity_tool", {"doc_index_a": doc_index_a, "doc_index_b": doc_index_b}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation over top terms by variance.

        Selects top terms by population variance (up to 20), computes their pairwise
        Pearson correlation matrix, and builds an annotated heatmap plot stored on
        session.pending_figure.

        Returns:
            Dict containing result_id, selected terms, and correlation matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No DTM in session. Run build_dtm_tool first."}

        dtm = session.feature_matrix
        feature_names = np.array(session.feature_names)

        # Compute population variance directly on the sparse matrix: Var(X) = E[X^2] - (E[X])^2
        mean = np.array(dtm.mean(axis=0)).ravel()
        mean_sq = np.array(dtm.power(2).mean(axis=0)).ravel()
        variances = mean_sq - (mean ** 2)

        top_k = min(20, len(feature_names))
        sorted_indices = np.argsort(-variances)[:top_k]

        selected_terms = feature_names[sorted_indices].tolist()

        # Densify only the selected top-k columns
        sub_matrix = dtm[:, sorted_indices].toarray()

        corr_matrix = np.corrcoef(sub_matrix, rowvar=False)

        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            xticklabels=selected_terms,
            yticklabels=selected_terms,
            cmap="vlag",
            ax=ax,
        )
        ax.set_title("Top Feature Correlation Matrix")
        fig.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        corr_matrix_list = [
            [round(float(val), 4) for val in row] for row in corr_matrix
        ]

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "terms": selected_terms,
            "correlation_matrix": corr_matrix_list,
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
