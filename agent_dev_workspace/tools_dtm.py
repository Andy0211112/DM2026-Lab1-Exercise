from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session) -> List[Any]:
    """Factory function creating DTM tools bound to the session."""

    @tool
    def build_dtm_tool(ngram_range: List[int] = [1, 1]) -> Dict[str, Any]:
        """Build a Document-Term Matrix (DTM) from the text column in session.dataframe.

        Fits a CountVectorizer on session.dataframe['text'], populates
        session.feature_matrix and session.feature_names, saves the fitted vectorizer
        in session.artifacts['count_vectorizer'], sets category labels, and reports DTM sparsity.

        Args:
            ngram_range: Lower and upper boundary of the range of n-values for different
                n-grams to be extracted. Passed as a 2-element list [min_n, max_n]. Default is [1, 1].

        Returns:
            Dict containing result_id, n_documents, n_terms, non_zero, total_elements,
            sparsity_pct, and vocabulary_preview (first 20 terms).
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session.dataframe."}

        if "text" not in session.dataframe.columns:
            return {"error": "session.dataframe does not contain a 'text' column."}

        # Convert list parameter to tuple for CountVectorizer
        ngram_tuple = tuple(ngram_range) if isinstance(ngram_range, list) else (1, 1)

        vectorizer = CountVectorizer(ngram_range=ngram_tuple)
        matrix = vectorizer.fit_transform(session.dataframe["text"])
        feature_names = list(vectorizer.get_feature_names_out())

        # Update session state
        session.feature_matrix = matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].to_numpy())

        # Sparsity calculations matching course spec
        n_documents, n_terms = matrix.shape
        non_zero = int(matrix.nnz)
        total_elements = int(n_documents * n_terms)
        sparsity_pct = float(100.0 * (1.0 - (non_zero / total_elements))) if total_elements > 0 else 0.0

        vocabulary_preview = feature_names[:20]

        summary = {
            "result_id": session.next_result_id("dtm"),
            "n_documents": n_documents,
            "n_terms": n_terms,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_preview": vocabulary_preview,
        }

        session.store_result("build_dtm_tool", {"ngram_range": ngram_range}, summary)
        return summary

    @tool
    def term_frequency_tool(top_k: int = 20) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in session.feature_matrix.

        Sums term counts across documents, returns top_k most frequent terms,
        and stores full frequency table as full_report for visualization.

        Args:
            top_k: Number of top frequent terms to return in summary. Default is 20.

        Returns:
            Dict containing result_id, vocabulary_size, top_k, and top_terms list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Run build_dtm_tool first."}

        # Sum total count per term across all documents (column sums)
        term_counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()

        df_freq = pd.DataFrame({
            "term": session.feature_names,
            "frequency": term_counts
        }).sort_values(by="frequency", ascending=False).reset_index(drop=True)

        vocabulary_size = len(df_freq)
        top_df = df_freq.head(top_k)

        top_terms = top_df.to_dict(orient="records")

        summary = {
            "result_id": session.next_result_id("term_freq"),
            "vocabulary_size": vocabulary_size,
            "top_k": top_k,
            "top_terms": top_terms,
        }

        session.store_result("term_frequency_tool", {"top_k": top_k}, summary, full_report=df_freq)
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Render a heatmap of a positional slice of the raw Document-Term Matrix.

        Slices session.feature_matrix[0:n_documents, 0:n_terms], generates an annotated
        heatmap on session.pending_figure, and returns the matrix slice as a nested list.

        Args:
            n_terms: Maximum number of vocabulary terms (columns) to include. Default is 20.
            n_documents: Maximum number of documents (rows) to include. Default is 20.

        Returns:
            Dict containing result_id, slice_shape, terms, document_indices, and matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Run build_dtm_tool first."}

        max_docs, max_terms = session.feature_matrix.shape
        actual_docs = min(n_documents, max_docs)
        actual_terms = min(n_terms, max_terms)

        matrix_slice = session.feature_matrix[:actual_docs, :actual_terms].toarray()
        selected_terms = session.feature_names[:actual_terms]
        doc_indices = list(range(actual_docs))

        # Build figure and heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_terms * 0.5), max(4, actual_docs * 0.4)))
        sns.heatmap(
            matrix_slice,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=selected_terms,
            yticklabels=doc_indices,
            ax=ax
        )
        ax.set_title(f"DTM Heatmap (First {actual_docs} Docs x First {actual_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Convert matrix slice to list of lists of plain Python ints
        nested_matrix = matrix_slice.astype(int).tolist()

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "slice_shape": [actual_docs, actual_terms],
            "terms": selected_terms,
            "document_indices": doc_indices,
            "matrix": nested_matrix,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary
        )
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
