"""Filtering tools for feature selection in text mining."""

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from scipy.stats import rankdata
from langchain_core.tools import tool


def make_tools(session):
    """Factory function to build filtering tools bound to a shared session state."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Flags terms in the DTM with variance at or below a given threshold for removal.

        Computes term-level population variance directly on the sparse feature matrix
        (ddof=0) without converting it to a dense matrix. Stores kept and removed
        terms in session.artifacts and full report dataframe for downstream tools.

        Args:
            threshold: Minimum variance required to keep a term. Terms with variance
                less than or equal to threshold are flagged for removal.

        Returns:
            Dict containing result_id, summary statistics (n_terms, n_kept, n_removed),
            and previews of kept and removed terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature matrix found in session. Please build DTM first using build_dtm_tool."
            }

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_docs = X.shape[0]

        if n_docs == 0:
            return {"error": "Feature matrix has zero documents."}

        # Calculate population variance (ddof=0) efficiently on sparse or dense matrices:
        # Var(X) = E[X^2] - (E[X])^2
        if issparse(X):
            mean = np.array(X.mean(axis=0)).ravel()
            mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
            variances = mean_sq - (mean ** 2)
        else:
            variances = np.var(X, axis=0, ddof=0)

        # Identify terms to keep and remove
        # Flag for removal if variance <= threshold
        kept_mask = variances > threshold
        removed_mask = ~kept_mask

        kept_terms = feature_names[kept_mask].tolist()
        removed_terms = feature_names[removed_mask].tolist()

        # Store term lists in artifacts for down-pipeline tools
        session.artifacts["variance_kept_terms"] = kept_terms
        session.artifacts["variance_removed_terms"] = removed_terms
        session.artifacts["variance_threshold"] = threshold

        # Prepare full_report DataFrame sorted by variance descending
        # First column: "term", Second column: "variance" (required for visualize_result_tool)
        report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        # Previews for summary to avoid context flooding on large vocabularies
        kept_preview = kept_terms[:20]
        removed_preview = removed_terms[:20]

        summary = {
            "result_id": session.next_result_id("variance"),
            "n_terms": int(len(feature_names)),
            "n_kept": int(len(kept_terms)),
            "n_removed": int(len(removed_terms)),
            "threshold": float(threshold),
            "kept_terms_preview": kept_preview,
            "removed_terms_preview": removed_preview,
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, min_abs_r: float = 0.0) -> dict:
        """Computes Pearson correlation between each term count and a target class indicator.

        Performs one-vs-rest correlation on sparse feature matrix. Handles zero variance
        by setting r = 0.0. Keeps terms with |r| >= min_abs_r.

        Args:
            target_class: Category label to treat as the positive class (1 vs 0).
            min_abs_r: Minimum absolute correlation coefficient required to keep a term.

        Returns:
            Dict containing result_id, n_terms, n_kept, and top_terms_preview (top 20 terms
            with their Pearson r values sorted by absolute correlation descending).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature matrix found in session. Please build DTM first using build_dtm_tool."
            }

        if session.labels is None or session.categories is None:
            return {"error": "No category labels found in session."}

        if target_class not in session.categories:
            return {
                "error": f"target_class '{target_class}' not found in session categories: {session.categories}"
            }

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_docs = X.shape[0]

        if n_docs == 0:
            return {"error": "Feature matrix has zero documents."}

        # One-vs-rest binary target vector y (1 if label == target_class else 0)
        y = (np.array(session.labels) == target_class).astype(float)

        y_mean = y.mean()
        y_diff = y - y_mean
        y_var = np.sum(y_diff ** 2)

        if y_var == 0:
            # Target class has no variation (e.g. all 1s or all 0s)
            pearson_r = np.zeros(len(feature_names))
        else:
            if issparse(X):
                x_mean = np.array(X.mean(axis=0)).ravel()
                # Compute (X - x_mean)^T * (y - y_mean) efficiently:
                # X^T * y_diff - x_mean * sum(y_diff) -> sum(y_diff) == 0, so X^T * y_diff!
                cov = np.array(X.T.dot(y_diff)).ravel()

                # Sum of squared differences for X: sum((X - x_mean)^2) = N * Var(X)
                x_mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
                x_var_sum = (x_mean_sq - (x_mean ** 2)) * n_docs
            else:
                x_mean = X.mean(axis=0)
                x_diff = X - x_mean
                cov = np.dot(x_diff.T, y_diff)
                x_var_sum = np.sum(x_diff ** 2, axis=0)

            # Avoid division by zero: if x_var_sum == 0 or y_var == 0, r = 0.0
            denom = np.sqrt(x_var_sum * y_var)
            with np.errstate(divide='ignore', invalid='ignore'):
                pearson_r = np.where(denom > 0, cov / denom, 0.0)
                pearson_r = np.nan_to_num(pearson_r, nan=0.0)

        abs_r = np.abs(pearson_r)
        kept_mask = abs_r >= min_abs_r

        kept_terms = feature_names[kept_mask].tolist()
        session.artifacts["pearson_kept_terms"] = kept_terms

        # Create full_report DataFrame sorted by |r| descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r,
            "abs_r": abs_r
        }).sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        # Build top 20 preview dict for summary
        top_preview_df = report_df.head(20)
        top_terms_preview = [
            {"term": row["term"], "pearson_r": float(row["pearson_r"])}
            for _, row in top_preview_df.iterrows()
        ]

        summary = {
            "result_id": session.next_result_id("pearson"),
            "target_class": target_class,
            "min_abs_r": float(min_abs_r),
            "n_terms": int(len(feature_names)),
            "n_kept": int(len(kept_terms)),
            "top_terms_preview": top_terms_preview
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "min_abs_r": min_abs_r},
            summary=summary,
            full_report=report_df
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, min_abs_r: float = 0.0) -> dict:
        """Computes Spearman correlation between each term count and a target class indicator.

        Ranks term counts chunk-by-chunk (to avoid densifying large sparse matrices at once)
        and computes Pearson correlation on the ranked values. Handles zero-variance terms
        by setting r = 0.0. Keeps terms with |r| >= min_abs_r.

        Args:
            target_class: Category label to treat as the positive class (1 vs 0).
            min_abs_r: Minimum absolute Spearman correlation coefficient required to keep a term.

        Returns:
            Dict containing result_id, n_terms, n_kept, and top_terms_preview (top 20 terms
            with their Spearman r values sorted by absolute correlation descending).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No feature matrix found in session. Please build DTM first using build_dtm_tool."
            }

        if session.labels is None or session.categories is None:
            return {"error": "No category labels found in session."}

        if target_class not in session.categories:
            return {
                "error": f"target_class '{target_class}' not found in session categories: {session.categories}"
            }

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)
        n_docs, n_terms = X.shape

        if n_docs == 0:
            return {"error": "Feature matrix has zero documents."}

        # One-vs-rest binary target vector y
        y = (np.array(session.labels) == target_class).astype(float)
        # Rank y (with average ties)
        y_ranked = rankdata(y, method="average")
        y_mean = y_ranked.mean()
        y_diff = y_ranked - y_mean
        y_var_sum = np.sum(y_diff ** 2)

        spearman_r = np.zeros(n_terms, dtype=float)

        if y_var_sum > 0:
            chunk_size = 1000
            for start_col in range(0, n_terms, chunk_size):
                end_col = min(start_col + chunk_size, n_terms)
                X_chunk = X[:, start_col:end_col]
                
                if issparse(X_chunk):
                    X_dense = X_chunk.toarray()
                else:
                    X_dense = np.array(X_chunk)

                # Rank each column across documents (axis 0)
                # rankdata returns 1D array if 1 column or flattens unless handled, so rank along axis 0:
                X_ranked = np.apply_along_axis(rankdata, 0, X_dense, method="average")

                x_mean = X_ranked.mean(axis=0)
                x_diff = X_ranked - x_mean
                cov = np.dot(x_diff.T, y_diff)
                x_var_sum = np.sum(x_diff ** 2, axis=0)

                denom = np.sqrt(x_var_sum * y_var_sum)
                with np.errstate(divide='ignore', invalid='ignore'):
                    chunk_r = np.where(denom > 0, cov / denom, 0.0)
                    chunk_r = np.nan_to_num(chunk_r, nan=0.0)

                spearman_r[start_col:end_col] = chunk_r

        abs_r = np.abs(spearman_r)
        kept_mask = abs_r >= min_abs_r

        kept_terms = feature_names[kept_mask].tolist()
        session.artifacts["spearman_kept_terms"] = kept_terms

        # Create full_report DataFrame sorted by |r| descending
        report_df = pd.DataFrame({
            "term": feature_names,
            "spearman_r": spearman_r,
            "abs_r": abs_r
        }).sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        # Build top 20 preview dict for summary
        top_preview_df = report_df.head(20)
        top_terms_preview = [
            {"term": row["term"], "spearman_r": float(row["spearman_r"])}
            for _, row in top_preview_df.iterrows()
        ]

        summary = {
            "result_id": session.next_result_id("spearman"),
            "target_class": target_class,
            "min_abs_r": float(min_abs_r),
            "n_terms": int(len(feature_names)),
            "n_kept": int(len(kept_terms)),
            "top_terms_preview": top_terms_preview
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "min_abs_r": min_abs_r},
            summary=summary,
            full_report=report_df
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
