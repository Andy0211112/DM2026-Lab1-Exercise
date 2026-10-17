import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal, Optional, List, Dict, Any
from langchain_core.tools import tool

from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        max_terms: int = 1000
    ) -> Dict[str, Any]:
        """Reduces the dimensionality of the document-term matrix using PCA, t-SNE, or UMAP.

        Filters the DTM to top max_terms by population variance if dense conversion is needed,
        stores reduced coordinates in session.artifacts, and creates a scatter plot of the
        first two dimensions colored by category_name on session.pending_figure.

        Args:
            method: Dimensionality reduction technique ('pca', 'tsne', or 'umap'). Default is 'pca'.
            n_components: Number of target dimensions. Default is 2.
            random_state: Seed for reproducible decomposition. Default is 42.
            perplexity: Perplexity parameter for t-SNE algorithm. Default is 30.0.
            n_neighbors: Number of neighbors parameter for UMAP algorithm. Default is 15.
            max_terms: Maximum top-variance vocabulary terms to retain prior to densification. Default is 1000.

        Returns:
            A summary dictionary containing reduction parameters, explained variance ratio (for PCA),
            perplexity_used / n_neighbors_used if adjusted, and a 5-document coordinate preview.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No feature_matrix found in session. Run build_dtm_tool first."}

        X = session.feature_matrix
        n_docs, n_terms = X.shape

        # Densification safeguard: pick top max_terms by population variance on sparse matrix
        if n_terms > max_terms:
            # Population variance for CSR matrix columns: E[X^2] - (E[X])^2
            mean = np.array(X.mean(axis=0)).ravel()
            X_sq = X.copy()
            X_sq.data **= 2
            mean_sq = np.array(X_sq.mean(axis=0)).ravel()
            var = mean_sq - mean ** 2
            
            top_indices = np.argsort(var)[::-1][:max_terms]
            X_sub = X[:, top_indices]
            n_terms_used = int(max_terms)
        else:
            X_sub = X
            n_terms_used = int(n_terms)

        # Algorithm fitting
        explained_variance_ratio = None
        perplexity_used = None
        n_neighbors_used = None

        if method == "pca":
            from sklearn.decomposition import PCA
            X_dense = (X_sub.toarray() if hasattr(X_sub, "toarray") else X_sub).astype(np.float64)
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(X_dense)
            explained_variance_ratio = [float(x) for x in model.explained_variance_ratio_]
        elif method == "tsne":
            from sklearn.manifold import TSNE
            X_dense = (X_sub.toarray() if hasattr(X_sub, "toarray") else X_sub).astype(np.float64)
            # Adjust perplexity if dataset is smaller than default recommendation
            perplexity_used = float(min(perplexity, max(1.0, float(n_docs - 1))))
            model = TSNE(n_components=n_components, perplexity=perplexity_used, random_state=random_state)
            coords = model.fit_transform(X_dense)
        elif method == "umap":
            import umap
            X_dense = (X_sub.toarray() if hasattr(X_sub, "toarray") else X_sub).astype(np.float64)
            # Adjust n_neighbors if dataset is small
            n_neighbors_used = int(min(n_neighbors, max(2, n_docs - 1)))
            model = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors_used, random_state=random_state)
            coords = model.fit_transform(X_dense)
        else:
            return {"status": "error", "message": f"Unsupported method '{method}'. Choose 'pca', 'tsne', or 'umap'."}

        # Store full coordinates array in artifacts
        coords_array = np.asarray(coords, dtype=float)
        session.artifacts["reduced_coordinates"] = coords_array
        session.artifacts[f"reduced_coordinates_{method}"] = coords_array

        # Generate scatter plot for session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Determine labels / category colors
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            cat_labels = session.dataframe["category_name"].tolist()
        elif session.labels is not None:
            cat_labels = list(session.labels)
        else:
            cat_labels = ["default"] * n_docs

        unique_cats = sorted(list(set(cat_labels)))
        cat_array = np.array(cat_labels)

        for cat in unique_cats:
            mask = cat_array == cat
            ax.scatter(coords_array[mask, 0], coords_array[mask, 1], label=str(cat), alpha=0.7)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Category")
        plt.tight_layout()

        session.pending_figure = fig
        plt.close(fig)

        # Store result summary
        result_id = session.next_result_id(f"reduce_{method}")
        preview_coords = coords_array[:5].tolist()

        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_terms_used": n_terms_used,
            "explained_variance_ratio": explained_variance_ratio,
            "coordinates_preview": preview_coords
        }

        if perplexity_used is not None:
            summary["perplexity_used"] = perplexity_used
        if n_neighbors_used is not None:
            summary["n_neighbors_used"] = n_neighbors_used

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "random_state": random_state,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "max_terms": max_terms
            },
            summary=summary,
            full_report={"coordinates": coords_array.tolist()}
        )

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarizes categorical labels in session.dataframe['category_name'] using LabelBinarizer.

        Stores the encoded vectors as lists in a new 'bin_category' column in session.dataframe.

        Returns:
            A summary dictionary containing the result_id, classes in order, shape of the
            encoded matrix, and a preview of the first 5 rows with category_name and bin_category.
        """
        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset with 'category_name' loaded in session."}

        from sklearn.preprocessing import LabelBinarizer

        lb = LabelBinarizer()
        category_names = session.dataframe["category_name"].tolist()
        encoded = lb.fit_transform(category_names)

        # Store each row's vector as a list in 'bin_category'
        bin_category_list = [row.tolist() for row in encoded]
        session.dataframe["bin_category"] = bin_category_list

        classes = [str(c) for c in lb.classes_]
        matrix_shape = list(encoded.shape)

        # Store binarizer artifact
        session.artifacts["label_binarizer"] = lb

        # Create 5-row preview
        preview = []
        for cat_name, bin_vec in zip(category_names[:5], bin_category_list[:5]):
            preview.append({"category_name": cat_name, "bin_category": bin_vec})

        result_id = session.next_result_id("binarize")

        summary = {
            "result_id": result_id,
            "classes": classes,
            "encoded_shape": matrix_shape,
            "preview": preview
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
            full_report={"bin_category": bin_category_list}
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
