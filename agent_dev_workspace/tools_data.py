import os
from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt
from langchain_core.tools import tool
import nltk
from nltk.tokenize import word_tokenize
import pandas as pd
import seaborn as sns

# Ensure NLTK 'punkt' / 'punkt_tab' resource is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _resolve_within_repo(target_path_str: str) -> Path:
    """Resolve a path string relative to repo root (CWD) and verify it remains inside repo root."""
    repo_root = Path.cwd().resolve()
    target = (repo_root / target_path_str).resolve()
    try:
        target.relative_to(repo_root)
    except ValueError:
        raise ValueError(f"Path '{target_path_str}' escapes repository root.")
    return target


def make_tools(session) -> list:
    """Factory function to instantiate tools for dataset inspection and data quality checks.

    Args:
        session: The shared SessionState object.

    Returns:
        List of initialized LangChain tools.
    """

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenize each document's text into lowercased unigrams using NLTK.

        Stores the token lists as a new column session.dataframe['unigrams'].
        An empty text string yields an empty list [].

        Returns:
            Dict containing result_id, n_documents, total_tokens, and token_preview (first 3 docs, top 10 tokens each + total tokens).
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first.",
            }

        df = session.dataframe

        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "'text' column not found in session.dataframe.",
            }

        def tokenize_doc(text: Any) -> list:
            if pd.isna(text) or not str(text).strip():
                return []
            return word_tokenize(str(text).lower())

        # Tokenize all rows and assign to unigrams column
        unigrams_series = df["text"].apply(tokenize_doc)
        session.dataframe["unigrams"] = unigrams_series

        n_documents = int(len(df))
        total_tokens = int(unigrams_series.apply(len).sum())

        # Preview first 3 documents: top 10 tokens and total count for each
        preview_doc_count = min(3, n_documents)
        token_preview = []
        for i in range(preview_doc_count):
            doc_tokens = unigrams_series.iloc[i]
            token_preview.append({
                "doc_index": i,
                "first_10_tokens": doc_tokens[:10],
                "doc_total_tokens": len(doc_tokens),
            })

        res_id = session.next_result_id("tokenize")

        summary = {
            "result_id": res_id,
            "n_documents": n_documents,
            "total_tokens": total_tokens,
            "token_preview": token_preview,
        }

        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary,
        )

        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists directories and files in a repo subdirectory.

        Resolves subdirectory relative to repository root (CWD) and checks that it stays within bounds.

        Args:
            subdirectory: Subdirectory path relative to repo root. Default is "." (repo root).

        Returns:
            Dict containing result_id, subdirectory, directories, and files (sorted lists).
        """
        try:
            target_path = _resolve_within_repo(subdirectory)
        except ValueError as e:
            return {
                "status": "error",
                "message": str(e),
            }

        if not target_path.exists():
            return {
                "status": "error",
                "message": f"Subdirectory '{subdirectory}' does not exist.",
            }

        if not target_path.is_dir():
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' is not a directory.",
            }

        directories = []
        files = []

        for entry in os.scandir(target_path):
            if entry.is_dir():
                directories.append(entry.name)
            elif entry.is_file():
                files.append(entry.name)

        directories.sort()
        files.sort()

        res_id = session.next_result_id("list_files")

        summary = {
            "result_id": res_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary,
        )

        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check for missing or empty text values in session.dataframe.

        Examines the 'text' column for NaN/null values or empty/whitespace-only strings.

        Returns:
            Dict containing result_id, missing_count, total_rows, and missing_indices.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first.",
            }

        df = session.dataframe

        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "'text' column not found in session.dataframe.",
            }

        # Check for NaN/null OR whitespace-only strings in the 'text' column
        is_missing = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
        missing_df = df[is_missing]

        missing_count = int(len(missing_df))
        total_rows = int(len(df))
        # Use DataFrame index labels (e.g. df.index) as requested
        missing_indices = list(missing_df.index)

        res_id = session.next_result_id("check_missing")

        summary = {
            "result_id": res_id,
            "missing_count": missing_count,
            "total_rows": total_rows,
            "missing_indices": missing_indices,
        }

        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary,
        )

        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check for and optionally drop duplicate rows in session.dataframe.

        Matches Master's behavior by comparing entire rows across all columns.
        Counting uses keep='first' (so 1 duplicate pair counts as 1 duplicate).
        When dropping (drop=True), uses keep=False (dropping all copies of duplicate rows).

        Args:
            drop: If True, removes all duplicate rows from session.dataframe and resets index. Default False.

        Returns:
            Dict containing result_id, duplicate_count, duplicate_indices, rows_to_drop,
            dropped status, and (if dropped) rows_before and rows_after.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first.",
            }

        df = session.dataframe

        # Count extra duplicate rows (keep="first" marks duplicate copies, e.g. 1 duplicate for 1 pair)
        extra_duplicates_mask = df.duplicated(keep="first")
        duplicate_count = int(extra_duplicates_mask.sum())
        duplicate_indices = [int(idx) for idx in df.index[extra_duplicates_mask].tolist()]

        # All rows involved in duplication (keep=False marks every copy, e.g. rows 5 and 6)
        all_duplicates_mask = df.duplicated(keep=False)
        rows_to_drop = [int(idx) for idx in df.index[all_duplicates_mask].tolist()]

        res_id = session.next_result_id("check_duplicates")

        summary: Dict[str, Any] = {
            "result_id": res_id,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "rows_to_drop": rows_to_drop,
            "dropped": drop,
        }

        if drop:
            rows_before = int(len(df))
            # Drop every copy of a duplicated row (keep=False) as required, then reset index
            session.dataframe = df.drop_duplicates(keep=False).reset_index(drop=True)
            rows_after = int(len(session.dataframe))

            summary["rows_before"] = rows_before
            summary["rows_after"] = rows_after

        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary,
        )

        return summary

    @tool
    def sample_data_tool(n: int, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Subsamples session.dataframe without mutating the original dataframe.

        Stores the sampled DataFrame in session.artifacts['sample'] with index reset.

        Args:
            n: Number of items to sample.
            random_state: Optional seed for reproducible sampling.

        Returns:
            Dict containing result_id, sampled_indices, n_sampled, total_rows, and counts_per_category.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first.",
            }

        df = session.dataframe
        total_rows = int(len(df))

        if n > total_rows:
            return {
                "status": "error",
                "message": f"Requested sample size n={n} exceeds total available rows ({total_rows}).",
            }

        if n <= 0:
            return {
                "status": "error",
                "message": f"Sample size n must be a positive integer, got {n}.",
            }

        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = [int(idx) for idx in sampled_df.index.tolist()]

        # Store sample in session.artifacts with reset index
        session.artifacts["sample"] = sampled_df.reset_index(drop=True)

        # Compute category distribution in sample if category column exists
        counts_per_category: Dict[str, int] = {}
        category_col = "category_name" if "category_name" in sampled_df.columns else ("category" if "category" in sampled_df.columns else None)
        if category_col:
            counts_per_category = {str(k): int(v) for k, v in sampled_df[category_col].value_counts().to_dict().items()}

        res_id = session.next_result_id("sample_data")

        summary = {
            "result_id": res_id,
            "sampled_indices": sampled_indices,
            "n_sampled": len(sampled_indices),
            "total_rows": total_rows,
            "counts_per_category": counts_per_category,
        }

        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary,
        )

        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Compute summary statistics on document text length and plot a boxplot by category.

        Computes text length as character count without adding it to session.dataframe.
        Returns .describe() statistics (count, mean, std, min, 25%, 50%, 75%, max)
        overall and per category_name, converted to plain Python floats.
        Generates a seaborn boxplot on its own figure, assigns it to session.pending_figure,
        and closes the figure.

        Returns:
            Dict containing result_id, overall_stats, and per_category_stats.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first.",
            }

        df = session.dataframe

        if "text" not in df.columns:
            return {
                "status": "error",
                "message": "'text' column not found in session.dataframe.",
            }

        # Compute text length as character count using df["text"].apply(len)
        text_length = df["text"].apply(len)

        # Helper to convert pandas describe() Series to plain python dict of floats
        def desc_to_dict(desc_series: pd.Series) -> Dict[str, float]:
            return {stat: float(val) for stat, val in desc_series.to_dict().items()}

        overall_stats = desc_to_dict(text_length.describe())

        per_category_stats: Dict[str, Dict[str, float]] = {}
        category_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        if category_col:
            # Create temporary DataFrame for grouping by category
            temp_df = pd.DataFrame({
                "text_length": text_length,
                "category_name": df[category_col].astype(str),
            })
            for cat_name, group in temp_df.groupby("category_name"):
                per_category_stats[cat_name] = desc_to_dict(group["text_length"].describe())

            # Plot boxplot on its own figure
            fig, ax = plt.subplots()
            sns.boxplot(data=temp_df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length Distribution by Category")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length (characters)")
            session.pending_figure = fig
            plt.close(fig)

        res_id = session.next_result_id("describe_data")

        summary = {
            "result_id": res_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
        )

        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Peeks at the first n rows of session.dataframe.

        Args:
            n: Number of rows to inspect (must be between 1 and 20). Default is 5.

        Returns:
            Dict containing result_id, total_rows, columns, n_inspected, and rows (a list of row previews).
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Please load a dataset first.",
            }

        if n < 1 or n > 20:
            return {
                "status": "error",
                "message": f"n must be between 1 and 20, got {n}.",
            }

        df = session.dataframe
        total_rows = int(len(df))
        columns = [str(c) for c in df.columns.tolist()]

        # Slice the first n rows
        head_df = df.head(n)

        category_col = "category_name" if "category_name" in df.columns else ("category" if "category" in df.columns else None)

        rows = []
        for idx, row in head_df.iterrows():
            raw_text = str(row["text"]) if "text" in df.columns and pd.notna(row["text"]) else ""
            char_length = len(raw_text)
            truncated_text = raw_text[:200]

            cat_val = str(row[category_col]) if category_col and pd.notna(row[category_col]) else None

            rows.append({
                "index": int(idx) if isinstance(idx, (int, pd.Series)) else str(idx),
                "category_name": cat_val,
                "text_truncated": truncated_text,
                "text_length": char_length,
            })

        res_id = session.next_result_id("inspect_data")

        summary = {
            "result_id": res_id,
            "total_rows": total_rows,
            "columns": columns,
            "n_inspected": len(rows),
            "rows": rows,
        }

        session.store_result(
            tool_name="inspect_data_tool",
            args={"n": n},
            summary=summary,
        )

        return summary

    return [
        tokenize_tool,
        list_files_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        inspect_data_tool,
    ]
