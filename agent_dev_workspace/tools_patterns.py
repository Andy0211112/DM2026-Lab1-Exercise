import sys
import os
import tempfile
from typing import List, Optional, Dict, Any, Literal
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as db_converter
from PAMI.frequentPattern.basic.FPGrowth import FPGrowth
from PAMI.frequentPattern.topk.FAE import FAE
from PAMI.frequentPattern.maximal.MaxFPGrowth import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Mines frequent patterns for a specific document category using specified vocabulary filtering and PAMI algorithms.

        Args:
            category_name: Target category to analyze.
            filtering_method: Metric for filtering vocabulary ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support count required for 'fpgrowth' and 'maxfpgrowth'.
            k: Number of top patterns to return required for 'topk'.

        Returns:
            Dict containing result_id, n_terms_before, n_terms_kept, n_patterns, top_patterns, and status details.
        """
        # Step 0: Check UTF-8 mode on Windows / platform if needed
        if getattr(sys.flags, 'utf8_mode', 0) != 1 and os.name == 'nt':
            if os.environ.get("PYTHONUTF8") != "1":
                return {
                    "error": "UTF-8 mode is not enabled. On Windows, please set PYTHONUTF8=1 in your environment to handle non-ASCII terms with PAMI DF2DB."
                }

        # Step 1: Validate parameter requirements
        if algorithm in ["fpgrowth", "maxfpgrowth"] and min_sup is None:
            return {
                "error": f"Parameter 'min_sup' (int) is required when algorithm='{algorithm}'."
            }
        if algorithm == "topk" and k is None:
            return {
                "error": "Parameter 'k' (int) is required when algorithm='topk'."
            }

        # Step 2: Validate session dataframe & category existence
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataset loaded in session."}

        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            return {"error": f"No documents found for category '{category_name}'."}

        # Step 3: Local DTM with CountVectorizer(stop_words="english")
        texts = df_cat["text"].fillna("").tolist()
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(texts).toarray()
            feature_names = np.array(vectorizer.get_feature_names_out())
        except ValueError:
            # e.g., empty vocabulary after stopword removal
            feature_names = np.array([])
            dtm = np.empty((len(texts), 0))

        n_terms_before = len(feature_names)
        if n_terms_before == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_terms_before": 0,
                "n_terms_kept": 0,
                "n_patterns": 0,
                "top_patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 4: Apply percentile filtering
        if filtering_method == "variance":
            scores = np.var(dtm, axis=0)
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            mask = (scores >= p5) & (scores <= p95)
        elif filtering_method == "term_frequency":
            scores = np.sum(dtm, axis=0)
            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            mask = (scores >= p5) & (scores <= p95)
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_mat = transformer.fit_transform(dtm).toarray()
            scores = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(scores, 20)
            mask = scores >= p20
        else:
            return {"error": f"Unknown filtering method '{filtering_method}'."}

        kept_terms = feature_names[mask]
        kept_dtm = dtm[:, mask]
        n_terms_kept = len(kept_terms)

        if n_terms_kept == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_terms_before": n_terms_before,
                "n_terms_kept": 0,
                "n_patterns": 0,
                "top_patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)
            return summary

        # Step 5: Convert surviving terms into Transactional Database via PAMI DF2DB
        df_binary = pd.DataFrame(kept_dtm, columns=kept_terms)
        
        normalized_patterns = []
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_filepath = os.path.join(tmp_dir, "transactional_db.txt")
            
            converter = db_converter.DF2DB(df_binary)
            converter.convert2TransactionalDatabase(db_filepath, ">=", 1)

            # Step 6: Execute requested PAMI miner
            if algorithm == "fpgrowth":
                miner = FPGrowth(db_filepath, minSup=min_sup)
            elif algorithm == "maxfpgrowth":
                miner = MaxFPGrowth(db_filepath, minSup=min_sup)
            elif algorithm == "topk":
                miner = FAE(db_filepath, k=k)
            else:
                return {"error": f"Unknown algorithm '{algorithm}'."}

            miner.mine()
            raw_patterns = miner.getPatterns()

            # Step 7: Normalize patterns into [{"items": [...], "support": int}, ...]
            for key, sup in raw_patterns.items():
                if isinstance(key, tuple):
                    items = [str(x).strip() for x in key if str(x).strip()]
                elif isinstance(key, str):
                    items = [x.strip() for x in key.split('\t') if x.strip()]
                elif isinstance(key, list):
                    items = [str(x).strip() for x in key if str(x).strip()]
                else:
                    items = [str(key).strip()]
                
                normalized_patterns.append({
                    "items": items,
                    "support": int(sup)
                })

        # Step 8: Sort patterns by support descending
        normalized_patterns.sort(key=lambda p: p["support"], reverse=True)

        # Step 9: Store artifacts and result
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = normalized_patterns

        top_20 = normalized_patterns[:20]
        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_terms_before": n_terms_before,
            "n_terms_kept": n_terms_kept,
            "n_patterns": len(normalized_patterns),
            "top_patterns": top_20
        }
        session.store_result("mine_patterns_tool", {
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "min_sup": min_sup,
            "k": k
        }, summary)

        return summary

    return [mine_patterns_tool]
