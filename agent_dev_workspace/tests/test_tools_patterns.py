import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools

def load_fixture_data():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    tools = make_tools(session)
    mine_patterns = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_patterns

def test_mine_patterns_zero_term_case():
    session, mine_patterns = load_fixture_data()
    
    # Known answer #17: filtering_method "variance" and "term_frequency" for catA and catB return n_terms_kept 0
    for f_method in ["variance", "term_frequency"]:
        for cat in ["catA", "catB"]:
            res = mine_patterns.invoke({"category_name": cat, "filtering_method": f_method, "algorithm": "fpgrowth", "min_sup": 1})
            assert res.get("n_terms_kept") == 0
            assert "no terms survived filtering" in res.get("note", "")

def test_mine_patterns_tfidf_fpgrowth():
    session, mine_patterns = load_fixture_data()
    
    # catA min_sup=1: n_terms_before 2, n_terms_kept 1, top pattern ["alpha"] with support 3
    res_a1 = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 1})
    assert res_a1.get("n_terms_before") == 2
    assert res_a1.get("n_terms_kept") == 1
    patterns_a1 = res_a1.get("top_patterns", [])
    assert len(patterns_a1) >= 1
    assert set(patterns_a1[0]["items"]) == {"alpha"}
    assert patterns_a1[0]["support"] == 3

    # catA min_sup=4: 0 patterns
    res_a4 = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 4})
    assert len(res_a4.get("top_patterns", [])) == 0
    assert res_a4.get("n_patterns") == 0

    # catB min_sup=4: ["gamma"] with support 4
    res_b4 = mine_patterns.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 4})
    patterns_b4 = res_b4.get("top_patterns", [])
    assert len(patterns_b4) >= 1
    assert set(patterns_b4[0]["items"]) == {"gamma"}
    assert patterns_b4[0]["support"] == 4

def test_mine_patterns_topk_and_maxfpgrowth():
    session, mine_patterns = load_fixture_data()
    
    # Known answer #20: topk for k=1, 2, 3 returns exactly 1 pattern for catA (alpha, sup 3) and catB (gamma, sup 4)
    for k_val in [1, 2, 3]:
        res_a_topk = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": k_val})
        catA_p = res_a_topk.get("top_patterns", [])
        assert len(catA_p) == 1
        assert set(catA_p[0]["items"]) == {"alpha"}
        assert catA_p[0]["support"] == 3

        res_b_topk = mine_patterns.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "topk", "k": k_val})
        catB_p = res_b_topk.get("top_patterns", [])
        assert len(catB_p) == 1
        assert set(catB_p[0]["items"]) == {"gamma"}
        assert catB_p[0]["support"] == 4

    # Known answer #20: maxfpgrowth with min_sup=1 and min_sup=4 returns 0 patterns for both categories
    for sup in [1, 4]:
        res_a_max = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": sup})
        assert len(res_a_max.get("top_patterns", [])) == 0

        res_b_max = mine_patterns.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": sup})
        assert len(res_b_max.get("top_patterns", [])) == 0

def test_mine_patterns_missing_param_errors():
    session, mine_patterns = load_fixture_data()
    
    # Calling fpgrowth without min_sup returns error dict
    res_no_sup = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth"})
    assert "error" in res_no_sup
    
    # Calling topk without k returns error dict
    res_no_k = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk"})
    assert "error" in res_no_k
