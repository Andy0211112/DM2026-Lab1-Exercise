"""Tests for tools in tools_exploration.py using agent_dev/sample_fixture.csv."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_tool():
    """Tests cosine_similarity_tool against known answer #16 in TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known answer #16 tests
    res_0_1 = cosine_sim_tool.invoke({"doc_index_a": 0, "doc_index_b": 1})
    assert res_0_1["similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_3_4 = cosine_sim_tool.invoke({"doc_index_a": 3, "doc_index_b": 4})
    assert res_3_4["similarity"] == pytest.approx(0.9847, abs=1e-3)

    res_0_3 = cosine_sim_tool.invoke({"doc_index_a": 0, "doc_index_b": 3})
    assert res_0_3["similarity"] == pytest.approx(0.0909, abs=1e-3)

    res_0_7 = cosine_sim_tool.invoke({"doc_index_a": 0, "doc_index_b": 7})
    assert res_0_7["similarity"] == pytest.approx(0.0, abs=1e-3)

    # Error handling for out-of-bounds index
    res_out_of_bounds = cosine_sim_tool.invoke({"doc_index_a": 0, "doc_index_b": 99})
    assert "error" in res_out_of_bounds


def test_feature_correlation_matrix_tool():
    """Tests feature_correlation_matrix_tool against known answer #19 in TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Load exploration tools
    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    res = feat_corr_tool.invoke({})

    # Known answer #19 tests
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    corr = res["correlation_matrix"]

    # Diagonal is all 1.0
    for i in range(len(expected_terms)):
        assert corr[i][i] == pytest.approx(1.0, abs=1e-3)

    # Specific correlation assertions
    idx_alpha = expected_terms.index("alpha")
    idx_beta = expected_terms.index("beta")
    idx_gamma = expected_terms.index("gamma")
    idx_delta = expected_terms.index("delta")

    assert corr[idx_alpha][idx_beta] == pytest.approx(0.8885, abs=1e-3)
    assert corr[idx_gamma][idx_delta] == pytest.approx(0.8307, abs=1e-3)
    assert corr[idx_alpha][idx_gamma] == pytest.approx(-0.5718, abs=1e-3)

    # Check pending figure is populated
    assert session.pending_figure is not None
