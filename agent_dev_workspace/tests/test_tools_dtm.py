import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def setup_fixture_session():
    """Helper to create a fresh session and load sample_fixture.csv."""
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Instantiate workspace DTM tools
    tools = make_tools(session)
    tools_dict = {t.name: t for t in tools}

    return session, tools_dict


def test_build_dtm_tool():
    session, tools = setup_fixture_session()
    build_dtm_tool = tools["build_dtm_tool"]

    result = build_dtm_tool.invoke({"ngram_range": [1, 1]})

    assert result["n_documents"] == 8
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)

    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool():
    session, tools = setup_fixture_session()
    term_frequency_tool = tools["term_frequency_tool"]

    # Calling before build_dtm_tool should return an error
    error_result = term_frequency_tool.invoke({"top_k": 20})
    assert "error" in error_result

    # Build DTM first
    tools["build_dtm_tool"].invoke({"ngram_range": [1, 1]})

    result = term_frequency_tool.invoke({"top_k": 20})

    assert result["vocabulary_size"] == 5

    # Known answer #14 frequency totals
    expected_freqs = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    # Convert top_terms list of dicts to term -> frequency mapping
    actual_freqs = {item["term"]: item["frequency"] for item in result["top_terms"]}
    assert actual_freqs == expected_freqs


def test_dtm_heatmap_tool():
    session, tools = setup_fixture_session()
    dtm_heatmap_tool = tools["dtm_heatmap_tool"]

    # Calling before build_dtm_tool should return an error
    error_result = dtm_heatmap_tool.invoke({})
    assert "error" in error_result

    # Build DTM first
    tools["build_dtm_tool"].invoke({"ngram_range": [1, 1]})

    # Test default slice (full 8x5 table)
    default_result = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert default_result["slice_shape"] == [8, 5]
    assert default_result["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert default_result["document_indices"] == list(range(8))

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert default_result["matrix"] == expected_full_matrix
    assert session.pending_figure is not None

    # Test 3x2 slice
    slice_result = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert slice_result["slice_shape"] == [3, 2]
    assert slice_result["terms"] == ["alpha", "always"]
    assert slice_result["document_indices"] == [0, 1, 2]
    assert slice_result["matrix"] == [[3, 1], [2, 1], [1, 1]]
