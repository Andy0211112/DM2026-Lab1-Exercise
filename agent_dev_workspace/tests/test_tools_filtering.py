"""Tests for tools_filtering.py against TEST_FIXTURE.md known answers #5 and #6."""

import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def _setup_dtm_session():
    """Helper to initialize a fresh session, load sample_fixture.csv, and build DTM."""
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM using student's build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_variance_filter_tool():
    """Test variance_filter_tool against Known Answer #5."""
    session = _setup_dtm_session()
    filtering_tools = make_filtering_tools(session)
    variance_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold=0.15
    result = variance_tool.invoke({"threshold": 0.15})

    # Summary asserts
    assert result["n_terms"] == 5
    assert result["n_kept"] == 4
    assert result["n_removed"] == 1
    assert "always" in result["removed_terms_preview"]
    assert set(result["kept_terms_preview"]) == {"alpha", "beta", "delta", "gamma"}

    # Artifacts asserts
    assert set(session.artifacts["variance_kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert session.artifacts["variance_removed_terms"] == ["always"]

    # Verify session.feature_matrix shape is UNCHANGED (still 5 columns)
    assert session.feature_matrix.shape == (8, 5)

    # Verify full_report variances against Known Answer #5
    full_report = session.results_store[result["result_id"]]["full_report"]
    var_dict = dict(zip(full_report["term"], full_report["variance"]))

    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    for term, exp_var in expected_variances.items():
        assert var_dict[term] == pytest.approx(exp_var, abs=1e-4)


def test_pearson_filter_tool():
    """Test pearson_filter_tool against Known Answer #6."""
    session = _setup_dtm_session()
    filtering_tools = make_filtering_tools(session)
    pearson_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run with target_class="catB"
    result = pearson_tool.invoke({"target_class": "catB", "min_abs_r": 0.0})

    assert result["n_terms"] == 5
    assert result["n_kept"] == 5

    # Check Pearson r values against Known Answer #6
    full_report = session.results_store[result["result_id"]]["full_report"]
    r_dict = dict(zip(full_report["term"], full_report["pearson_r"]))

    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, exp_r in expected_pearson.items():
        assert r_dict[term] == pytest.approx(exp_r, abs=1e-3)

    # Test error handling for non-existent target class
    error_result = pearson_tool.invoke({"target_class": "non_existent_class"})
    assert "error" in error_result


def test_spearman_filter_tool():
    """Test spearman_filter_tool against Known Answer #6."""
    session = _setup_dtm_session()
    filtering_tools = make_filtering_tools(session)
    spearman_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run with target_class="catB"
    result = spearman_tool.invoke({"target_class": "catB", "min_abs_r": 0.0})

    assert result["n_terms"] == 5
    assert result["n_kept"] == 5

    # Check Spearman r values against Known Answer #6
    full_report = session.results_store[result["result_id"]]["full_report"]
    r_dict = dict(zip(full_report["term"], full_report["spearman_r"]))

    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for term, exp_r in expected_spearman.items():
        assert r_dict[term] == pytest.approx(exp_r, abs=1e-3)

    # Test error handling for non-existent target class
    error_result = spearman_tool.invoke({"target_class": "invalid_category"})
    assert "error" in error_result
