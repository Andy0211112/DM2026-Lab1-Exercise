import numpy as np
import pytest

from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_dtm():
    """Helper to initialize a fresh session, load sample_fixture.csv, and run build_dtm_tool."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    build_dtm_tool.invoke({})
    
    return session, reduce_dimensions_tool, binarize_labels_tool


def test_reduce_dimensions_pca():
    """Test PCA reduction against known answer #7 in TEST_FIXTURE.md."""
    session, reduce_dimensions_tool, _ = _setup_dtm()
    
    result = reduce_dimensions_tool.invoke({
        "method": "pca",
        "n_components": 2
    })
    
    expected_coords = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    expected_var = [0.7532, 0.1965]
    
    actual_coords = np.array(session.artifacts["reduced_coordinates"])
    
    assert np.allclose(actual_coords, expected_coords, atol=1e-3)
    assert np.allclose(result["explained_variance_ratio"], expected_var, atol=1e-3)
    assert session.pending_figure is not None


def test_reduce_dimensions_tsne():
    """Test t-SNE reduction properties (shape, duplicate mapping, determinism, perplexity_used).
    
    Note: Exact coordinates from known answer #8 are platform/sklearn-version sensitive,
    so we check structural and invariant properties rather than exact numeric alignment.
    """
    session, reduce_dimensions_tool, _ = _setup_dtm()
    
    result1 = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    coords1 = np.array(session.artifacts["reduced_coordinates"])
    
    # 1. Shape check (8 documents, 2 dimensions) and finite values
    assert coords1.shape == (8, 2)
    assert np.isfinite(coords1).all()
    
    # 2. Duplicate documents (rows 5 and 6) should land on the same point
    assert np.allclose(coords1[5], coords1[6], atol=1e-3)
    
    # 3. Determinism check: running twice with same random_state gives identical coordinates
    session2, reduce_dimensions_tool2, _ = _setup_dtm()
    reduce_dimensions_tool2.invoke({
        "method": "tsne",
        "perplexity": 2,
        "random_state": 42
    })
    coords2 = np.array(session2.artifacts["reduced_coordinates"])
    assert np.allclose(coords1, coords2, atol=1e-5)
    
    # 4. Result dict checks
    assert result1.get("perplexity_used") == 2.0
    assert session.pending_figure is not None


def test_reduce_dimensions_umap():
    """Test UMAP reduction against known answer #9 in TEST_FIXTURE.md."""
    session, reduce_dimensions_tool, _ = _setup_dtm()
    
    result = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    
    expected_coords = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])
    
    actual_coords = np.array(session.artifacts["reduced_coordinates"])
    
    assert np.allclose(actual_coords, expected_coords, atol=1e-3)
    assert session.pending_figure is not None


def test_binarize_labels():
    """Test binarize_labels_tool behavior on sample_fixture.csv labels."""
    session, _, binarize_labels_tool = _setup_dtm()
    
    result = binarize_labels_tool.invoke({})
    
    assert result["classes"] == ["catA", "catB"]
    assert result["encoded_shape"] == [8, 1]
    
    bin_category = session.dataframe["bin_category"]
    for idx in range(3):
        assert bin_category.iloc[idx] == [0]
    for idx in range(3, 7):
        assert bin_category.iloc[idx] == [1]
    assert bin_category.iloc[7] == [0]
