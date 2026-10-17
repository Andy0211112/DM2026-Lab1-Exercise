import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_check_missing_tool_known_answer_2():
    """Test check_missing_tool against known answer #2 in TEST_FIXTURE.md."""
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load sample_fixture.csv dataset into session
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Instantiate student tools
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Run tool
    res = check_missing_tool.invoke({})

    # Known answer #2 assertions: 1 row missing text out of 8 (row index 7)
    assert res.get("missing_count") == 1
    assert res.get("total_rows") == 8
    assert res.get("missing_indices") == [7]


def test_check_missing_tool_no_dataset_loaded():
    """Test check_missing_tool when session has no dataframe loaded."""
    session = SessionState()
    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    # Invoke tool on empty session state
    res = check_missing_tool.invoke({})

    # Verify that an error response was returned rather than crashing with an exception
    assert "error" in res or res.get("status") == "error"


def test_check_duplicates_tool_known_answer_3_report_only():
    """Test check_duplicates_tool with drop=False against known answer #3."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": False})

    # Known answer #3 assertions (report only)
    assert res.get("duplicate_count") == 1
    assert res.get("rows_to_drop") == [5, 6]
    assert res.get("dropped") is False


def test_check_duplicates_tool_known_answer_3_drop():
    """Test check_duplicates_tool with drop=True against known answer #3."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    res = check_duplicates_tool.invoke({"drop": True})

    # Known answer #3 assertions (dropping)
    assert res.get("duplicate_count") == 1
    assert res.get("rows_to_drop") == [5, 6]
    assert res.get("dropped") is True
    assert res.get("rows_before") == 8
    assert res.get("rows_after") == 6

    # Verify that session.dataframe index was reset to contiguous 0..5
    assert list(session.dataframe.index) == list(range(6))


def test_sample_data_tool_known_answer_13():
    """Test sample_data_tool with n=4, random_state=42 against known answer #13."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    # Known answer #13 assertions with corrected key `n_sampled`
    assert res.get("n_sampled") == 4
    assert res.get("sampled_indices") == [1, 5, 0, 7]

    # Ensure original dataframe remains untouched
    assert len(session.dataframe) == 8

    # Ensure artifact index was reset to 0..3
    assert "sample" in session.artifacts
    assert list(session.artifacts["sample"].index) == list(range(4))


def test_sample_data_tool_n_exceeds_rows():
    """Test sample_data_tool returns an error when n exceeds row count."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    sample_data_tool = next(t for t in tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 100, "random_state": 42})

    assert "error" in res or res.get("status") == "error"


def test_list_files_tool_known_answer_11():
    """Test list_files_tool against known answer #11 and error handling."""
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    # Valid subdirectory call
    res_valid = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res_valid.get("directories") == []
    assert res_valid.get("files") == ["Reddit-stock-sentiment.csv"]

    # Invalid path traversal directory call
    res_invalid = list_files_tool.invoke({"subdirectory": "../.."})
    assert res_invalid.get("status") == "error" or "error" in res_invalid


def test_tokenize_tool_known_answer_10():
    """Test tokenize_tool against known answer #10 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    tokenize_tool = next(t for t in tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})

    assert res.get("n_documents") == 8
    assert res.get("total_tokens") == 27

    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    assert list(session.dataframe["unigrams"]) == expected_unigrams


def test_describe_data_tool_known_answer_18():
    """Test describe_data_tool against known answer #18 with pytest.approx."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    describe_data_tool = next(t for t in tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({})

    # Check overall stats
    overall = res.get("overall_stats", {})
    assert overall.get("mean") == pytest.approx(19.875, abs=1e-3)
    assert overall.get("std") == pytest.approx(9.4330, abs=1e-3)

    # Check per category stats
    per_cat = res.get("per_category_stats", {})
    assert per_cat.get("catA", {}).get("mean") == pytest.approx(17.25, abs=1e-3)
    assert per_cat.get("catB", {}).get("std") == pytest.approx(5.7446, abs=1e-3)

    # Check pending figure and temporary column cleanup
    assert session.pending_figure is not None
    assert "text_length" not in session.dataframe.columns


def test_inspect_data_tool_custom():
    """Test inspect_data_tool functionality and error handling."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    # Valid call with n=3
    res = inspect_data_tool.invoke({"n": 3})
    assert res.get("n_inspected") == 3
    assert res.get("total_rows") == 8

    rows = res.get("rows", [])
    assert len(rows) == 3
    assert rows[0].get("index") == 0
    assert rows[0].get("text_length") == 29

    # Edge cases / boundary validation
    res_zero = inspect_data_tool.invoke({"n": 0})
    assert res_zero.get("status") == "error" or "error" in res_zero

    res_exceed = inspect_data_tool.invoke({"n": 21})
    assert res_exceed.get("status") == "error" or "error" in res_exceed
