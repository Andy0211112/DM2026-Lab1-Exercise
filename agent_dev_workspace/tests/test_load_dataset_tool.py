import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool_fixture():
    # Single session instance for loading premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Known Answer #1 in TEST_FIXTURE.md
    result = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Check return dictionary fields against known answer #1
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}

    # Verify session state dataframe is populated
    assert session.dataframe is not None
    assert len(session.dataframe) == 8
