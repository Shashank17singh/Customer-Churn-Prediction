import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebooks" / "customer_churn.ipynb"


def code_cells():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_encoded_columns_are_assigned_back():
    """The original notebook called .replace() without assigning, which collapsed every label to 0."""
    cells = "\n".join(code_cells())
    assert "df1[col] = df1[col].replace" in cells
    assert "df1['gender'] = df1['gender'].replace" in cells


def test_notebook_ships_without_stale_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert all(not c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
