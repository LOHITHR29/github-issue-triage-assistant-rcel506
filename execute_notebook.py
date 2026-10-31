"""Execute and save the project notebook with its local data files."""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).parent.resolve()
os.environ.setdefault("MPLCONFIGDIR", "/tmp/rcel_mpl")
os.environ.setdefault("IPYTHONDIR", "/tmp/rcel_ipython")
os.environ["PATH"] = str(Path(os.sys.executable).parent) + os.pathsep + os.environ["PATH"]
path = ROOT / "issue_triage_full_pipeline_lr103.ipynb"
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, timeout=180, kernel_name="python3",
                        resources={"metadata": {"path": str(ROOT)}})
client.execute()
nbformat.validate(notebook)
errors = [(index, output.get("ename")) for index, cell in enumerate(notebook.cells)
          if cell.cell_type == "code" for output in cell.get("outputs", [])
          if output.output_type == "error"]
if errors:
    raise RuntimeError(f"Notebook errors: {errors}")
nbformat.write(notebook, path)
print(f"Executed {sum(bool(cell.get('execution_count')) for cell in notebook.cells)} code cells")
