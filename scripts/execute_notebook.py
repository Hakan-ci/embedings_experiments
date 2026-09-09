"""Execute the notebook in a fresh local kernel and retain verifiable outputs."""
from pathlib import Path
import base64
import os
import re
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("IPYTHONDIR", str(CACHE / "ipython"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(CACHE / "jupyter-runtime"))
for directory in ("MPLCONFIGDIR", "IPYTHONDIR", "JUPYTER_RUNTIME_DIR"):
    Path(os.environ[directory]).mkdir(parents=True, exist_ok=True)

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

path = ROOT / "glove_from_scratch.ipynb"
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)
headings = [int(match.group(1)) for cell in notebook.cells if cell.cell_type == "markdown"
            for match in [re.match(r"## (\d+)\.", cell.source)] if match]
assert headings == list(range(1, 27)), headings
for index, cell in enumerate(notebook.cells):
    if cell.cell_type == "code":
        compile(cell.source, f"cell-{index}", "exec")
    if cell.cell_type == "markdown" and cell.source.startswith("**Knowledge check"):
        assert notebook.cells[index + 1].source.startswith("**Answer**")

manager = KernelManager(kernel_name="python3")
manager.kernel_spec.argv[0] = sys.executable
client = NotebookClient(notebook, km=manager, timeout=600,
                        startup_timeout=30,
                        resources={"metadata": {"path": str(ROOT)}}, allow_errors=False)
started = perf_counter()
last_section = [0]

def progress(cell, cell_index, **kwargs):
    match = re.match(r"## (\d+)\.", cell.source)
    if match:
        last_section[0] = int(match.group(1))
        print(f"Section {last_section[0]}: {cell.source.splitlines()[0]}", flush=True)

client.on_cell_start = progress
print("Starting a fresh kernel with " + sys.executable, flush=True)
try:
    client.execute()
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
nbformat.validate(notebook)
assert all(cell.execution_count is not None for cell in notebook.cells if cell.cell_type == "code")
nbformat.write(notebook, path)
artifacts = ROOT / "artifacts"
artifacts.mkdir(exist_ok=True)
plot_count = 0
for cell in notebook.cells:
    for output in cell.get("outputs", []):
        assert output.output_type != "error"
        data = output.get("data", {})
        if "image/png" in data:
            plot_count += 1
            (artifacts / f"plot-{plot_count:02d}.png").write_bytes(base64.b64decode(data["image/png"]))
print(f"Executed {len(notebook.cells)} cells in {perf_counter() - started:.1f}s; saved {plot_count} plots.")
