# GloVe from first principles

Open [glove_from_scratch.ipynb](glove_from_scratch.ipynb) for the complete 26-section
tutorial. Saved outputs include worked calculations, labeled matrices, training
curves, neighbors, analogies, and controlled experiments. The implementation lives
inside the notebook and uses NumPy plus explicit PyTorch parameters and Adam.

## Setup and run

Use Python 3.12 or newer (verified with Python 3.14). From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m jupyterlab glove_from_scratch.ipynb
```

The CPU wheel avoids unnecessary CUDA packages on Linux. For other platforms,
follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/)
if that wheel is unavailable. Dependencies need downloading once; the notebook
itself makes no network requests and needs no external dataset.

Choose the Python kernel from `.venv`, then **Restart Kernel and Run All Cells**.
Edit the single configuration cell in Section 2 to change the corpus or training
settings. Defaults use 30 sentences, two-dimensional embeddings, 1,000 Adam updates,
and three seeds. Start by running the defaults, then change one factor at a time.
The optional pretrained loader in Section 22 runs on a small local fixture unless
you supply a real text-format GloVe path.

## Reproduce the saved notebook

```bash
.venv/bin/python scripts/execute_notebook.py
```

This starts a fresh kernel using that interpreter, executes every cell in order,
checks notebook structure, saves outputs, and extracts plot PNGs to `artifacts/`
for visual inspection. Jupyter requires local kernel socket communication.
The notebook's executable checks verify hand-counted windows, filtering distances,
positive-count masking, NumPy/PyTorch loss agreement, a finite-difference gradient,
retrieval edge cases, the file loader, finite training, and seed reproducibility.
The tests do not require analogies to succeed.

`scripts/build_notebook.py` is the optional authoring source. Rebuilding with it
**clears saved outputs**; run `scripts/execute_notebook.py` afterwards. To keep edits
made directly in Jupyter, do not rebuild from that source unless you also port the
edits to it.

## Reading the experiments

All requested dimension, window, distance-weighting, alpha, and threshold settings
are repeated across seeds 7, 17, and 27. Identical baselines are cached. Losses are
comparable across dimensions on the same matrix; changing counts or weights changes
the objective. Tiny-corpus neighbors and analogy ranks are descriptive observations,
not evidence of general language understanding. No seed or result is selected to
make an analogy succeed.

Primary references: [GloVe paper](https://aclanthology.org/D14-1162/),
[official code and vectors](https://nlp.stanford.edu/projects/glove/),
[SGNS paper](https://arxiv.org/abs/1310.4546).
