# Word embeddings from first principles

## Skip-Gram / Word2Vec

Open [skip_gram_exploration.ipynb](skip_gram_exploration.ipynb) for a 20-section
teaching notebook with saved outputs. It derives full-softmax Skip-Gram and
negative sampling, implements both with NumPy, checks their gradients, and explores
context windows, embedding dimensions, cosine similarity, and PCA.

The core needs only NumPy and Matplotlib, plus Jupyter to run the notebook.
It does not require the GloVe tutorial's PyTorch dependencies. On Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install numpy matplotlib nbformat nbclient ipykernel jupyterlab
.\.venv\Scripts\python -m jupyterlab skip_gram_exploration.ipynb
```

On macOS/Linux use `.venv/bin/python` in place of `.\.venv\Scripts\python`.
Select that environment's kernel, then **Restart Kernel and Run All Cells**.
Defaults use nine sentences, seed 42, window 2, dimension 5, learning rate 0.05,
300 epochs, and five negative samples. All data are embedded; no cell downloads
data or installs packages. Gensim is an optional, disabled comparison in section 18.

To reproduce the saved outputs in a fresh kernel:

```powershell
.\.venv\Scripts\python -X utf8 scripts/execute_notebook.py skip_gram_exploration.ipynb
```

The runner validates all 20 sections, executes every code cell, and exports plots
to `artifacts/skip_gram_exploration/`. In-notebook checks cover hand-counted pairs,
sentence boundaries, stable losses, both manual gradients (including repeated
negative samples), finite training, and seed reproducibility. Semantic neighbors
are observations, not pass/fail requirements. Edit this notebook directly;
`scripts/build_notebook.py` authors only the GloVe notebook.

## GloVe from first principles

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
checks notebook structure, saves outputs, and extracts plot PNGs to `artifacts/glove_from_scratch/`
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
