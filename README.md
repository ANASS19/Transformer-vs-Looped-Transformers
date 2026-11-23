# Transformer vs. Looped Transformer

<div align="center">

**A reproducible experiment comparing standard and weight-shared looped Transformers on Tiny Shakespeare.**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1%20CUDA%2012.8-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)](https://jupyter.org/)

</div>

## Overview

This repository contains three interactive Jupyter notebooks for studying recurrent-depth weight sharing in a small character-level language model.

| Notebook | Purpose |
|---|---|
| `standard_transformer_experiment.ipynb` | Train and evaluate a standard Transformer |
| `looped_transformer_experiment.ipynb` | Train and evaluate a looped Transformer |
| `standard_vs_looped_transformer_experiment.ipynb` | Run both architectures and compare them directly |

## Experiment

Both architectures are trained on the Tiny Shakespeare character-level dataset.

- **Standard Transformer:** four independent Transformer blocks
- **Looped Transformer:** one Transformer block reused four times
- **Same:** tokenizer, data split, batch size, context length, hidden width, attention heads, optimizer, and update count
- **Important:** weight sharing reduces unique parameters, but it is not a parameter-matched experiment

The notebooks record:

- Training and validation loss
- Trainable parameter count
- Forward-pass latency
- Peak memory usage
- Training throughput
- Perplexity
- Generated text samples

## Quick Start

### 1. Clone the repository

```powershell
git clone https://github.com/ANASS19/Transformer-vs-Looped-Transformers.git
cd "Transformer-vs-Looped-Transformers"
```

### 2. Install the dependencies

```powershell
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
py -3.12 -m ipykernel install --user --name transformer-looped-lab --display-name "Transformer Looped Lab (Python 3.12)"
```

### 3. Open VS Code

```powershell
code .
```

Open any notebook and select the kernel:

```text
Transformer Looped Lab (Python 3.12)
```

Then use **Run All** to execute the notebook from top to bottom.

## Recommended Workflow

1. Run `standard_transformer_experiment.ipynb`.
2. Run `looped_transformer_experiment.ipynb`.
3. Open `standard_vs_looped_transformer_experiment.ipynb` for the direct comparison.

Both independent notebooks save their metrics into the `results/` directory, which the comparison notebook can read.

## Outputs

Running the notebooks creates:

```text
results/
├── standard_history.csv
├── standard_metrics.json
├── standard_model.pt
├── looped_history.csv
├── looped_metrics.json
└── looped_model.pt
```

Model checkpoints and generated result files are machine-specific and should generally not be committed.

## Project Structure

```text
.
├── standard_transformer_experiment.ipynb
├── looped_transformer_experiment.ipynb
├── standard_vs_looped_transformer_experiment.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

## Interpretation

This project is a small, controlled comparison rather than a universal benchmark. Results depend on:

- Random seed
- Number of training updates
- Loop iterations
- Dataset split
- Batch size
- Hidden width
- Hardware
- PyTorch kernels

For stronger conclusions, repeat experiments across multiple seeds and add parameter-matched and compute-matched baselines.

## License

Add a license of your choice before publishing the repository.
