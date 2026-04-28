# Character-Level Language Modelling: N-gram & Neural Architectures

Character-level language models built from scratch — Unigram, Bigram & Trigram with Laplace/interpolation smoothing, a PyTorch FNN-LM (Bengio et al., 2003) with character embeddings, and an RNN-LM with tied weights. Includes evaluation via perplexity, KL divergence & prefix-conditioned name generation.

---

## Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## Overview

This project explores **character-level language modelling** on a dataset of city names. Starting from classical statistical approaches and progressing to neural architectures, it compares 5 models on their ability to learn character distributions, compute sequence probabilities, and generate novel names.

Key tasks covered:
- Estimating character-level probability distributions
- Computing sequence log-probabilities and perplexity
- Prefix-conditioned name generation with temperature scaling
- Benchmarking generative vs. discriminative language model architectures

---

## Project Structure

```
character-level-language-modelling/
│
├── notebook.ipynb          # Main implementation notebook
├── requirements.txt        # Dependencies
├── fnn/
│   ├── model.pt            # Saved FNN-LM weights
│   ├── vocab.pt            # Vocabulary
│   └── loss.json           # Training/validation loss history
├── rnn/
│   ├── model.pt            # Saved RNN-LM weights
│   ├── vocab.pt            # Vocabulary
│   └── loss.json           # Training/validation loss history
└── README.md
```

---

## Models Implemented

### 1. Unigram Language Model
- Estimates character probabilities independent of context
- Implements **Add-1 (Laplace) smoothing** to handle unseen characters
- Computes **position-specific probability distributions** (e.g., how likely is each character at position 1, 2, etc.)
- Evaluates **KL divergence** between positional distributions

### 2. Bigram Language Model
- Conditions next character on the previous one: `P(cₜ | cₜ₋₁)`
- Implements both **Laplace smoothing** and **interpolation smoothing** variants
- Handles OOV tokens via UNK token substitution

### 3. Trigram Language Model
- Conditions next character on the previous two: `P(cₜ | cₜ₋₂, cₜ₋₁)`
- Log-space computation to prevent numerical underflow
- Laplace smoothing for zero-count n-grams

### 4. Feedforward Neural Language Model (FNN-LM)
- Based on [Bengio et al., 2003 — Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- Architecture: Character **embeddings** (dim=64) → flatten `(n-1)` context → **hidden layer** (dim=128) → softmax over vocabulary
- Includes a **shortcut (direct) connection** from input embeddings to output layer
- Trained with **cross-entropy loss** and **Adam** optimiser

### 5. RNN Language Model (RNN-LM)
- Processes full character sequence using a **Recurrent Neural Network**
- **Tied embedding and pre-softmax weights** to reduce parameters and improve generalisation
- Trained with **Adam** optimiser (lr=0.001)
- Supports **prefix-conditioned generation** by seeding the hidden state

---

## Results

| Model | Validation Perplexity |
|---|---|
| Unigram (no smoothing) | ∞ |
| Smoothed Unigram (Laplace) | 15.93 |
| Bigram (Laplace) | 20.34 |
| Bigram (Interpolation) | 12.13 |
| Trigram (Laplace) | **9.51** ✅ |
| FNN-LM (4-gram context) | 18.31 |
| RNN-LM | 50.39 |
> Fill in your actual perplexity values after running the notebook.

### Sample Generated Names

**No prefix:**
```
Ravena, Solmith, Keldar, Briven, Maston
```

**Prefix `<s>Ma`:**
```
Macon, Madrid, Malton, Manvik, Marsen
```

> These are illustrative samples — actual outputs vary per run due to temperature-scaled sampling.

---

## Installation

```bash
git clone https://github.com/your-username/character-level-language-modelling.git
cd character-level-language-modelling
pip install -r requirements.txt
```

**requirements.txt**
```
torch==2.3.0
torchtext==0.18
pandas
numpy
matplotlib
tqdm
```

> Python 3.9–3.11 is recommended.

---

## Usage

Open and run the notebook end-to-end:

```bash
jupyter notebook notebook.ipynb
```

Or on **Google Colab**, simply upload the notebook and run all cells.

Key hyperparameters you can tune:

| Parameter        | Default | Description                        |
|------------------|---------|------------------------------------|
| `EMB_SIZE`       | 64      | Character embedding dimension      |
| `HID_SIZE`       | 128     | Hidden layer size (FNN)            |
| `N_GRAM_LENGTH`  | 3       | Context window for FNN-LM          |
| `RNN_HID_SIZE`   | 256     | Hidden state size for RNN-LM       |
| `RNN_STEP_SIZE`  | 0.001   | Learning rate for RNN Adam         |
| `MAX_NAME_LENGTH`| 15      | Max characters in generated name   |

---

## Tech Stack

- **Language:** Python 3.10
- **Deep Learning:** PyTorch 2.3, torchtext 0.18
- **Data Processing:** Pandas, NumPy
- **Visualisation:** Matplotlib
- **Training Utilities:** tqdm

---

## References

- Bengio, Y., et al. (2003). [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf). *JMLR*.
- Jurafsky, D. & Martin, J.H. [Speech and Language Processing, Ch. 3 — N-gram Language Models](https://web.stanford.edu/~jurafsky/slp3/3.pdf).
- Holtzman, A., et al. (2019). [The Curious Case of Neural Text Degeneration](https://arxiv.org/pdf/1904.09751.pdf). *ICLR 2020*.
