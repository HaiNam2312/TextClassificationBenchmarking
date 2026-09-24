# Text Classification Benchmarking: Text CNN vs RNN vs BERT/Transformer

Deep learning coursework project (Monash University, Semester 2 2025) benchmarking three model architectures — Text CNN, RNN, and a fine-tuned BERT/Transformer classifier — on a text classification task.

## Results

| Model | Test Accuracy | Notes |
|---|---|---|
| Text CNN | 96.46% | Baseline convolutional model |
| RNN | up to 100% | Minimal overfitting observed |
| BERT / Transformer (fine-tuned) | 99.875% | Train loss 0.001, train accuracy 1.0 |

## Repository Structure

```
├── notebooks/
│   ├── 01_main_textcnn.ipynb        # Text CNN baseline
│   ├── 02_rnn.ipynb                 # RNN model
│   └── 03_transformer_bert.ipynb    # Fine-tuned BERT/Transformer classifier
├── html_exports/
│   ├── 01_main_textcnn.html
│   ├── 02_rnn.html
│   └── 03_transformer_bert.html
└── README.md
```

The `html_exports/` folder contains rendered, read-only versions of each notebook (outputs included) for quick viewing without running the code.

## Tech Stack

- Python
- PyTorch / TensorFlow 
- Hugging Face Transformers (BERT fine-tuning)
- Jupyter Notebook (developed on Google Colab)

## Overview

Each notebook covers one architecture end-to-end: data preprocessing, model definition, training, and evaluation. The three models are compared on the same task and dataset to benchmark performance across a classical CNN baseline, a recurrent architecture, and a fine-tuned transformer.

---
**Author:** Hoang Hai Nam Dinh — [GitHub](https://github.com/HaiNam2312) · [LinkedIn](linkedin.com/in/namdinh231204/)
