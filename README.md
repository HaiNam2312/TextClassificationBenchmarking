# Text Classification Benchmarking: Text CNN vs RNN vs Transformer

Deep learning coursework (FIT3181 Deep Learning, Assignment 2, Monash University, Semester 2 2025). Compares a Text CNN, RNN variants (including attention) and a Transformer classifier on question classification, plus a BERT prefix-tuning experiment.

## Dataset and setup

- **Dataset:** TREC Question Classification (`train_2000.label`), 2,000 questions, 6 coarse classes (ABBR, ENTY, DESC, HUM, LOC, NUM).
- **Split:** 80 / 10 / 10 random split, giving 1,601 train / 198 validation / 201 test questions.
- **Preprocessing:** lower-casing, `bert-base-uncased` tokenizer, padding to the longest sequence.
- **Metric:** accuracy (a single random split, one run per model).

## Results

| Model | Validation accuracy | Test accuracy | Notes |
|---|---|---|---|
| GloVe feature baseline | 90.40% | - | Pretrained `glove-wiki-gigaword-100` vectors with a classical classifier |
| Text CNN | 96.46% (final epoch) | 97.01% | 50 epochs |
| RNN (incl. attention variant) | about 97% (best 96.97%) | not evaluated | 30 epochs; compared frozen vs fine-tuned embeddings |
| Transformer classifier (best model) | 98.48% (best epoch) | **97.01%** | 12 layers, 512-dim, 8 heads; checkpoint chosen by validation accuracy |
| BERT prefix-tuning | see notebook 3 | - | Pretrained BERT with trainable prefix and classifier; trained for 100 epochs |

Training accuracy reaches about 99-100% for all models, so the models fit the training data almost perfectly. Validation and test accuracy are the numbers to compare.

## Repository structure

```
├── notebooks/
│   ├── 01_main_textcnn.ipynb        # RNN from scratch, GloVe baseline, Text CNN
│   ├── 02_rnn.ipynb                 # RNN, embedding initialisation vs fine-tuning, attention RNN
│   └── 03_transformer_bert.ipynb    # Transformer classifier, BERT prefix-tuning, best model
├── html_exports/
│   ├── 01_main_textcnn.html
│   ├── 02_rnn.html
│   └── 03_transformer_bert.html
└── README.md
```

The `html_exports/` folder contains rendered, read-only versions of each notebook (outputs included) for quick viewing without running the code.

## What is implemented

- A multi-timestep RNN written from scratch in PyTorch: manual forward pass, cross-entropy loss, backpropagation and an SGD parameter update.
- Text CNN, RNN and attention-RNN classifiers trained with the same data pipeline.
- A Transformer encoder classifier, and BERT prefix-tuning via Hugging Face Transformers.

## Limitations

- **Small dataset and test set:** 201 test questions, so one misclassified question changes accuracy by about 0.5%. Differences of 1-2 questions between models are not meaningful.
- **Single split, single run:** no repeated runs or confidence intervals.
- **Possible label leakage:** the printed question text appears to start with the TREC fine-grained label (for example `manner`, `exp`, `animal`), which may reveal the coarse class. This has not been tested by removing that token, so the absolute accuracy figures should be read with caution.
- The RNN notebook does not report a test-set result.

## Tech stack

- Python, PyTorch
- Hugging Face Transformers (`bert-base-uncased` tokenizer, BERT prefix-tuning)
- gensim (GloVe), scikit-learn
- Jupyter Notebook (developed on Google Colab)

---
**Author:** Hoang Hai Nam Dinh — [GitHub](https://github.com/HaiNam2312) · [LinkedIn](https://linkedin.com/in/namdinh231204/)
