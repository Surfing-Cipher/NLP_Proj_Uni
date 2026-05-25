# NLP Pipeline: Customer Support Ticket Classification

This project implements an end-to-end NLP pipeline for classifying customer support tickets. 
It uses a Dual-Track Architecture to compare a baseline Machine Learning model (TF-IDF + Random Forest) against a Deep Learning approach (PyTorch Bi-LSTM).

## Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running the Project

To run the complete pipeline:
```bash
python main.py --run-all
```

This will:
1. Generate/Load the dataset.
2. Run linguistic preprocessing (POS & NER).
3. Train the Baseline model.
4. Train the Deep Learning model.
5. Evaluate and save the results (Confusion Matrices) in the `results/` folder.
