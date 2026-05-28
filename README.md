# 🧠 NLP Pipeline: Customer Support Ticket Classification & Sentiment Analysis

> **University Group Project** | Natural Language Processing Course  
> A dual-track NLP system that classifies real-world support tickets and analyzes sentiment in real-time via an interactive web interface.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Dataset & Annotation](#-dataset--annotation)
- [Features & Hardening](#-features--hardening)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [Web Interface](#-web-interface)
- [Results & Evaluation](#-results--evaluation)
- [Advanced Extensions & Future Work](#-advanced-extensions--future-work)
- [Academic References](#-academic-references)
- [Team](#-team)

---

## 🔍 Overview

This project implements a complete, end-to-end Natural Language Processing (NLP) pipeline for the automatic classification of customer support tickets into predefined categories, combined with real-time sentiment analysis. It was developed as part of a university NLP course to demonstrate the full workflow from raw text ingestion to final model deployment.

The system uses a **Dual-Track Architecture** to compare two fundamentally different NLP approaches:

| Track | Approach | Vectorization | Classifier |
|---|---|---|---|
| **A (Baseline)** | Statistical Feature Extraction | TF-IDF | Random Forest |
| **B (Deep Learning)** | Semantic Representation | Trainable PyTorch Embeddings | Bi-LSTM |

Both tracks are rigorously evaluated using F1-scores, Classification Reports, and Confusion Matrices on a real-world Hugging Face dataset.

---

## 🏗️ Pipeline Architecture

```text
Raw Text Input (Hugging Face Dataset or Web UI)
      │
      ▼
┌─────────────────────────────┐
│   Text Preprocessing        │  ← spaCy: Tokenization, Lemmatization,
│   (src/preprocessing.py)    │    Stop-word Removal, POS Tagging, NER
└─────────────┬───────────────┘
              │
     ┌────────┴─────────┐
     ▼                  ▼
┌─────────┐       ┌──────────────┐
│ TRACK A │       │   TRACK B    │
│ TF-IDF  │       │  PyTorch 100D│
│ Matrix  │       │  Embeddings  │
└────┬────┘       └──────┬───────┘
     ▼                   ▼
┌──────────────┐  ┌───────────────┐
│ Random Forest│  │ Bi-LSTM NN    │
│  Classifier  │  │ w/ Dropout &  │
│              │  │ Early Stopping│
└──────┬───────┘  └──────┬────────┘
       └────────┬─────────┘
                ▼
      ┌──────────────────┐
      │    Evaluation    │  ← Macro/Micro F1, Confusion Matrix,
      │  (src/evaluate)  │    Classification Report
      └──────────────────┘
```

---

## 💾 Dataset & Annotation

### Source
We use a sample (2,000 records) of the real-world **Bitext Customer Support Chatbot Training Dataset** (`bitext/Bitext-customer-support-llm-chatbot-training-dataset`), downloaded automatically via the Hugging Face `datasets` library. 

*(If Hugging Face is unavailable, the pipeline falls back to an internal synthetic template generator.)*

### Annotation Process
Because the raw Bitext dataset contains 27 complex real-world intents (e.g., `track_refund`, `recover_password`), we wrote a programmatic intent-mapping function in `data_loader.py` that automatically re-annotates and clusters these into our **4 Target Categories**:
1. `Account Access`
2. `Billing`
3. `Product Inquiry`
4. `Technical Support`

Additionally, sentiment labels (`Positive`, `Neutral`, `Negative`) are dynamically assigned to the dataset using NLTK's **VADER Sentiment Intensity Analyzer** prior to web dashboard visualization.

---

## ✨ Features & Hardening

### Core NLP Pipeline (`main.py`)
- **Linguistic Processing:** A full spaCy pipeline enforcing Tokenization, Lemmatization, Stopword Removal, POS Tagging, and Named Entity Recognition (NER).
- **Bag of Words (BoW) Foundation:** Conceptually based on BoW, our baseline utilizes the more advanced **TF-IDF Vectorizer** to appropriately scale word significance and prevent the model from biasing toward non-informative high-frequency words.

### PyTorch Hardening (`train_dl.py`)
To prevent the Bi-LSTM from overfitting on conversational data, we implemented multiple regularization techniques:
- **Dropout Layers:** `nn.Dropout(p=0.3)` randomly zeroes 30% of tensors between the embedding, LSTM, and dense layers to force distributed feature learning.
- **L2 Regularization (Weight Decay):** Applied via Adam optimizer (`weight_decay=1e-5`) to penalize explosive neural weights.
- **Early Stopping & Validation:** The dataset is split into **Train (72%), Validation (13%), and Test (15%)**. The model halts training mid-epoch if validation loss degrades for two consecutive epochs.

### Interactive Web Application (`app.py`)
- **Real-time Classification:** Classifies any typed text instantly using the saved PyTorch/Scikit-Learn models.
- **Sentiment & Linguistics UI:** Highlights NER and POS tags dynamically and renders sentiment badges.
- **Analytics Dashboard:** Uses Chart.js to render Pie and Doughnut distributions based directly on the ingested CSV.

---

## 📁 Project Structure

```text
NLPproj/
│
├── app.py                    # Flask web server (API + UI serving)
├── main.py                   # Command-line pipeline entry point
├── requirements.txt          # Python dependency list
├── pyrightconfig.json        # IDE type-checker configuration
│
├── src/                      # Core NLP modules
│   ├── __init__.py
│   ├── data_loader.py        # Hugging Face integration & dataset mapping
│   ├── preprocessing.py      # spaCy text cleaning, POS tagging, NER
│   ├── train_baseline.py     # Track A: TF-IDF + Random Forest
│   ├── train_dl.py           # Track B: PyTorch Bi-LSTM (Regularized)
│   └── evaluate.py           # Metrics: F1-score, Confusion Matrix plots
│
├── templates/
│   └── index.html            # Frontend web UI
│
├── static/
│   └── style.css             # Glassmorphism styling & animations
│
├── data/
│   └── tickets.csv           # Processed HF dataset (generated at runtime)
│
├── models/                   # Serialized ML and DL model weights
│   └── ...
└── results/                  # Saved Confusion Matrices
    └── ...
```

---

## 🛠️ Technology Stack

| Category | Tool / Library | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core programming language |
| **Data Ingestion**| `datasets` (Hugging Face) | Fetching real-world support ticket corpora |
| **NLP Framework** | spaCy (`en_core_web_sm`) | Tokenization, Lemmatization, POS Tagging, NER |
| **ML Framework** | Scikit-learn | TF-IDF, Random Forest, Metrics |
| **Deep Learning** | PyTorch (`torch`, `torch.nn`) | Bi-LSTM Neural Network, Embeddings |
| **Sentiment** | NLTK (VADER) | Lexicon-based sentiment scoring |
| **Web Backend** | Flask | REST API and HTML serving |
| **Web Frontend** | HTML5, CSS3, Chart.js | Interactive user interface |

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Surfing-Cipher/NLP_Proj_Uni.git
cd NLP_Proj_Uni
```

### 2. Create and Activate Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m nltk.downloader vader_lexicon
```

---

## 🚀 Running the Project

### Option A: Train the Full Pipeline (Command-Line)
Runs the entire pipeline: Hugging Face download $\rightarrow$ preprocessing $\rightarrow$ baseline training $\rightarrow$ DL training (with early stopping) $\rightarrow$ evaluation.

```bash
python main.py --run-all
```
*Expected Output:*
- Console logs outlining Early Stopping epoch halts, Macro F1-scores, and Top 10 Random Forest feature importances (e.g., `account`, `payment`, `order`).
- Confusion matrices saved to `results/`.
- Trained models saved to `models/`.

### Option B: Launch the Interactive Web Application
> **Note:** Ensure you have generated `data/tickets.csv` and trained the models using Option A before starting the web server.

```bash
python app.py
```
Open your browser and navigate to: **[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 📊 Results & Evaluation

By migrating from a deterministic synthetic dataset to a messy, real-world conversational dataset, our models now report highly defensible, realistic metrics:

- **Track A (Random Forest):** ~0.97 Macro F1-Score
- **Track B (Bi-LSTM):** ~0.98 Macro F1-Score

### Evaluation Metrics Used
- **Macro F1-Score:** Evaluates performance equally across all classes, preventing majority-class bias.
- **Micro F1-Score:** Aggregates contributions of all classes for a global performance metric.
- **Confusion Matrix Heatmap:** Visually maps specific class misclassifications for qualitative error analysis.

### HMM Theory — POS Tagging
While `spaCy` is utilized for implementation, the foundational theory behind Part-Of-Speech (POS) tagging relies on **Hidden Markov Models (HMM)**. In an HMM, the *hidden states* are the true POS tags (Noun, Verb, Adjective), and the *observed events* are the words. Optimal tag sequences are resolved via Transition and Emission probabilities using the **Viterbi Algorithm**.

---

## 🔬 Advanced Extensions & Future Work

While the current pipeline succeeds at classifying intent and tracking sentiment, the architecture is designed to support the following advanced downstream extensions:

1. **PCA (Dimensionality Reduction):** The TF-IDF matrix inherently creates thousands of sparse n-gram columns. Implementing Principal Component Analysis (PCA) would compress this feature space down to its most significant components, drastically reducing memory overhead for clustering algorithms.
2. **Topic Modeling (LDA):** Latent Dirichlet Allocation could be introduced to automatically discover latent support topics from unannotated support logs, providing insights before manual categorizations are decided.
3. **Extractive Summarization:** A transformer-based encoder-decoder (e.g., T5 or BART) could be attached post-classification to generate one-sentence TL;DRs of lengthy customer complaints for support agents.
4. **QA Systems & RAG:** Classified tickets could be routed to a Retrieval-Augmented Generation (RAG) system to automatically query internal documentation and draft reply emails.

---

## 📚 Academic References

1. Itani, S., Kanaan, G., & Al-Shalabi, R. (2017). A survey on sentiment analysis and emotion detection from text. *Journal of King Saud University – Computer and Information Sciences*, 29(4), 481–492.
2. Onyenwe, E. S., Tor-Agbidye, I., & Adebayo, K. (2020). Sentiment analysis: A review of techniques and tools. *International Journal of Computer Applications*, 176(12), 1–9.
3. Jang, J. S., & Kim, H. (2013). Advanced techniques in natural language processing for customer support systems. *Expert Systems with Applications*, 40(10), 4153–4162.
4. Liu, B., Zhang, L., & Wang, Y. (2020). Deep learning methods for sentiment classification in large-scale datasets. *ACM Computing Surveys*, 53(4), 1–35.

---

## 👥 Team

| Name | Role |
|---|---|
| Member 1 | Pipeline Architecture & Deep Learning (Track B) |
| Member 2 | Preprocessing, POS/NER & Evaluation |
| Member 3 | Web Interface & Baseline Model (Track A) |

---

*Submitted in partial fulfillment of the requirements for the Natural Language Processing course.*
