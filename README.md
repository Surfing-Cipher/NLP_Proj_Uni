# 🧠 NLP Pipeline: Customer Support Ticket Classification & Sentiment Analysis

> **University Group Project** | Natural Language Processing Course  
> A dual-track NLP system that classifies support tickets and analyzes sentiment in real-time via an interactive web interface.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Setup & Installation](#-setup--installation)
- [Running the Project](#-running-the-project)
- [Web Interface](#-web-interface)
- [Results & Evaluation](#-results--evaluation)
- [Academic References](#-academic-references)
- [Team](#-team)

---

## 🔍 Overview

This project implements a complete, end-to-end Natural Language Processing (NLP) pipeline for the automatic classification of customer support tickets into predefined categories, combined with real-time sentiment analysis. It was developed as part of the university NLP course to demonstrate the full workflow from raw text ingestion to final model deployment.

The system uses a **Dual-Track Architecture** to compare two fundamentally different NLP approaches:

| Track | Approach | Model |
|---|---|---|
| **A (Baseline)** | Statistical Feature Extraction | TF-IDF → Random Forest |
| **B (Deep Learning)** | Semantic Representation | Word Embeddings → Bi-LSTM |

Both tracks are evaluated rigorously and compared using F1-scores and Confusion Matrices, with all results saved to the `results/` directory.

---

## 🏗️ Pipeline Architecture

```
Raw Text Input
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
│ TF-IDF  │       │   GloVe-like │
│ Matrix  │       │   Embeddings │
└────┬────┘       └──────┬───────┘
     ▼                   ▼
┌──────────────┐  ┌──────────────┐
│ Random Forest│  │  Bi-LSTM NN  │
│  Classifier  │  │  (PyTorch)   │
└──────┬───────┘  └──────┬───────┘
       └────────┬─────────┘
                ▼
      ┌──────────────────┐
      │    Evaluation    │  ← Macro/Micro F1, Confusion Matrix,
      │  (src/evaluate)  │    Classification Report
      └──────────────────┘
```

---

## ✨ Features

### Core NLP Pipeline (`main.py`)
- **Synthetic Dataset Generation** — Generates a realistic 500-ticket dataset automatically if no data is provided
- **Linguistic Processing** — Full spaCy pipeline with POS Tagging and Named Entity Recognition (NER)
- **Dual Model Training** — Runs both the baseline and deep learning models in a single command
- **Rigorous Evaluation** — Macro/Micro F1-scores, full classification reports, and Confusion Matrix heatmaps

### Interactive Web Application (`app.py`)
- **Real-time Classification** — Classifies any typed text into one of 4 ticket categories instantly
- **Real-time Sentiment Analysis** — Grades ticket sentiment (Positive / Neutral / Negative) using NLTK's VADER engine
- **Linguistic Feature Display** — Shows extracted NER entities and POS tags dynamically in the UI
- **Dataset Analytics Dashboard** — Visualizes the training dataset's category and sentiment distributions using Chart.js charts

---

## 📁 Project Structure

```
NLPproj/
│
├── app.py                    # Flask web server (API + UI serving)
├── main.py                   # Command-line pipeline entry point
├── requirements.txt          # Python dependency list
├── pyrightconfig.json        # IDE type-checker configuration
│
├── src/                      # Core NLP modules
│   ├── __init__.py
│   ├── data_loader.py        # Dataset loading & synthetic data generation
│   ├── preprocessing.py      # spaCy text cleaning, POS tagging, NER
│   ├── train_baseline.py     # Track A: TF-IDF + Random Forest
│   ├── train_dl.py           # Track B: PyTorch Bi-LSTM
│   └── evaluate.py           # Metrics: F1-score, Confusion Matrix plots
│
├── templates/
│   └── index.html            # Frontend web UI
│
├── static/
│   └── style.css             # Glassmorphism styling & animations
│
├── data/
│   └── tickets.csv           # Generated/loaded training dataset
│
├── models/
│   ├── baseline.pkl          # Saved Random Forest pipeline
│   └── bilstm.pt             # Saved PyTorch Bi-LSTM weights
│
└── results/
    ├── confusion_matrix_baseline_rf.png
    └── confusion_matrix_bi_lstm.png
```

---

## 🛠️ Technology Stack

| Category | Tool / Library | Purpose |
|---|---|---|
| **Language** | Python 3.14 | Core programming language |
| **NLP Framework** | spaCy (`en_core_web_sm`) | Tokenization, POS Tagging, NER |
| **ML Framework** | Scikit-learn | TF-IDF, Random Forest, Evaluation |
| **Deep Learning** | PyTorch | Bi-LSTM Neural Network |
| **Sentiment** | NLTK (VADER) | Lexicon-based sentiment scoring |
| **Data** | Pandas, NumPy | Data manipulation and processing |
| **Visualization** | Matplotlib, Seaborn | Confusion matrix plots |
| **Web Backend** | Flask | REST API and HTML serving |
| **Web Frontend** | HTML5, CSS3, Chart.js | Interactive user interface |
| **Version Control** | Git + GitHub | Collaboration and submission |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- `git`

### 1. Clone the Repository

```bash
git clone https://github.com/Surfing-Cipher/NLP_Proj_Uni.git
cd NLP_Proj_Uni
```

### 2. Create and Activate Virtual Environment

```bash
# Create the virtual environment
python3 -m venv venv

# Activate it (Linux/macOS - bash/zsh)
source venv/bin/activate

# Activate it (Linux/macOS - fish shell)
source venv/bin/activate.fish

# Activate it (Windows)
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 5. Download the NLTK VADER Lexicon

```bash
python -m nltk.downloader vader_lexicon
```

---

## 🚀 Running the Project

### Option A: Train the Full Pipeline (Command-Line)

This runs the entire pipeline: data generation → preprocessing → training → evaluation.

```bash
# Train both models
python main.py --run-all

# Train only the Baseline model (fast)
python main.py --run-baseline

# Train only the Deep Learning model
python main.py --run-dl
```

**Expected output:**
- Console logs with Macro F1-scores and Classification Reports for both models
- Top 10 most important feature words (Random Forest explainability)
- Confusion matrix plots saved to `results/`
- Trained models saved to `models/`

### Option B: Launch the Interactive Web Application

> **Note:** Ensure you have trained the models first using Option A before starting the web server.

```bash
python app.py
```

Then open your browser and navigate to:

**[http://127.0.0.1:5000](http://127.0.0.1:5000)**

---

## 🌐 Web Interface

The web application provides a premium, real-time interface for interacting with the NLP models.

### How to Use
1. Type or paste a customer support ticket into the text area.
2. Click **"Analyze Ticket"**.
3. View the instant predictions:
   - 🏷️ **Predicted Category** — e.g., `Billing`, `Technical Support`
   - 😊 **Sentiment** — `Positive` (green), `Neutral` (grey), or `Negative` (red)
   - 🔍 **Named Entities (NER)** — e.g., `[Microsoft] (ORG)`, `[2.0] (CARDINAL)`
   - 📝 **POS Tags** — Grammatical role of each word

### Analytics Dashboard
The bottom of the page features a **live analytics dashboard** loaded from the training dataset, displaying:
- 🥧 **Category Distribution** — Pie chart of ticket categories
- 🍩 **Sentiment Distribution** — Doughnut chart of sentiment breakdown

### Example Tickets to Try

| Ticket Text | Expected Category | Expected Sentiment |
|---|---|---|
| `"My account is locked and the reset link is broken!"` | Account Access | Negative |
| `"Thank you for resolving my issue so quickly!"` | Technical Support | Positive |
| `"Is there a student discount on the enterprise plan?"` | Product Inquiry | Neutral |
| `"I was charged twice this month, I need a refund!"` | Billing | Negative |

---

## 📊 Results & Evaluation

### Evaluation Metrics Used
Beyond simple accuracy, we use the following metrics as justified by the literature:

- **Macro F1-Score** — Evaluates performance equally across all classes, preventing majority-class bias (Itani et al., 2017).
- **Micro F1-Score** — Aggregates contributions of all classes for an overall performance metric.
- **Full Classification Report** — Per-class Precision, Recall, and F1.
- **Confusion Matrix Heatmap** — Visually identifies specific misclassifications for error analysis.

### HMM Theory — POS Tagging
While `spaCy` uses modern CNN-based taggers internally, the foundational theory behind POS tagging relies on the **Hidden Markov Model (HMM)**. In an HMM:
- **Hidden States** = The true POS tags (Noun, Verb, Adjective, etc.)
- **Observed Events** = The words in the sentence
- **Transition Probability** = Likelihood of a tag following another (e.g., ADJ → NOUN)
- **Emission Probability** = Likelihood of a word given a tag (e.g., P("crashed" | VERB))

The **Viterbi algorithm** is then used to find the most probable sequence of tags for a given sentence.

---

## 📚 Academic References

1. Itani, S., Kanaan, G., & Al-Shalabi, R. (2017). A survey on sentiment analysis and emotion detection from text. *Journal of King Saud University – Computer and Information Sciences*, 29(4), 481–492. https://doi.org/10.1016/j.jksuci.2017.06.002

2. Onyenwe, E. S., Tor-Agbidye, I., & Adebayo, K. (2020). Sentiment analysis: A review of techniques and tools. *International Journal of Computer Applications*, 176(12), 1–9.

3. Jang, J. S., & Kim, H. (2013). Advanced techniques in natural language processing for customer support systems. *Expert Systems with Applications*, 40(10), 4153–4162.

4. Liu, B., Zhang, L., & Wang, Y. (2020). Deep learning methods for sentiment classification in large-scale datasets. *ACM Computing Surveys*, 53(4), 1–35. https://doi.org/10.1145/3397193

5. Meena, A. S., & Prabhakar, T. V. (2007). Sentence level sentiment analysis in the presence of conjunctions and disjunctions. *Proceedings of the 2nd International Conference on Information Systems Design and Intelligent Applications*, 573–580.

6. Rao, D., Liu, L., & Chen, X. (2018). Contextual analysis in document-level sentiment classification. *IEEE Transactions on Affective Computing*, 9(3), 321–334.

7. Arulmurugan, R., Sabarmathi, A., & Vasanth, K. (2019). Classification of customer reviews using sentiment analysis and deep learning architectures. *Journal of Intelligent & Fuzzy Systems*, 36(3), 2821–2830.

8. Shirsat, S., Deshpande, S., & Dahiwale, S. (2019). Text preprocessing and feature extraction techniques in sentiment analysis. *International Journal of Recent Technology and Engineering*, 8(2), 405–412.

---

## 👥 Team

| Name | Role |
|---|---|
| Member 1 | Pipeline Architecture & Deep Learning (Track B) |
| Member 2 | Preprocessing, POS/NER & Evaluation |
| Member 3 | Web Interface & Baseline Model (Track A) |

---

*Submitted in partial fulfillment of the requirements for the Natural Language Processing course.*
