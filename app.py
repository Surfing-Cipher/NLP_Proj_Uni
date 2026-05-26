import os
import joblib
from flask import Flask, render_template, request, jsonify
import spacy
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure we can import from src directory
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from preprocessing import extract_linguistic_features, clean_text

app = Flask(__name__)

# Load models at startup
print("Loading NLP models...")
try:
    nlp = spacy.load("en_core_web_sm")
    # Load the trained baseline model
    model_path = os.path.join('models', 'baseline.pkl')
    if os.path.exists(model_path):
        clf = joblib.load(model_path)
    else:
        print("Warning: baseline.pkl not found. Please run main.py first.")
        clf = None
except Exception as e:
    print(f"Error loading models: {e}")
    clf = None

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
        
    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text provided'}), 400
        
    # Extract linguistic features (POS, NER)
    features = extract_linguistic_features(text)
    
    # Preprocess text and predict
    category = "Unknown"
    if clf:
        cleaned = clean_text(text)
        prediction = clf.predict([cleaned])
        category = prediction[0]
        
    # Analyze Sentiment
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.05:
        sentiment = "Positive"
    elif compound <= -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"
        
    return jsonify({
        'category': category,
        'sentiment': sentiment,
        'pos_tags': features['pos_tags'],
        'ner': features['ner']
    })

@app.route('/api/stats')
def stats():
    data_path = os.path.join('data', 'tickets.csv')
    if not os.path.exists(data_path):
        return jsonify({'error': 'No dataset found'}), 404
        
    df = pd.read_csv(data_path)
    cat_counts = df['category'].value_counts().to_dict()
    sent_counts = df['sentiment'].value_counts().to_dict()
    
    return jsonify({
        'categories': cat_counts,
        'sentiments': sent_counts
    })

if __name__ == '__main__':
    app.run(debug=True)
