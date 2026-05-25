import os
import joblib
from flask import Flask, render_template, request, jsonify
import spacy

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
        
    return jsonify({
        'category': category,
        'pos_tags': features['pos_tags'],
        'ner': features['ner']
    })

if __name__ == '__main__':
    app.run(debug=True)
