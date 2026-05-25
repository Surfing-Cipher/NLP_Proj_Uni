import spacy
import pandas as pd

# Load spaCy English model (ensure it's downloaded: python -m spacy download en_core_web_sm)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """
    Cleans text by removing stop words, punctuation, and lemmatizing.
    """
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extract_linguistic_features(text):
    """
    Extracts POS tags and Named Entities (NER) as per Weeks 3-4 requirements.
    Returns a dictionary with extracted features for demonstration.
    """
    doc = nlp(text)
    
    pos_tags = [(token.text, token.pos_) for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return {
        "pos_tags": pos_tags,
        "ner": entities
    }

def preprocess_dataset(df, text_col='text'):
    """
    Applies cleaning to the entire dataset.
    """
    print("Starting preprocessing...")
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # Just to demonstrate linguistic feature extraction on the first few rows
    print("\n--- Linguistic Feature Extraction (Sample) ---")
    for idx, row in df.head(3).iterrows():
        features = extract_linguistic_features(row[text_col])
        print(f"Original Text: {row[text_col]}")
        print(f"Entities (NER): {features['ner']}")
        print(f"POS Tags (first 5): {features['pos_tags'][:5]}\n")
        
    print("Preprocessing completed.")
    return df

if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    df_clean = preprocess_dataset(df)
    print(df_clean[['text', 'cleaned_text']].head())
