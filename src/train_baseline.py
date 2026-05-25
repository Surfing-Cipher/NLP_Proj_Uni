import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

from evaluate import evaluate_model

def train_and_evaluate_baseline(df):
    """
    Trains the Track A Baseline Model (TF-IDF + Random Forest).
    """
    print("Preparing Track A Baseline...")
    
    # We'll use the target categories
    X = df['cleaned_text']
    y = df['category']
    
    # For a real project we could encode categories, here we let sklearn handle strings
    classes = sorted(y.unique())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the Scikit-Learn pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    print("Training Baseline Model (Random Forest)...")
    pipeline.fit(X_train, y_train)
    
    print("Predicting on test set...")
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    evaluate_model(y_test, y_pred, classes, model_name="Baseline RF")
    
    # Feature Importance (Explainability as per constraint management)
    print("\n--- Model Explainability ---")
    tfidf = pipeline.named_steps['tfidf']
    clf = pipeline.named_steps['clf']
    feature_names = tfidf.get_feature_names_out()
    importances = clf.feature_importances_
    
    # Map importances to words
    feature_imp_df = pd.DataFrame({'word': feature_names, 'importance': importances})
    top_features = feature_imp_df.sort_values(by='importance', ascending=False).head(10)
    print("Top 10 Most Important Words/N-grams:")
    print(top_features.to_string(index=False))
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(pipeline, 'models/baseline.pkl')
    print("Baseline model saved to models/baseline.pkl")

if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_dataset
    
    df = load_data()
    df = preprocess_dataset(df)
    train_and_evaluate_baseline(df)
