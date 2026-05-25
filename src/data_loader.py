import pandas as pd
import numpy as np
import os
import random

def generate_synthetic_data(num_samples=500):
    """
    Generates a synthetic dataset of customer support tickets
    for testing the NLP pipeline.
    """
    categories = ['Billing', 'Technical Support', 'Account Access', 'Product Inquiry']
    sentiments = ['Positive', 'Neutral', 'Negative']
    
    # Templates for synthetic tickets
    templates = {
        'Billing': [
            "I need help with my invoice. The amount seems incorrect.",
            "Why was I charged twice for my subscription?",
            "Thank you for refunding my overcharge so quickly!",
            "I would like to update my credit card information.",
            "Can you explain the charges on my latest bill? It's confusing."
        ],
        'Technical Support': [
            "My software keeps crashing when I try to open a new file.",
            "The app is so fast now after the update, amazing work!",
            "I am getting an error code 500 when I log in.",
            "How do I configure the server settings?",
            "Your website is down and I cannot access my dashboard!"
        ],
        'Account Access': [
            "I forgot my password and the reset link is not working.",
            "How can I change my email address?",
            "My account is locked, please help me regain access.",
            "I successfully updated my profile, thanks.",
            "Someone is trying to log into my account from another country!"
        ],
        'Product Inquiry': [
            "Does the enterprise plan include 24/7 support?",
            "I love the new features in version 2.0!",
            "When will the new hardware be available for purchase?",
            "Is there a discount for students?",
            "The product description is very misleading, I am disappointed."
        ]
    }
    
    # Base sentiment mapping for templates (just for synthetic realism)
    sentiment_map = {
        "I need help with my invoice. The amount seems incorrect.": "Negative",
        "Why was I charged twice for my subscription?": "Negative",
        "Thank you for refunding my overcharge so quickly!": "Positive",
        "I would like to update my credit card information.": "Neutral",
        "Can you explain the charges on my latest bill? It's confusing.": "Negative",
        "My software keeps crashing when I try to open a new file.": "Negative",
        "The app is so fast now after the update, amazing work!": "Positive",
        "I am getting an error code 500 when I log in.": "Negative",
        "How do I configure the server settings?": "Neutral",
        "Your website is down and I cannot access my dashboard!": "Negative",
        "I forgot my password and the reset link is not working.": "Negative",
        "How can I change my email address?": "Neutral",
        "My account is locked, please help me regain access.": "Negative",
        "I successfully updated my profile, thanks.": "Positive",
        "Someone is trying to log into my account from another country!": "Negative",
        "Does the enterprise plan include 24/7 support?": "Neutral",
        "I love the new features in version 2.0!": "Positive",
        "When will the new hardware be available for purchase?": "Neutral",
        "Is there a discount for students?": "Neutral",
        "The product description is very misleading, I am disappointed.": "Negative"
    }

    data = []
    for _ in range(num_samples):
        cat = random.choice(categories)
        text = random.choice(templates[cat])
        
        # Add some random noise/entities to make NER interesting
        if random.random() > 0.7:
            entities = ["Apple", "Microsoft", "Google", "Amazon", "John Doe", "Jane Smith", "New York", "London"]
            text += f" Also, I am using {random.choice(entities)}."
            
        sentiment = sentiment_map[text.split(" Also,")[0]]
        
        data.append({
            'text': text,
            'category': cat,
            'sentiment': sentiment
        })
        
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/tickets.csv', index=False)
    print(f"Generated {num_samples} synthetic tickets at data/tickets.csv")
    return df

def load_data(filepath='data/tickets.csv'):
    """Loads the dataset. Generates it if it doesn't exist."""
    if not os.path.exists(filepath):
        print("Dataset not found. Generating synthetic dataset...")
        return generate_synthetic_data()
    return pd.read_csv(filepath)

if __name__ == "__main__":
    df = load_data()
    print(df.head())
