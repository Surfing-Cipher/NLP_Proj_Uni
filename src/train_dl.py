import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import os

from evaluate import evaluate_model

# 1. Define Dataset
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=50):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        text = self.texts.iloc[index]
        tokens = text.split()
        
        # Tokenize and Pad
        token_ids = [self.vocab.get(t, self.vocab['<UNK>']) for t in tokens][:self.max_len]
        padding = [self.vocab['<PAD>']] * (self.max_len - len(token_ids))
        token_ids.extend(padding)
        
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(self.labels.iloc[index], dtype=torch.long)

# 2. Build Vocabulary
def build_vocab(texts, max_size=5000):
    all_words = " ".join(texts).split()
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(max_size)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for idx, (word, _) in enumerate(common_words):
        vocab[word] = idx + 2
    return vocab

# 3. Define Bi-LSTM Architecture
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)
        # Bi-LSTM Layer as requested in methodology
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # The output size is hidden_dim * 2 because it's bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Concat the final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden_cat = self.dropout(hidden_cat)
        out = self.fc(hidden_cat)
        return out

def train_and_evaluate_dl(df):
    """
    Trains Track B: PyTorch Bi-LSTM Model.
    """
    print("Preparing Track B: PyTorch Bi-LSTM...")
    
    # Preprocess targets to numeric
    classes = sorted(df['category'].unique())
    label_map = {label: idx for idx, label in enumerate(classes)}
    df['encoded_cat'] = df['category'].map(label_map)
    
    X = df['cleaned_text']
    y = df['encoded_cat']
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)
    
    # Vocab and Datasets
    vocab = build_vocab(X_train)
    vocab_size = len(vocab)
    
    train_dataset = TextDataset(X_train, y_train, vocab)
    val_dataset = TextDataset(X_val, y_val, vocab)
    test_dataset = TextDataset(X_test, y_test, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Model Setup
    model = BiLSTMClassifier(vocab_size=vocab_size, embed_dim=100, hidden_dim=64, num_classes=len(classes))
    criterion = nn.CrossEntropyLoss()
    # L2 Regularization
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training Loop with Early Stopping
    epochs = 15
    patience = 2
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs('models', exist_ok=True)
    
    print("Training Deep Learning Model with Early Stopping...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation Phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item()
                
        avg_train_loss = total_loss/len(train_loader)
        avg_val_loss = val_loss/len(val_loader)
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'models/bilstm.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break
                
    # Load best model for testing
    model.load_state_dict(torch.load('models/bilstm.pt'))
    
    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            predictions = model(batch_x)
            preds = torch.argmax(predictions, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    evaluate_model(all_targets, all_preds, classes, model_name="Bi LSTM")
    
    # Save Model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/bilstm.pt')
    print("PyTorch model saved to models/bilstm.pt")

if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import preprocess_dataset
    
    df = load_data()
    df = preprocess_dataset(df)
    train_and_evaluate_dl(df)
