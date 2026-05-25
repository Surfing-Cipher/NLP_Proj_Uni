import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os

def plot_confusion_matrix(y_true, y_pred, classes, model_name="Model"):
    """
    Plots and saves a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix: {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs('results', exist_ok=True)
    filepath = f'results/confusion_matrix_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(filepath)
    plt.close()
    print(f"Saved confusion matrix plot to {filepath}")

def evaluate_model(y_true, y_pred, classes, model_name="Model"):
    """
    Calculates Macro F1 and prints the classification report.
    Generates visual plots.
    """
    print(f"\n{'='*40}")
    print(f"Evaluation for {model_name}")
    print(f"{'='*40}")
    
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print(f"Micro F1-Score: {micro_f1:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    
    plot_confusion_matrix(y_true, y_pred, classes, model_name)
    return macro_f1
