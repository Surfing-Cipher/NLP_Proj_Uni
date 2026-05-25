import argparse
import sys
import os

# Add src to path so we can import from it easily
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data
from preprocessing import preprocess_dataset
from train_baseline import train_and_evaluate_baseline
from train_dl import train_and_evaluate_dl

def main():
    parser = argparse.ArgumentParser(description="NLP Classification Pipeline")
    parser.add_argument('--run-baseline', action='store_true', help="Run Track A (TF-IDF + Random Forest)")
    parser.add_argument('--run-dl', action='store_true', help="Run Track B (PyTorch Bi-LSTM)")
    parser.add_argument('--run-all', action='store_true', help="Run entire pipeline (both tracks)")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        print("Please specify a flag. Example: python main.py --run-all")
        parser.print_help()
        sys.exit(1)
        
    print("\n--- Phase 1: Data Loading ---")
    df = load_data()
    
    print("\n--- Phase 2: Linguistic Preprocessing ---")
    df = preprocess_dataset(df)
    
    if args.run_baseline or args.run_all:
        print("\n--- Phase 3: Track A (Baseline ML) ---")
        train_and_evaluate_baseline(df)
        
    if args.run_dl or args.run_all:
        print("\n--- Phase 4: Track B (Deep Learning) ---")
        train_and_evaluate_dl(df)

if __name__ == "__main__":
    main()
