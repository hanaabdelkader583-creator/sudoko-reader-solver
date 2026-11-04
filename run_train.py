"""
run_train.py
Simple runner script to execute training and force console output.
"""
from pathlib import Path
from src.nn.train import main

if __name__ == "__main__":
    print("Starting CNN training wrapper...")
    # Passes arguments directly to the training script main function
    main(["--epochs", "5", "--source", "raw"])
    print("Training process finished!")