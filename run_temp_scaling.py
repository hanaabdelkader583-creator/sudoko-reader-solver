import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from src.nn.train import load_raw_mnist
import src.nn.model as model_module

def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    ece = 0.0
    for i in range(n_bins):
        bin_mask = (confidences >= bin_limits[i]) & (confidences < bin_limits[i+1])
        if np.any(bin_mask):
            acc = np.mean(y_true[bin_mask] == predictions[bin_mask])
            conf = np.mean(confidences[bin_mask])
            ece += np.sum(bin_mask) / len(y_true) * np.abs(acc - conf)
    return ece

class ModelWithTemperature(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return logits / self.temperature

def main():
    print("Loading validation data...")
    raw_dir = Path("data/MNIST/raw")
    splits = load_raw_mnist(raw_dir, val_fraction=0.1, seed=42)
    
    X_val = torch.tensor(splits["val"][0], dtype=torch.float32).unsqueeze(1) / 255.0
    y_val = torch.tensor(splits["val"][1], dtype=torch.long)

    print("Loading CNN model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_classes = [getattr(model_module, m) for m in dir(model_module) 
                     if isinstance(getattr(model_module, m), type) 
                     and issubclass(getattr(model_module, m), nn.Module) 
                     and m != 'Module']
    
    model = model_classes[0]()
    loaded = torch.load("models/sudoku_digit_cnn.pt", map_location=device, weights_only=True)
    
    if isinstance(loaded, dict):
        model.load_state_dict(loaded.get("state_dict", loaded))
    else:
        model = loaded
        
    model.to(device)
    model.eval()

    print("Evaluating initial calibration...")
    with torch.no_grad():
        X_val_device = X_val.to(device)
        logits_before = model(X_val_device)
        probs_before = torch.softmax(logits_before, dim=1).cpu().numpy()
        
    ece_before = expected_calibration_error(y_val.numpy(), probs_before)
    print(f"\n---> ECE BEFORE Temperature Scaling: {ece_before:.4f}")

    print("\nOptimizing Temperature (T)... please wait a moment...")
    scaled_model = ModelWithTemperature(model).to(device)
    nll_criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([scaled_model.temperature], lr=0.01, max_iter=20) # reduced max_iter for speed

    def eval_closure():
        optimizer.zero_grad()
        loss = nll_criterion(scaled_model(X_val_device), y_val.to(device))
        loss.backward()
        print(f"   [Optimization step] Current Loss: {loss.item():.4f} | Temp: {scaled_model.temperature.item():.4f}")
        return loss

    optimizer.step(eval_closure)
    optimal_t = scaled_model.temperature.item()
    print(f"\n---> Optimal Temperature (T): {optimal_t:.4f}")

    with torch.no_grad():
        logits_after = scaled_model(X_val_device)
        probs_after = torch.softmax(logits_after, dim=1).cpu().numpy()

    ece_after = expected_calibration_error(y_val.numpy(), probs_after)
    print(f"---> ECE AFTER Temperature Scaling: {ece_after:.4f}\n")

if __name__ == "__main__":
    main()