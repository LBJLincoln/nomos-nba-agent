import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

class CustomLossCalibrator(nn.Module):
    """
    Custom loss function and probability calibrator for improved model performance.
    """
    
    def __init__(self, model, device='cpu'):
        super(CustomLossCalibrator, self).__init__()
        self.model = model
        self.device = device
        
        # Custom loss function: weighted Brier score
        self.loss_fn = nn.BCELoss(reduction='mean')
        
        # Probability calibrator: isotonic regression
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        
    def forward(self, x):
        return self.model(x)
    
    def custom_loss(self, predictions, targets):
        # Weighted Brier score
        weights = torch.tensor([0.7, 0.3], device=self.device)  # Example weights
        loss = self.loss_fn(predictions, targets)
        weighted_loss = loss * weights[0] + (1 - loss) * weights[1]
        return weighted_loss
    
    def calibrate_probabilities(self, predictions, targets):
        # Calibrate probabilities using isotonic regression
        self.calibrator.fit(predictions.detach().cpu().numpy(), targets.cpu().numpy())
        calibrated_probabilities = self.calibrator.predict(predictions.detach().cpu().numpy())
        return torch.tensor(calibrated_probabilities, device=self.device)
    
    def train(self, train_loader, optimizer, epochs=10):
        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.forward(x)
                loss = self.custom_loss(predictions, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")
                
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                predictions = self.forward(x)
                loss = self.custom_loss(predictions, y)
                total_loss += loss.item()
        average_loss = total_loss / len(test_loader)
        print(f"Test Loss: {average_loss:.4f}")
        
    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            calibrated_probabilities = self.calibrate_probabilities(predictions, torch.zeros_like(predictions))
            return calibrated_probabilities

# Example usage:
# model = CustomLossCalibrator(nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1), nn.Sigmoid()))
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
# optimizer = Adam(model.parameters(), lr=0.001)
# model.train(train_loader, optimizer, epochs=10)
# test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
# model.evaluate(test_loader)
# predictions = model.predict(torch.randn(1, 10))

