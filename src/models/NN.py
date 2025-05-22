import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

class BaseNN(nn.Module):
    # input: 4096
    # output: 2 (x,y coordinates)
    def __init__(self):
        super(BaseNN, self).__init__()
        
        # Simpler architecture with just linear layers and ReLU
        self.layers = nn.ModuleList([       
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        ])

    def forward(self, x):
        # Forward pass through layers
        for layer in self.layers:
            x = layer(x)
            
        return x

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class EnsembleNN(nn.Module):
    def __init__(self, num_models=8):
        super(EnsembleNN, self).__init__()
        self.models = nn.ModuleList([BaseNN() for _ in range(num_models)])
        self.register_buffer('input_min', None)
        self.register_buffer('input_max', None)
        
    def normalize_input(self, x):
        if self.input_min is None or self.input_max is None:
            raise ValueError("Normalization parameters not set. Call set_normalization_params first.")
        
        # Avoid division by zero by adding small epsilon where max equals min
        eps = 1e-8
        denominator = (self.input_max - self.input_min)
        # Where max equals min, set denominator to 1 to keep the original value
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        
        # Scale to [-1, 1]
        x_normalized = 2 * (x - self.input_min) / (denominator + eps) - 1
        return x_normalized
        
    def set_normalization_params(self, dataloader):
        """Compute min and max values for each dimension from training data"""
        print("Computing normalization parameters...")
        device = next(self.parameters()).device
        
        # Initialize min and max
        first_batch = True
        
        for csi, _, _ in dataloader:
            csi = csi.to(device)
            x = csi.flatten(start_dim=1)
            x = torch.cat((torch.real(x), torch.imag(x)), dim=1)
            
            if first_batch:
                current_min = torch.min(x, dim=0)[0]
                current_max = torch.max(x, dim=0)[0]
                first_batch = False
            else:
                current_min = torch.min(torch.stack([current_min, torch.min(x, dim=0)[0]]), dim=0)[0]
                current_max = torch.max(torch.stack([current_max, torch.max(x, dim=0)[0]]), dim=0)[0]
        
        self.register_buffer('input_min', current_min)
        self.register_buffer('input_max', current_max)
        print("Normalization parameters computed and set.")
        
    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)
        x = self.normalize_input(x)
        
        # Get predictions from all models
        predictions = torch.stack([model(x) for model in self.models])
        # Average the predictions
        return torch.mean(predictions, dim=0)
    
    def forward_all(self, x):
        x = x.flatten(start_dim=1)
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)
        x = self.normalize_input(x)
        
        # Returns both individual predictions and ensemble prediction
        individual_preds = torch.stack([model(x) for model in self.models])
        ensemble_pred = torch.mean(individual_preds, dim=0)
        return ensemble_pred, individual_preds
    
    def count_parameters(self):
        return sum(model.count_parameters() for model in self.models)