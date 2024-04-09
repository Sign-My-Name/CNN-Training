import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader



class CNN(nn.Module):
    def __init__(self, num_classes, pic_size=128):
        print('hello world')
        super(CNN, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )

        x = torch.randn(1, 1, pic_size, pic_size)  # Adjust the size here to match your input size
        x = self.feature_extractor(x)
        feature_size = x.numel()

        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.Dropout(),
            nn.Linear(512,num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)

        class_scores = self.classifier(features)

        return class_scores
    
    def train_model(self, X, y, epochs=50, lr=0.00001, batch_size=128):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

        X = torch.tensor(X)
        y = torch.tensor(y)
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size)

        for epoch in range(epochs):
            for batch in dataloader:
            # Select data for this batch
                X_batch, y_batch = batch

                y_hat = self(X_batch)
                loss = criterion(y_hat, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch+1}/{epochs} | Loss: {loss.item()}")
    
    def predict(self, X, batch_size=128):
        self.eval()
        y_hat = []

        X = torch.tensor(X)
        dataloader = DataLoader(dataset=X, batch_size=batch_size)

        with torch.no_grad():
            for X_batch in dataloader:
            # Select data for this batch
                batch_predictions = self(X_batch)
                y_hat.append(batch_predictions)

        y_hat = torch.cat(y_hat, dim=0)
        
        return y_hat.to('cpu')



