import torch
import torch.nn as nn
import numpy as np
import pandas as pd



class CNN(nn.Module):
    def __init__(self, num_classes):
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

        self.classifier = nn.Sequential(
            nn.Linear(32, 512),
            nn.Dropout(),
            nn.Linear(512,num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.reshape(features.size(0), -1)

        class_scores = self.classifier(features)

        return class_scores
    
    def train_model(self, X, y, epochs=50, lr=0.00001, batch_size=64):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.to(self.device)

        num_batches = int(np.ceil(len(X) / batch_size))

        for epoch in range(epochs):
            permutation = torch.randperm(len(X))
            X = X[permutation]
            y = y[permutation]
            for batch_idx in range(num_batches):
            # Select data for this batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(X))

                X_batch = X[start_idx:end_idx].to(self.device)
                y_batch = y[start_idx:end_idx].to(self.device)

                y_hat = self(X_batch)
                loss = criterion(y_hat, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch: {epoch+1}/{epochs} | Loss: {loss.item()}")
    
    def predict(self, X):
        X = X.to(self.device)
        self.eval()
        y_hat = []

        with torch.no_grad():
            y_hat = self(X)
            # _, predicted = torch.softmax(outputs, 1)
            # y_hat.extend(predicted.cpu().numpy())
    
        return y_hat.to('cpu')



