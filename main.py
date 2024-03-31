import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from cnn import CNN


def main():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root='./data/train_images', transform=transform)

    batch_size = 64
    num_classes = 15
    learning_rate = 0.001
    epochs = 100
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = CNN(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
            loss = criterion(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch}/{epochs} | Loss: {loss.item()}")
