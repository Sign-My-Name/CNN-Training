import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from cnn import CNN


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    ])

train_dataset = datasets.ImageFolder(root='./data/transformed', transform=transform)


X = []  # to store the image tensors
y = []  # to store the labels

for img, label in train_dataset:
    X.append(img)
    y.append(label)

# Convert the lists to a tensor and a LongTensor, respectively, if needed
X = torch.stack(X)
y = torch.LongTensor(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


loaded_model = torch.load('./model_batch128-001lr')


correct = 0
total = 0
loaded_model.to(device)


y_pred = loaded_model.predict(X_test) # Only Test 


print(((torch.argmax(y_pred,1)==y_test).sum().item())/len(y_test)) # batch 