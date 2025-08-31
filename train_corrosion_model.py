# train_corrosion_model.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# === Settings ===
data_dir = "corrosion-segmentation-5/train"
model_save_path = "models/corrosion_model.pth"
num_epochs = 50
batch_size = 8
learning_rate = 0.001
image_size = (224, 224)

# Create models folder
os.makedirs("models", exist_ok=True)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Transforms ===
transform = T.Compose([
    T.Resize(image_size),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# === Load Data ===
dataset = ImageFolder(root=data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"Found {len(dataset)} images in {len(dataset.classes)} classes: {dataset.classes}")

# === Create Model ===
model = torch.hub.load('pytorch/vision', 'resnet18', weights='ResNet18_Weights.DEFAULT')
num_classes = 2
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# === Training Loop ===
model.train()
loss_history = []

for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for images, labels in progress:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress.set_postfix(loss=loss.item(), acc=100.*correct/total)

    epoch_loss = running_loss / len(dataloader)
    loss_history.append(epoch_loss)
    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {100.*correct/total:.2f}%")

# Save model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Plot loss
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("training_loss.png")
plt.show()

print("✅ Training complete! Now update your model.py to use this model.")