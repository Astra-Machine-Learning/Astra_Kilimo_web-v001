# ============================================
# Astra Kilimo Training Script (Corrected)
# ============================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import kagglehub

# -----------------------------
# 1. Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("⚡ Using device:", device)

# Download PlantVillage dataset (KaggleHub)
path = kagglehub.dataset_download("emmarex/plantdisease")
DATA_DIR = os.path.join(path, "PlantVillage")   # ✅ Correct path
print("📂 Dataset path:", DATA_DIR)

# -----------------------------
# 2. Data preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
num_classes = len(dataset.classes)
print(f"✅ Found {num_classes} classes:", dataset.classes[:10], "...")

# Train/Validation split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# -----------------------------
# 3. Model setup
# -----------------------------
model = models.resnet18(weights="IMAGENET1K_V1")   # ✅ proper pretrained weights
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# -----------------------------
# 4. Training loop
# -----------------------------
NUM_EPOCHS = 10
MODEL_PATH = "astra_kilimo_model.pth"
best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
          f"Train Loss: {running_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_names': dataset.classes
        }, MODEL_PATH)
        print(f"✅ Best model saved with Val Acc: {best_acc:.2f}%")

    scheduler.step()

print("🎉 Training finished! Best Accuracy:", best_acc)
