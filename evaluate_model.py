# ================================================
# evaluate_model.py
# Evaluate Astra Kilimo trained model on dataset
# ================================================

import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os

# -----------------------------
# 1. Setup
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("⚡ Using device:", device)

# Paths
DATA_DIR = os.path.join("PlantVillage")   # <-- point to correct dataset folder
MODEL_PATH = "plant_disease_model.pth"     # <-- use your trained checkpoint

# -----------------------------
# 2. Transforms
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# 3. Dataset & Loader
# -----------------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

print(f"📂 Loaded dataset with {len(dataset)} images across {len(dataset.classes)} classes.")

# -----------------------------
# 4. Load Model
# -----------------------------
checkpoint = torch.load(MODEL_PATH, map_location=device)
num_classes = len(checkpoint['class_names'])

model = models.resnet18(weights=None)  # no pretrained needed
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print("✅ Model loaded with classes:", checkpoint['class_names'][:10], "...")

# -----------------------------
# 5. Evaluation
# -----------------------------
correct, total = 0, 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"🎯 Evaluation complete. Accuracy: {accuracy:.2f}%")