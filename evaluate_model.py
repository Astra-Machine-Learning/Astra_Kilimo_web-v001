# evaluate_model.py

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
DATA_DIR = 'PlantVillage'
MODEL_PATH = 'plant_disease_model.pth'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Dataset
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
val_loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Model
num_classes = len(dataset.classes)
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"✅ Evaluation complete. Accuracy: {accuracy:.2f}%")
