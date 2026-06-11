"""import torch
import torch.nn as nn
from torchvision import models


from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(
    root = "datasets/train",
    transform = transform
)

#print("Classes:", train_dataset.classes)
#print("Total Images:", len(train_dataset))

image, label = train_dataset[0]


from torch.utils.data import Subset

train_dataset = Subset(
    train_dataset,
    range(5000)
)
#print("Image Shape:", image.shape)
#print("label:", label)

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size = 32,
    shuffle = True
)

images, labels = next(iter(train_loader))

val_dataset = datasets.ImageFolder(
    root="datasets/validation",
    transform=transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

#print("Batch Shape:", images.shape)
#print("Labels Shape:", labels.shape)

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = 0.001
)

#print("Using Device", device)
print("Training Images:", len(train_dataset))
print("Total Batches:", len(train_loader))
print("Using Device:", device)

epochs = 3

model.train()

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch[{epoch+1}/{epochs}],"
        f"Loss: {running_loss/len(train_loader):.4f}"
    )

    model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Validation Accuracy: {accuracy:.2f}%")
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# =========================
# Image Transformations
# =========================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =========================
# Datasets
# =========================

train_dataset = datasets.ImageFolder(
    root="datasets/train",
    transform=transform
)

val_dataset = datasets.ImageFolder(
    root="datasets/validation",
    transform=transform
)

# Use only first 5000 training images for now

from torch.utils.data import random_split

full_train_dataset = datasets.ImageFolder(
    root="datasets/train",
    transform=transform
)

subset_size = 5000

train_dataset, _ = random_split(
    full_train_dataset,
    [subset_size, len(full_train_dataset) - subset_size]
)

print("Training Images:", len(train_dataset))
print("Validation Images:", len(val_dataset))

print("Train Classes:", train_dataset.dataset.classes)
print("Validation Classes:", val_dataset.classes)

print("Train Mapping:", train_dataset.dataset.class_to_idx)
print("Validation Mapping:", val_dataset.class_to_idx)

# =========================
# DataLoaders
# =========================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)

print("Total Training Batches:", len(train_loader))

# =========================
# Device
# =========================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Using Device:", device)

# =========================
# Model
# =========================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model = model.to(device)

# =========================
# Loss & Optimizer
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# =========================
# Training
# =========================

epochs = 3

for epoch in range(epochs):

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {epoch_loss:.4f}"
    )

# =========================
# Validation
# =========================

model.eval()

correct = 0
total = 0

fake_predictions = 0
real_predictions = 0

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        fake_predictions += (predicted == 0).sum().item()
        real_predictions += (predicted == 1).sum().item()

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("\n========== RESULTS ==========")
print(f"Validation Accuracy: {accuracy:.2f}%")
print(f"Predicted Fake: {fake_predictions}")
print(f"Predicted Real: {real_predictions}")

# =========================
# Save Model
# =========================

torch.save(
    model.state_dict(),
    "deepfake_detector.pth"
)

print("\nModel saved as: deepfake_detector.pth")