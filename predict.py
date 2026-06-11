import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# Class Names
# =========================

class_names = ["Fake", "Real"]

# =========================
# Device
# =========================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# =========================
# Image Transform
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
# Load Model
# =========================

model = models.resnet18()

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    torch.load(
        "deepfake_detector.pth",
        map_location=device
    )
)

model = model.to(device)

model.eval()

# =========================
# Load Image
# =========================

#image_path = "test.jpg"

image_path = "/Users/sribalaayyappaswamybezawada/Visual Studio Code/Deepfake_detector/datasets/Test/Real/real_7.jpg"
image = Image.open(image_path).convert("RGB")

image = transform(image)

image = image.unsqueeze(0)

image = image.to(device)

# =========================
# Prediction
# =========================

with torch.no_grad():

    outputs = model(image)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted = torch.max(
        probabilities,
        1
    )

prediction = class_names[predicted.item()]

confidence = confidence.item() * 100

print(f"Prediction: {prediction}")
print(f"Confidence: {confidence:.2f}%")