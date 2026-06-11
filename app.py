import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# =========================
# Page Title
# =========================

st.set_page_config(page_title="Deepfake Detector")

st.title("🔍 Deepfake Image Detector")
st.write("Upload an image and the AI model will classify it as Fake or Real.")

# =========================
# Device
# =========================

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# =========================
# Class Names
# =========================

class_names = ["Fake", "Real"]

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

@st.cache_resource
def load_model():

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

    return model

model = load_model()

# =========================
# Upload Image
# =========================

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

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

    st.subheader("Prediction")

    st.success(
        f"{prediction}"
    )

    st.subheader("Confidence")

    st.write(
        f"{confidence:.2f}%"
    )