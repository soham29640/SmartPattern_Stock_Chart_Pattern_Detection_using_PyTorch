import torch
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model import ChartPatternCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChartPatternCNN(num_classes=20)
model.load_state_dict(torch.load("models/chart_pattern_model.h5", map_location=device))
model.to(device)
model.eval()

class_labels = [
    "Head and Shoulders", "Double Top", "Double Bottom", "Cup and Handle", "Flag",
    "Pennant", "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle",
    "Rounding Bottom", "Wedge", "Rectangle", "Triple Top", "Triple Bottom",
    "Breakout", "Fakeout", "Channel Up", "Channel Down", "Parabolic Curve", "Other"
]

st.title("📈 Chart Pattern Recognition")
st.write("Upload a candlestick chart image, and the model will detect the pattern.")

uploaded_file = st.file_uploader("Choose a chart image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        confidence = torch.softmax(output, dim=1)[0][predicted_class].item()

    label = class_labels[predicted_class]
    st.success(f"🧠 Predicted Pattern: **{label}**")
    st.info(f"Confidence: {confidence*100:.2f}%")
