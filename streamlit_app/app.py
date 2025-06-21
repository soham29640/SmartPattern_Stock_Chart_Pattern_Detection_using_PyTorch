import torch
from PIL import Image
import streamlit as st
import torchvision.transforms as transforms
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model import ChartPatternCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChartPatternCNN(num_classes=20)
model.load_state_dict(torch.load("models/chart_pattern_model.pth", map_location=device))
model.to(device)
model.eval()

class_names = [
    'Ascending-Triangle', 'Channel-down', 'Channel-up', 'Cup-and-handle', 'Descending-Triangle',
    'Double-Bottom', 'Double-Top', 'Falling-Wedge', 'Head-Shoulders', 'Inverse-Head-Shoulders',
    'Resistance-Emerging', 'Resistance-breakout', 'Rising-Wedge', 'Rounding-Bottom',
    'Rounding-Top', 'Support-breakout', 'Triangle', 'Triple-Bottom', 'Triple-Top', 'rectangle'
]

st.title("📈 Chart Pattern Recognition")
st.write("Upload a candlestick chart image, and the model will detect the pattern.")

uploaded_file = st.file_uploader("Choose a chart image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)[0]
        top_probs, top_idxs = torch.topk(probs, 3)

    st.success("🧠 Top Predictions:")
    for i in range(3):
        label = class_names[top_idxs[i]]
        confidence = top_probs[i].item() * 100
        st.write(f"{i+1}. **{label}** — {confidence:.2f}% confidence")

    top_df = pd.DataFrame({
        'Pattern': [class_names[i] for i in top_idxs],
        'Confidence (%)': [round(p.item() * 100, 2) for p in top_probs]
    })

    st.bar_chart(top_df.set_index('Pattern'))
