import os
import sys

import torch
import pandas as pd
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image

# Allow running directly from the app/ directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.model import ChartPatternCNN

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = "models/chart_pattern_model.pth"
NUM_CLASSES = 20
THRESHOLD   = 0.5   # sigmoid confidence cutoff for a positive detection
TOP_K       = 3     # how many patterns to surface in the UI

CLASS_NAMES = [
    "Ascending-Triangle", "Channel-down", "Channel-up", "Cup-and-handle",
    "Descending-Triangle", "Double-Bottom", "Double-Top", "Falling-Wedge",
    "Head-Shoulders", "Inverse-Head-Shoulders", "Resistance-Emerging",
    "Resistance-breakout", "Rising-Wedge", "Rounding-Bottom", "Rounding-Top",
    "Support-breakout", "Triangle", "Triple-Bottom", "Triple-Top", "Rectangle",
]

# ── Inference transform (no augmentation) ─────────────────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


@st.cache_resource(show_spinner="Loading model …")
def load_model() -> tuple[ChartPatternCNN, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ChartPatternCNN(num_classes=NUM_CLASSES)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    model.to(device).eval()
    return model, device


def predict(image: Image.Image, model: ChartPatternCNN, device: torch.device):
    """
    Run multi-label inference.

    Returns:
        probs     : tensor of per-class sigmoid probabilities (shape NUM_CLASSES)
        top_idxs  : indices of top-K classes by confidence
        top_probs : corresponding probabilities
    """
    tensor = INFER_TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        # ✅ sigmoid — correct for multi-label; softmax would be wrong here
        probs  = torch.sigmoid(logits)[0]

    top_probs, top_idxs = torch.topk(probs, TOP_K)
    return probs, top_idxs, top_probs


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chart Pattern Detector", layout="centered")
st.title("📈 Chart Pattern Recognition")
st.write(
    "Upload a **candlestick chart image** and the model will predict "
    f"the top {TOP_K} most likely chart patterns."
)

# ── Sidebar upload ─────────────────────────────────────────────────────────────
uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Chart Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("⬅️ Please upload a chart image from the sidebar.")
    st.stop()

# ── Image display ──────────────────────────────────────────────────────────────
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

# ── Inference ──────────────────────────────────────────────────────────────────
model, device = load_model()

with st.spinner("Analysing …"):
    all_probs, top_idxs, top_probs = predict(image, model, device)

# ── Results ────────────────────────────────────────────────────────────────────
st.markdown(f"## 🚀 Top {TOP_K} Predicted Patterns")

for rank, (idx, prob) in enumerate(zip(top_idxs.tolist(), top_probs.tolist()), start=1):
    label      = CLASS_NAMES[idx]
    confidence = prob * 100
    detected   = "✅" if prob >= THRESHOLD else "⚠️ low confidence"
    st.markdown(f"**{rank}. {label}** — `{confidence:.2f}%` {detected}")

# ── Confidence chart ───────────────────────────────────────────────────────────
chart_df = pd.DataFrame({
    "Pattern":        [CLASS_NAMES[i] for i in top_idxs.tolist()],
    "Confidence (%)": [round(p * 100, 2) for p in top_probs.tolist()],
}).set_index("Pattern")

st.markdown("### 📊 Confidence Chart")
st.bar_chart(chart_df)

# ── All-class breakdown (expandable) ──────────────────────────────────────────
with st.expander("🔍 Show all class probabilities"):
    all_df = pd.DataFrame({
        "Pattern":        CLASS_NAMES,
        "Confidence (%)": [round(p * 100, 2) for p in all_probs.tolist()],
    }).sort_values("Confidence (%)", ascending=False)
    st.dataframe(all_df, use_container_width=True)