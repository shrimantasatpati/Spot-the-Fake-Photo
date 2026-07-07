import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import time

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spot the Fake Photo",
    page_icon="🔍",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #ffffff;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .result-box {
        border-radius: 16px;
        padding: 28px 32px;
        margin-top: 20px;
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        animation: fadeInUp 0.4s ease;
    }

    .result-real {
        background: linear-gradient(135deg, #065f46, #047857);
        border: 2px solid #34d399;
        color: #a7f3d0;
    }

    .result-fake {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 2px solid #f87171;
        color: #fecaca;
    }

    .confidence-label {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 4px;
        font-weight: 400;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stFileUploader label { color: #c4b5fd !important; }

    div[data-testid="stMetricValue"] { font-size: 2rem !important; }

    .stSpinner > div { border-color: #a78bfa transparent transparent; }
</style>
""", unsafe_allow_html=True)

# ── Model loading ─────────────────────────────────────────────────────────────
MODEL_PATH = "model.pth"
CLASS_NAMES = ["real", "screen"]   # alphabetical = ImageFolder order

@st.cache_resource
def load_model():
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

inference_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def predict(image: Image.Image):
    model = load_model()
    tensor = inference_transforms(image.convert("RGB")).unsqueeze(0)
    t0 = time.time()
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
    latency_ms = (time.time() - t0) * 1000
    pred_idx = probs.argmax().item()
    return CLASS_NAMES[pred_idx], probs[pred_idx].item(), latency_ms

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🔍 Spot the Fake Photo</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Upload an image — AI instantly tells you if it\'s a real photo or a screen screenshot.</div>', unsafe_allow_html=True)

st.divider()

tab_upload, tab_camera = st.tabs(["📁 Upload Image", "📷 Take Photo"])

def show_result(label, confidence, latency_ms):
    is_real = label == "real"
    icon = "✅" if is_real else "🚫"
    box_class = "result-real" if is_real else "result-fake"
    result_text = "REAL PHOTO" if is_real else "SCREEN / FAKE"
    st.markdown(
        f'<div class="result-box {box_class}">'
        f'{icon} {result_text}'
        f'<div class="confidence-label">Confidence: {confidence*100:.1f}% &nbsp;|&nbsp; Latency: {latency_ms:.0f} ms</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with tab_upload:
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload any photo to classify it as Real or Fake (screen)"
    )
    if uploaded:
        image = Image.open(io.BytesIO(uploaded.read()))
        st.image(image, caption="Uploaded image", use_column_width=True)
        with st.spinner("Analyzing…"):
            label, confidence, latency_ms = predict(image)
        show_result(label, confidence, latency_ms)

with tab_camera:
    photo = st.camera_input("Take a photo with your camera")
    if photo:
        image = Image.open(io.BytesIO(photo.read()))
        with st.spinner("Analyzing…"):
            label, confidence, latency_ms = predict(image)
        show_result(label, confidence, latency_ms)

st.divider()
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.8rem;'>"
    "Powered by MobileNetV3-Small · Transfer Learning · Trained on 100 real images"
    "</p>",
    unsafe_allow_html=True
)
