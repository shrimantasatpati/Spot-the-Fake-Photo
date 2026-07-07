# Spot the Fake Photo

**Salescode.ai Computer Vision Take-Home Assignment**

> Given one image, decide whether it is a **REAL photo** or a **PHOTO OF A SCREEN** (a "recapture").

---

## Live Demo

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/shrimantasatpati/Spot-the-Fake-Photo/main/streamlit_app.py)

Upload any image or use your device camera — the model classifies it instantly.

![Demo](https://raw.githubusercontent.com/shrimantasatpati/Spot-the-Fake-Photo/main/demo_preview.png)

---

## How It Works

Fine-tuned **MobileNetV3-Small** (ImageNet pre-trained) to detect screen artefacts:
- Moire patterns
- Pixel grids
- Screen glare & colour shift

**Two-phase training:**
1. Phase 1 (20 epochs) — freeze backbone, train head only (LR = 1e-3)
2. Phase 2 (25 epochs) — unfreeze last 4 blocks, fine-tune textures (LR = 3e-5)

**Regularization:** Dropout 0.5 · L2 weight decay · Label smoothing 0.1 · Aggressive augmentation

---

## Accuracy

| Class | Correct | Total | Accuracy |
|-------|---------|-------|----------|
| Real  | 50      | 50    | **100%** |
| Screen| 48      | 50    | **96%**  |
| **Overall** | **98** | **100** | **98%** |

---

## Performance

| Metric | Value |
|--------|-------|
| Inference latency | **~21 ms** (Laptop CPU) |
| On-device cost | **$0.00** (runs free on phone) |
| Cloud cost (AWS Lambda) | **~$0.89 per 1M images** |

---

## Repository Structure

```
.
├── streamlit_app.py   # Live web demo (upload or camera)
├── predict.py         # CLI predictor: python predict.py image.jpg
├── train.py           # Training script (MobileNetV3-Small, 2-phase)
├── model.pth          # Trained model weights
└── requirements.txt   # Dependencies
```

---

## Usage

### Run the predictor
```bash
python predict.py some_image.jpg
# → 0.07  (close to 0 = real photo, close to 1 = screen/fake)
```

### Run the web demo locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Retrain the model
```bash
# Put images in dataset/real/ and dataset/screen/
python train.py
```

---

## Deploy to Streamlit Cloud (Free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"Create app"** → select this repo
4. Set main file: `streamlit_app.py`
5. Deploy — done! Works on mobile with camera access.
