import os
import base64
import time
from io import BytesIO
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

# Load model globally
MODEL_PATH = 'model.pth'
device = torch.device("cpu")
model = None

def load_model():
    global model
    if model is not None:
        return
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400

        # The image is base64 encoded data URI: "data:image/jpeg;base64,..."
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        
        start_time = time.time()
        input_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            screen_prob = probabilities[1].item()
            
        latency_ms = (time.time() - start_time) * 1000
        
        return jsonify({
            'is_screen': bool(screen_prob > 0.5),
            'probability': screen_prob,
            'latency_ms': latency_ms
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=7860, debug=False)
