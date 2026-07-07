import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time


def build_model():
    """Build the exact same architecture used in train.py."""
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    return model


def predict(image_path, model_path='model.pth'):
    if not os.path.exists(image_path):
        print("Error: Image " + image_path + " not found.")
        return None
    if not os.path.exists(model_path):
        print("Error: Model " + model_path + " not found. Please run train.py first.")
        return None

    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print("Error loading image: " + str(e))
        return None

    input_tensor = preprocess(img).unsqueeze(0)  # [1, 3, 224, 224]

    # Load model (measure full latency including model load)
    start_time = time.time()
    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        # class 0 = real, class 1 = screen (alphabetical by ImageFolder)
        # Output: 0 = real photo, 1 = photo of a screen
        screen_prob = probabilities[1].item()

    latency_ms = (time.time() - start_time) * 1000

    # Required output: a single float from 0 to 1
    # 0 = real photo, 1 = photo of a screen
    print(round(screen_prob, 4))
    # Uncomment below to also see latency:
    # print("Latency: " + str(round(latency_ms, 1)) + " ms on CPU")
    return screen_prob


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        print("Output: a number 0 to 1.  0 = real photo, 1 = photo of a screen.")
        sys.exit(1)

    image_path = sys.argv[1]
    predict(image_path)
