import sys
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import time

def predict(image_path, model_path='model.pth'):
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return
    if not os.path.exists(model_path):
        print(f"Error: Model {model_path} not found. Please run train.py first.")
        return

    # Measure latency
    start_time = time.time()

    # Load model architecture
    model = models.mobilenet_v3_small(weights=None)
    num_ftrs = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    
    # Load weights
    device = torch.device("cpu") # Inference is fast enough on CPU
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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
        print(f"Error loading image: {e}")
        return

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # Predict
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 'real' is class 0, 'screen' is class 1 (alphabetical order by ImageFolder)
        # We output the probability of it being a SCREEN (1 = photo of a screen)
        screen_prob = probabilities[1].item()
    
    latency_ms = (time.time() - start_time) * 1000
    
    # The assignment asks for a single float output:
    print(f"{screen_prob:.4f}")
    # Optional: uncomment below to see latency
    # print(f"Latency: {latency_ms:.2f} ms")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    predict(image_path)
