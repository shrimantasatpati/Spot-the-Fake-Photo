import torch
import torch.nn as nn
import os
from torchvision import models, transforms
from PIL import Image

model = models.mobilenet_v3_small(weights=None)
num_ftrs = model.classifier[3].in_features
model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
model.classifier[3] = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load('model.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['real', 'screen']
results = {'real': {'correct': 0, 'total': 0}, 'screen': {'correct': 0, 'total': 0}}

for cls in classes:
    folder = os.path.join('dataset', cls)
    for f in os.listdir(folder):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        path = os.path.join(folder, f)
        img = Image.open(path).convert('RGB')
        t = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(t)
            pred_idx = torch.softmax(out, 1)[0].argmax().item()
        results[cls]['total'] += 1
        if classes[pred_idx] == cls:
            results[cls]['correct'] += 1
        else:
            print("WRONG: " + cls + "/" + f + " -> predicted " + classes[pred_idx])

total = sum(v['total'] for v in results.values())
correct = sum(v['correct'] for v in results.values())
print("Overall: " + str(correct) + "/" + str(total) + " = " + str(round(correct/total*100, 1)) + "%")
for cls, v in results.items():
    pct = round(v['correct']/v['total']*100, 1)
    print("  " + cls + ": " + str(v['correct']) + "/" + str(v['total']) + " = " + str(pct) + "%")
