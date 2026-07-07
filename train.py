import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os

# Configuration
DATA_DIR = 'dataset'
MODEL_PATH = 'model.pth'
BATCH_SIZE = 16
PHASE1_EPOCHS = 10
PHASE2_EPOCHS = 10
LEARNING_RATE_1 = 0.001
LEARNING_RATE_2 = 1e-5

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found.")
        return
    
    classes = os.listdir(DATA_DIR)
    
    data_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
    class_names = full_dataset.classes
    dataloader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Freeze the feature extraction layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classification head and increase Dropout for regularization
    num_ftrs = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_1 = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE_1, weight_decay=1e-4)

    print("--- Phase 1: Training Classification Head ---")
    for epoch in range(PHASE1_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_1.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_1.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(full_dataset)
        print(f"Epoch {epoch+1}/{PHASE1_EPOCHS} - Acc: {epoch_acc:.4f}")

    print("--- Phase 2: Fine-Tuning Top Layers ---")
    # Unfreeze the top layers (the last few convolutional blocks)
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    # Use a much smaller learning rate
    optimizer_2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                             lr=LEARNING_RATE_2, weight_decay=1e-4)

    for epoch in range(PHASE2_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_2.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_2.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(full_dataset)
        print(f"Epoch {epoch+1}/{PHASE2_EPOCHS} - Acc: {epoch_acc:.4f}")

    print("Training complete!")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
