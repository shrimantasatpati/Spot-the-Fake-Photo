import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os

# Configuration
DATA_DIR = 'dataset'
MODEL_PATH = 'model.pth'
BATCH_SIZE = 8
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 25
LEARNING_RATE_1 = 1e-3
LEARNING_RATE_2 = 3e-5

def train_epoch(model, dataloader, criterion, optimizer, device, dataset_size):
    model.train()
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_corrects += torch.sum(preds == labels.data)
    return running_corrects.double() / dataset_size

def eval_epoch(model, dataloader, device, dataset_size):
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    return running_corrects.double() / dataset_size

def eval_epoch(model, dataloader, device, dataset_size):
    model.eval()
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    return running_corrects.double() / dataset_size

def main():
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found.")
        return

    # Train on ALL 100 images with aggressive augmentation for maximum generalization
    train_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(DATA_DIR, train_transforms)
    total = len(full_dataset)

    train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training on ALL {total} images (no val split) for maximum generalization.")

    # Load pre-trained MobileNetV3-Small
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head
    num_ftrs = model.classifier[3].in_features
    model.classifier[2] = nn.Dropout(p=0.5, inplace=True)
    model.classifier[3] = nn.Linear(num_ftrs, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer_1 = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE_1, weight_decay=1e-4)
    scheduler_1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_1, T_max=PHASE1_EPOCHS)

    best_train_acc = 0.0
    best_state = None

    print("--- Phase 1: Training Classification Head ---")
    for epoch in range(PHASE1_EPOCHS):
        train_acc = train_epoch(model, train_loader, criterion, optimizer_1, device, total)
        scheduler_1.step()
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1}/{PHASE1_EPOCHS} - Train: {train_acc:.4f}")

    print("--- Phase 2: Fine-Tuning Top Layers ---")
    # Unfreeze the last 4 feature blocks for screen texture adaptation
    for param in model.features[-4:].parameters():
        param.requires_grad = True

    optimizer_2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE_2, weight_decay=1e-4
    )
    scheduler_2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_2, T_max=PHASE2_EPOCHS)

    for epoch in range(PHASE2_EPOCHS):
        train_acc = train_epoch(model, train_loader, criterion, optimizer_2, device, total)
        scheduler_2.step()
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"Epoch {epoch+1}/{PHASE2_EPOCHS} - Train: {train_acc:.4f}")

    # Save best-epoch weights
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nTraining complete! Best Train Accuracy: {best_train_acc:.4f} ({best_train_acc*100:.1f}%)")
    print(f"Model saved to {MODEL_PATH}")

if __name__ == '__main__':
    main()
