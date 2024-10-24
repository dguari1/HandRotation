print("resnet_regression.py")

import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image, ImageFilter
from custom.custom_dataset import CustomDataset
import torch.nn as nn
import random

# Updated function to get the next model index by checking directories
def get_next_model_idx(save_dir='[MODEL_SAVE_DIRECTORY]'):
    # List all entries in the directory
    entries = os.listdir(save_dir)
    # Filter out directories that match the pattern 'best_model_<number>'
    model_dirs = [d for d in entries if os.path.isdir(os.path.join(save_dir, d)) and d.startswith('best_model_')]
    
    if not model_dirs:
        return 1  # If no matching directories found, start with 1

    # Extract indices from directory names
    indices = [int(d.split('_')[-1]) for d in model_dirs if d.split('_')[-1].isdigit()]
    return max(indices) + 1 if indices else 1  # Increment the maximum index found

class ResNetForRegression(nn.Module):
    def __init__(self):
        super(ResNetForRegression, self).__init__()
        self.resnet = models.resnet50(pretrained=True)  # Changed to resnet50
        # Replace the final classification layer with a regression head
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 1)  # Output a single continuous value for regression

    def forward(self, x):
        return self.resnet(x)

# Initialize the model
model = ResNetForRegression()

# Paths to save the model
weights_file_path = '[MODEL_SAVE_DIRECTORY]'
weights_file_idx = get_next_model_idx(weights_file_path)
model_save_dir = os.path.join(weights_file_path, f'best_model_{weights_file_idx}')
os.makedirs(model_save_dir, exist_ok=True)  # Create the directory to save models

print(f"Model save directory created: {model_save_dir}")

""" Hyperparameters """
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 100
patience = 10  # Number of epochs to wait for improvement before stopping

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class EdgeDetection:
    def __call__(self, img):
        return img.filter(ImageFilter.FIND_EDGES)

class RandomEdgeDetection:
    def __init__(self, probability=0.2):
        self.probability = probability

    def __call__(self, img):
        if random.random() < self.probability:
            return img.filter(ImageFilter.FIND_EDGES)
        return img

# Define your transformation pipeline
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0)),  # Random crop between 80% to 100% of the image
    transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),  # Random shift up to 30% in all directions
    transforms.Resize((224, 224)),  # Ensure the final size is 224x224
    transforms.ColorJitter(brightness=0., contrast=0.2, saturation=0.2, hue=0.1),  # Adjust color properties
    RandomEdgeDetection(probability=0.4),  # Randomly apply edge detection to 40% of the images
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

""" Prepare Dataloaders """
# Train dataloader
train_images_dir = '[TRAIN_IMAGES_DIRECTORY]'
train_labels_dir = '[TRAIN_LABELS_DIRECTORY]'
train_dataset = CustomDataset(train_images_dir, train_labels_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=11)

# Val dataloader
val_images_dir = '[VAL_IMAGES_DIRECTORY]'
val_labels_dir = '[VAL_LABELS_DIRECTORY]'
val_dataset = CustomDataset(val_images_dir, val_labels_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=11)

@torch.no_grad()
def evaluate(model, val_loader, criterion):
    model.eval()
    correct = 0
    top_5_correct = 0
    total = 0
    avg_loss = 0
    for images, labels in val_loader:
        images, labels = images.to(device=device, dtype=torch.float32), labels.to(device=device, dtype=torch.float32)
        outputs = model(images)  # Directly get the regression output
        predicted = outputs.squeeze()  
        avg_loss += criterion(predicted, labels).item()  # Compute MSE loss
        total += labels.size(0)
        correct += torch.sum(torch.abs(predicted - labels) < 1).item()  # Within 1 degree
        top_5_correct += torch.sum(torch.abs(predicted - labels) < 5).item()  # Within 5 degrees

    return 100 * correct / total, 100 * top_5_correct / total, avg_loss / len(val_loader)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.MSELoss()  # Use MSE Loss for regression

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Track the best model for each metric
best_val_acc = 0.0
best_top_5_acc = 0.0
best_val_loss = float('inf')
epochs_no_improve = 0  # Initialize the counter for early stopping

print("Training started")

for epoch in range(num_epochs):
    model.train()  
    avg_loss = 0  
    
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device=device, dtype=torch.float32), labels.to(device=device, dtype=torch.float32)
        
        optimizer.zero_grad() 
        outputs = model(imgs)  # Directly get the regression output
        loss = criterion(outputs.squeeze(), labels)  # Ensure labels are in float format for regression
        loss.backward()  
        optimizer.step() 
        avg_loss += loss.item()  
    
    avg_loss /= len(train_loader)
    
    val_acc, val_top_5_acc, avg_val_loss = evaluate(model, val_loader, criterion)
    
    print(f"Epoch: {epoch+1}/{num_epochs}, Loss: {avg_loss:.5f}, Val Acc: {val_acc:.5f}, Val Top 5 Acc: {val_top_5_acc:.5f}, Val Loss: {avg_val_loss:.4f}")
    
    # Flag to check if there was an improvement
    improved = False

    # Save model based on validation accuracy
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        model_path = os.path.join(model_save_dir, 'best_val_acc.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved with Best Val Acc: {best_val_acc:.5f}")
        improved = True

    # Save model based on top-5 validation accuracy
    if val_top_5_acc > best_top_5_acc:
        best_top_5_acc = val_top_5_acc
        model_path = os.path.join(model_save_dir, 'best_top_5_acc.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved with Best Top 5 Val Acc: {best_top_5_acc:.5f}")
        improved = True

    # Save model based on validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model_path = os.path.join(model_save_dir, 'best_val_loss.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved with Best Val Loss: {best_val_loss:.4f}")
        improved = True

    # Check if there was any improvement; if so, reset the counter
    if improved:
        epochs_no_improve = 0  # Reset the counter when improvement happens
    else:
        epochs_no_improve += 1  # Increment the counter if no improvement

    # Early stopping
    if epochs_no_improve >= patience:
        print(f'Early stopping at epoch {epoch+1}. No improvement for {patience} consecutive epochs.')
        break

print("Training finished")
print(f"Model weights saved to {model_save_dir}")
