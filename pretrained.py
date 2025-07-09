import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image

logging.basicConfig(filename='training_log.txt', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')

class MRIDataset(Dataset):
    def __init__(self, image_paths, label_dict, transform=None):
        self.image_paths = image_paths
        self.label_dict = label_dict
        self.transform = transform
        self.class_map = {'CN': 0, 'AD': 1}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.class_map[self.label_dict[os.path.basename(path)]]

        img = nib.load(path).get_fdata()

        # Get middle indices for each axis
        mid_x = img.shape[0] // 2
        mid_y = img.shape[1] // 2
        mid_z = img.shape[2] // 2

        # Extract one middle slice along each axis
        axial = img[:, :, mid_z]
        coronal = img[:, mid_y, :]
        sagittal = img[mid_x, :, :]

        # Normalize each slice to [0, 255] and convert to PIL Image
        def to_img(slice_2d):
            slice_norm = (slice_2d - np.min(slice_2d)) / (np.ptp(slice_2d) + 1e-8)
            return Image.fromarray(np.uint8(255 * slice_norm))

        axial_img = to_img(axial)
        coronal_img = to_img(coronal)
        sagittal_img = to_img(sagittal)

        # Apply transform to each slice
        if self.transform:
            axial_tensor = self.transform(axial_img)
            coronal_tensor = self.transform(coronal_img)
            sagittal_tensor = self.transform(sagittal_img)
        else:
            tform = transforms.ToTensor()
            axial_tensor = tform(axial_img)
            coronal_tensor = tform(coronal_img)
            sagittal_tensor = tform(sagittal_img)

        # Stack the slices as channels: shape will be [3, H, W]
        stacked_tensor = torch.cat([axial_tensor, coronal_tensor, sagittal_tensor], dim=0)

        return stacked_tensor, label
    
def prepare_data(root_dir):
    def collect_paths_and_labels(directory, label):
        paths = []
        for file in Path(directory).rglob("*.nii"):
            paths.append(str(file))
        return paths, [label] * len(paths), [Path(f).name for f in paths]

    paths, labels, filenames = [], [], []

    structure = {
        'train': [('AD_training', 'AD'), ('CN_training', 'CN')],
        'validation': [('AD_validation', 'AD'), ('CN_validation', 'CN')],
        'test': [('AD_testing', 'AD'), ('CN_testing', 'CN')],
    }

    dataset_split = {'train': [], 'validation': [], 'test': []}

    for split, dirs in structure.items():
        for folder, label in dirs:
            full_dir = os.path.join(root_dir, split, folder)
            p, l, f = collect_paths_and_labels(full_dir, label)
            dataset_split[split].extend(p)
            paths += p
            labels += l
            filenames += f

    label_dict = {f: l for f, l in zip(filenames, labels)}
    return dataset_split, label_dict

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # Change to Normalize with 1 channel mean and std
    transforms.Normalize([0.5], [0.5])
])

def build_model(name):
    if name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[6] = nn.Linear(4096, 2)
    elif name == 'resnet18':
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == 'googlenet':
        model = models.googlenet(pretrained=True, aux_logits=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    return model

def train_model(model, model_name, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3, save_dir='./'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    best_acc = -1
    wait = 0
    train_acc_list, val_acc_list, train_loss_list, val_loss_list = [], [], [], []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs} - Training...")
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            if model_name == 'googlenet':
              outputs = outputs.logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        if total == 0:
            print("No training samples processed; check train_loader.")
            break
        train_acc = 100 * correct / total
        train_acc_list.append(train_acc)
        train_loss_list.append(running_loss / len(train_loader))

        # Validation
        print(f"Epoch {epoch+1}/{epochs} - Validation...")
        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        if total == 0:
            print("No validation samples processed; check val_loader.")
            break
        val_acc = 100 * correct / total
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc
            wait = 0
            path = f"{model_name}_best.pth"
            print(f"DEBUG: Saving model to {path}")
            torch.save(model.state_dict(), path)
            print(f"Saved best model to {path}")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping triggered.")
                break
    return train_acc_list, val_acc_list, train_loss_list, val_loss_list

import matplotlib.pyplot as plt

def plot_training_curves(train_acc_list, val_acc_list, train_loss_list, val_loss_list, model_name):

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_list, label='Train Accuracy', marker='o')
    plt.plot(val_acc_list, label='Validation Accuracy', marker='x')
    plt.title(f'{model_name.upper()} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_list, label='Train Loss', marker='o')
    plt.plot(val_loss_list, label='Validation Loss', marker='x')
    plt.title(f'{model_name.upper()} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save and close plot
    plt.tight_layout()
    plt.savefig(f'{model_name}_training_curves.png')
    plt.show()
    plt.close()

from google.colab import files
uploaded = files.upload()

import zipfile
import os

# Assuming your zip file is called "data.zip"
zip_file_path = '/content/split.zip'  # Uploaded file path
extract_to_path = '/content/extracted_data'

# Check if already extracted, if not, extract
try:
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)
    print("Extraction successful.")
except zipfile.BadZipFile:
    print("Error: Bad ZIP file. Try re-uploading the file.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

import os

root_dir = "/content/extracted_data"

# List everything inside extracted_data
print("Extracted contents:", os.listdir(root_dir))

import os
import glob
root_dir = "/content/extracted_data/split"

#extract_to_path = "/content/extracted_data/test_data"
# Define paths for train, validation, and test datasets
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "validation")
test_dir = os.path.join(root_dir, "test")

# Define paths for AD and CN classes inside each dataset
train_ad_dir = os.path.join(train_dir, "AD_training")
train_cn_dir = os.path.join(train_dir, "CN_training")

val_ad_dir = os.path.join(val_dir, "AD_validation")
val_cn_dir = os.path.join(val_dir, "CN_validation")

test_ad_dir = os.path.join(test_dir, "AD_testing")
test_cn_dir = os.path.join(test_dir, "CN_testing")

import os

# Get all .nii file paths and assign labels
image_paths = []
labels = []
filenames = []  # Store filenames for creating label_dict and feature_dict
from pathlib import Path

train_ad_dir = Path(train_ad_dir)  # Convert to Path object

for file_path in train_ad_dir.rglob("*.nii"):
    image_paths.append(str(file_path))
    labels.append("AD")
    filenames.append(file_path.name)  # Get filename

print(image_paths[:5], labels[:5], filenames[:5])

for file_path in glob.iglob(os.path.join(val_ad_dir, "**", "*.nii"), recursive=True):
    image_paths.append(file_path)
    labels.append("AD")
    filenames.append(os.path.basename(file_path))  # Get filename

for file_path in glob.iglob(os.path.join(test_ad_dir, "**", "*.nii"), recursive=True):
    image_paths.append(file_path)
    labels.append("AD")
    filenames.append(os.path.basename(file_path))  # Get filename

for file_path in glob.glob(os.path.join(train_cn_dir, "*.nii")):
    image_paths.append(file_path)
    labels.append("CN")
    filenames.append(os.path.basename(file_path))  # Get filename

for file_path in glob.glob(os.path.join(val_cn_dir, "*.nii")):
    image_paths.append(file_path)
    labels.append("CN")
    filenames.append(os.path.basename(file_path))  # Get filename

for file_path in glob.glob(os.path.join(test_cn_dir, "*.nii")):
    image_paths.append(file_path)
    labels.append("CN")
    filenames.append(os.path.basename(file_path))  # Get filename

# Create label_dict and feature_dict (replace with your actual feature extraction)
label_dict = {filename: label for filename, label in zip(filenames, labels)}

def main():
    # Root directory containing split data
    root_dir = "/content/extracted_data/split"

    # Prepare image paths and label dictionary
    dataset_split, label_dict = prepare_data(root_dir)

    # Create Datasets
    train_dataset = MRIDataset(dataset_split['train'], label_dict, transform)
    val_dataset   = MRIDataset(dataset_split['validation'], label_dict, transform)
    test_dataset  = MRIDataset(dataset_split['test'], label_dict, transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # Debug: Validation loader info
    print(f"Validation Dataset size: {len(val_dataset)}")
    print(f"Validation batches: {len(val_loader)}")
    if len(val_loader) > 0:
        inputs, labels = next(iter(val_loader))
        print(f"Val batch sample shape: {inputs.shape}, labels: {labels}")
    else:
        print("⚠️ Validation loader is empty. Check data folders or file extensions.")

    # Train and evaluate models
    for model_name in ['alexnet', 'resnet18', 'googlenet']:
        print(f"\n======== {model_name.upper()} ========")
        model = build_model(model_name)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        train_acc, val_acc, train_loss, val_loss = train_model(
            model, model_name, train_loader, val_loader,
            criterion, optimizer, epochs=25, patience=7, save_dir='.'
        )
        plot_training_curves(train_acc, val_acc, train_loss, val_loss, model_name)
        test_model(model, model_name, test_loader)

if __name__ == '__main__':
    main()
