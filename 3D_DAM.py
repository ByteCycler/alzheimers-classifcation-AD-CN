import torch
import numpy as np
from torch import nn
# from lib.model.attention_block import SpatialAttention3D, ChannelAttention3D, residual_block
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader  # Import Dataset here
import os
import glob
import nibabel as nib

# Define the paths to your folders (replace with actual paths)
from google.colab import files
uploaded = files.upload()

import zipfile
import os

# Assuming your zip file is called "data.zip"
zip_file_path = '/content/split.zip'  # Uploaded file path
extract_to_path = '/content/split'

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

root_dir = "/content/split"

# List everything inside extracted_data
print("Extracted contents:", os.listdir(root_dir))

#Preprocessing function
def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)

    minimum = np.min(image)
    maximum = np.max(image)

    if maximum > minimum:
        ret = (image - minimum) / (maximum - minimum)
    else:
        ret = image * 0.
    return ret

import nibabel as nib
import scipy.ndimage

def reshape_zero_padding(nii_path, target_shape=128):
    """Resizes a 3D NIfTI image while maintaining aspect ratio, then zero-pads to target size."""

    # Load NIfTI image
    img_nifti = nib.load(nii_path)
    img_data = img_nifti.get_fdata()  # Convert to numpy array

    # Get original shape
    orig_shape = img_data.shape

    # Compute scaling factor to fit the target size while maintaining aspect ratio
    scale = target_shape / max(orig_shape)  # Scale based on the largest dimension
    new_size = tuple([int(dim * scale) for dim in orig_shape])

    # Resize using trilinear interpolation
    img_resized = scipy.ndimage.zoom(img_data, (new_size[0] / orig_shape[0],
                                                 new_size[1] / orig_shape[1],
                                                 new_size[2] / orig_shape[2]),
                                     order=1)  # Order=1 → Trilinear interpolation

    # Compute padding for each axis
    pad_s = (target_shape - img_resized.shape[0]) // 2
    pad_c = (target_shape - img_resized.shape[1]) // 2
    pad_a = (target_shape - img_resized.shape[2]) // 2

    pad_s_extra = target_shape - img_resized.shape[0] - pad_s
    pad_c_extra = target_shape - img_resized.shape[1] - pad_c
    pad_a_extra = target_shape - img_resized.shape[2] - pad_a

    # Apply zero-padding to center the image
    img_padded = np.pad(img_resized, ((pad_s, pad_s_extra),
                                      (pad_c, pad_c_extra),
                                      (pad_a, pad_a_extra)),
                        mode='constant', constant_values=0)
    return img_padded

root_dir = "/content/split"
import os

#extract_to_path = "/content/extracted_data/split"
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
feature_dict = {filename: [] for filename in filenames}  # Replace [] with actual features

from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, image_paths, label_dict, feature_dict, task='AD_CN'):
        self.image_paths = image_paths
        self.label_dict = label_dict
        self.feature_dict = feature_dict

        if task == 'AD_CN':
            classes = ['AD', 'CN']

        self.idx_to_class = {i: j for i, j in enumerate(classes)}
        self.class_to_idx = {value: key for key, value in self.idx_to_class.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image_id = image_filepath.split('/')[-1]
        label = self.label_dict[image_id]
        label = self.class_to_idx[label]
        feature = self.feature_dict[image_id]  # If not needed, remove

        image_original = nib.load(image_filepath).get_fdata()
        image = normalise_zero_one(image_original)
        image = reshape_zero_padding(image_filepath)
        image = np.expand_dims(image, axis=0)

        # Robustly convert to PyTorch tensor
        try:
            # Attempt direct conversion with float32 dtype
            image = torch.tensor(image, dtype=torch.float32)
        except TypeError:
            try:
                # If direct conversion fails, try astype first
                image = torch.tensor(image.astype(np.float32), dtype=torch.float32)
            except (TypeError, ValueError):
                # If astype fails, further investigation or modification might be needed
                print(f"Error: Could not convert image to tensor. Filepath: {image_filepath}")
                print(f"Image shape: {image.shape}, dtype: {image.dtype}")
                # Potentially add more error handling or debugging here
                # e.g., inspect values within the array: print(np.unique(image))
                #       or save the problematic image for analysis
                raise  # Re-raise the exception to stop the process

        return image, label

# Create dataset
train_image_paths = glob.glob(os.path.join(root_dir, "train", "AD_training", "*.nii")) + \
                    glob.glob(os.path.join(root_dir, "train", "CN_training", "*.nii"))
train_dataset = MRIDataset(image_paths=train_image_paths, label_dict=label_dict, feature_dict=feature_dict)
val_image_paths = glob.glob(os.path.join(root_dir, "validation", "AD_validation", "*.nii")) + \
                    glob.glob(os.path.join(root_dir, "validation", "CN_validation", "*.nii"))
val_dataset = MRIDataset(image_paths=val_image_paths, label_dict=label_dict, feature_dict=feature_dict)
testimage_paths = glob.glob(os.path.join(root_dir, "test", "AD_testing", "*.nii")) + \
                    glob.glob(os.path.join(root_dir, "test", "CN_testing", "*.nii"))
splitset = MRIDataset(image_paths=testimage_paths, label_dict=label_dict, feature_dict=feature_dict)

# Create DataLoader, should be 3 lines each for train test and validate
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(splitset, batch_size=1, shuffle=False)

class ChannelAttention3D(nn.Module):
    def __init__(self, in_planes=64, ratio=8):
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

#uses tensor of input image
    def forward(self, x):
        residual = x
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * residual


class SpatialAttention3D(nn.Module):
    def __init__(self, out_channel=64, kernel_size=3, stride=1, padding=1):
        super(SpatialAttention3D, self).__init__()

        self.conv = nn.Conv3d(2, out_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        x = self.sigmoid(x)
        out = x * residual
        return out


class residual_block(nn.Module):
    def __init__(self, channel_size=64):
        super(residual_block, self).__init__()

        self.conv = nn.Conv3d(channel_size, channel_size, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(channel_size)

    def forward(self, x):
        residual = x
        y = self.conv(x)
        y = self.bn(y)
        y = self.relu(y)
        out = y + residual
        return out

import numpy as np
import torch
from torch import nn

class DAM(nn.Module):
    def __init__(self, channels=64):
        super(DAM, self).__init__()

        self.sa = SpatialAttention3D(out_channel=channels)
        self.ca = ChannelAttention3D(in_planes=channels)

    def forward(self, x):
        residual = x
        out = self.ca(x)
        out = self.sa(out)
        out = out + residual
        return out

class Duo_Attention(nn.Module):
    def __init__(
            self, input_size=(1, 128, 128, 128), num_classes=2, dropout=0
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(input_size[0], 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            nn.Conv3d(8, 16, 3, padding=1, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            residual_block(channel_size=16),
            nn.MaxPool3d(2, 2),

            nn.Conv3d(16, 32, 3, padding=1, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            residual_block(channel_size=32),
            DAM(channels=32),
            nn.MaxPool3d(2, 2),

            nn.Conv3d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            residual_block(channel_size=64),
            nn.MaxPool3d(2, 2),
            DAM(channels=64),

            nn.AvgPool3d(1, stride=1), # Changed from kernel_size=3 to kernel_size=1, can lead to overfitting
        )

        input_tensor = torch.zeros(input_size).unsqueeze(0)
        output_convolutions = self.conv(input_tensor)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(np.prod(list(output_convolutions.shape)).item(), 1024),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.fc(y)
        return y

import os
import numpy as np
import torch

#Caution
model_dict = {
    'DuoAttention':Duo_Attention,
}

def create_model(
        model_name: str,
        num_classes: int,
        pretrained_path: str = None,
        **kwargs,
):

#creates an instance of the class in the model
    model = model_dict[model_name](
        num_classes=num_classes,
        **kwargs,
    )

#loading weights from a pretrained model
    if pretrained_path is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        print('Load pretrained...')
        model.module.load_state_dict(
            torch.load(
                pretrained_path,
                map_location=str(device))
        )

    return model

import argparse
import ast

#average for any metric we use
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)

import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=1, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.

                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train_one_epoch(
        model,
        loader,
        optimizer,
        epoch_idx: int,
        lr_scheduler=None,
):
    losses_m = AverageMeter()
    acc_m = AverageMeter()

    model.train()
    print('Start training epoch: ', epoch_idx)
    for batch_idx, data in enumerate(tqdm(loader)):

        images, target = data
        images, target = images.to(device), target.to(device)
        target = target.flatten()

        output = model(images)

        loss = nn.CrossEntropyLoss()(output, target)

        losses_m.update(loss.item(), images.size(0))
        acc1 = accuracy(output, target, topk=(1,))
        acc_m.update(acc1[0].item(), output.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #torch.cuda.synchronize()

    print(optimizer.param_groups[0]['lr'])

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    metrics = OrderedDict([('loss', losses_m.avg), ('Acc', acc_m.avg)])
    if lr_scheduler is not None:
        lr_scheduler.step()

    return metrics

#laoader loads only validation dataset
def validate(model, loader):
    losses_m = AverageMeter()
    acc_m = AverageMeter()

    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            images, target = data
            images, target = images.to(device), target.to(device)
            target = target.flatten()

            output = model(images)

            loss = nn.CrossEntropyLoss()(output, target)
            acc1 = accuracy(output, target, topk=(1,))
            # reduced_loss = loss.data

            torch.cuda.synchronize()

            losses_m.update(loss.item(), images.size(0))
            acc_m.update(acc1[0].item(), output.size(0))

    metrics = OrderedDict([('loss', losses_m.avg), ('Acc', acc_m.avg)])

    return metrics

import matplotlib.pyplot as plt

def train(model,
          train_loader,
          val_loader,
          epoch_size=25,
          lr_scheduler=True,
          learning_rate=1e-4, optimizer_setup='Adam', w_decay=1e-5,
          patience=5, save_last=True,
          name='save',
          ):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    print('Training using:', device)
    # model = torch.nn.DataParallel(model)
    model.to(device)

    if optimizer_setup == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=w_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=w_decay)

    min_valid_loss = np.inf
    max_acc = 0
    highest_val_epoch = 0
    train_acc, train_losses, val_acc, val_losses = [], [], [], []

    if lr_scheduler:
        print('Applied lr_scheduler')
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    else:
        scheduler = None

    early_stopping = EarlyStopping(patience=patience, verbose=True)

    print('Start Training Process:...')

    for epoch in range(epoch_size):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            epoch_idx=epoch + 1,
            lr_scheduler=scheduler,
        )

        eval_metrics = validate(model, val_loader)

        train_acc.append(train_metrics['Acc'])
        train_losses.append(train_metrics['loss'])
        val_acc.append(eval_metrics['Acc'])
        val_losses.append(eval_metrics['loss'])

        if save_last:
            torch.save(model.state_dict(), '/content/split/best.pth')

        print(f'Epoch {epoch + 1}:  Train: {train_metrics} ----- Val: {eval_metrics}')

        if min_valid_loss > eval_metrics['loss']:
            print(f'Validation Loss Decreased. \t Saving The Model')
            min_valid_loss = eval_metrics['loss']
            torch.save(model.state_dict(), '/content/split/best.pth')

        if max_acc < eval_metrics['Acc']:
            print(f'Validation Acc Increased. \t Saving The Model')
            max_acc = eval_metrics['Acc']
            highest_val_epoch = epoch + 1
            torch.save(model.state_dict(), '/content/split/best.pth')

        early_stopping(eval_metrics['loss'], model)
        if early_stopping.early_stop:
            print(f'Early stopping at: {epoch + 1 - patience}')
            print(f'Highest validation accuracy: {max_acc:.4f} at epoch {highest_val_epoch}')
            break

    # Plotting at the end
    def plot_result(filename, val_data, train_data, ylabel):
        plt.figure()
        plt.plot(train_data, label='Train')
        plt.plot(val_data, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel(ylabel)
        plt.legend()
        plt.title(f'{ylabel} Over Epochs')
        plt.grid(True)
        plt.savefig(f'/content/split/{filename}.png')
        plt.show()

    plot_result('Loss', val_losses, train_losses, 'Loss')
    plot_result('Accuracy', val_acc, train_acc, 'Accuracy')

def test(
        model,
        test_loader,
        output_size
):
    y_pred = []
    y_true = []
    prob = []  # Initialize as an empty list to store probabilities

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.eval()
    with torch.no_grad():
        print('Start Testing:...')
        for batch_idx, data in enumerate(test_loader):
            images, target = data
            images, target = images.to(device), target.to(device)
            target = target.flatten()

            output = model(images)

            # Store probabilities (assuming output is a tensor of probabilities)
            prob.append(output.cpu())  # Move to CPU before appending

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)
            labels = target.data.cpu().numpy()
            y_true.extend(labels)  # Save Truth

    conf_mat = confusion_matrix(y_true, y_pred)
    y_true_1 = torch.LongTensor(y_true)
    y_true_2 = F.one_hot(y_true_1, num_classes=output_size)

    # Concatenate the list of probability tensors into a single tensor
    prob_1 = torch.cat(prob, dim=0).float()  # Concatenate and convert to float

    print('Testing has finished.')
    return prob_1, y_true_2, conf_mat

from torch.utils.data import DataLoader
import argparse
import torch
import numpy as np

class Args:
    experiment_name = 'AD_CN'
    task = 'AD_CN'
    fold = 0
    train_type = 'image_level'
    output_size = 2
    learning_rate = 0.0001
    w_decay = 1e-5
    batch_size = 8
    patch_size = 32
    epoch_size = 25
    drop_out = 0.5
    patience = 7
    image_folder = '/content/split'  # Update for Colab path
    train_path = None  # Set specific path if needed
    val_path = None  # Set specific path if needed
    model_name = 'SEModule'
    model_kwargs = {}  # Define model kwargs if necessary

args = Args()  # Create an instance so you can access values as args.learning_rate, etc.

if __name__ == '__main__':
    torch.cuda.empty_cache()
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not all(name in globals() for name in ['train_dataset', 'val_dataset', 'splitset']):
    raise NameError("train_dataset, val_dataset, or splitset are not defined. Please ensure the code to create them has been executed.")

 # Model configuration
model = create_model(
    model_name='DuoAttention',
    num_classes=Args.output_size,
    **Args.model_kwargs,
)

train(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epoch_size=Args.epoch_size,
    lr_scheduler=True,
    learning_rate=Args.learning_rate,
    optimizer_setup='Adam',
    w_decay=Args.w_decay,
    patience=Args.patience,
    save_last=True,
    name=Args.experiment_name,
)

prob_1, y_true_2, conf_mat = test(model, test_loader, Args.output_size)

# Assuming y_true_2 are the ground truth labels and prob_1 are the predicted probabilities:
# You need to calculate loss and accuracy based on these values

# Calculate loss (example with CrossEntropyLoss)
loss = nn.CrossEntropyLoss()(prob_1, y_true_2.argmax(dim=1))  # Assuming one-hot encoded labels

# Calculate accuracy (example with accuracy function)
acc = accuracy(prob_1, y_true_2.argmax(dim=1), topk=(1,))[0]

print(f"Test Set: Loss = {loss.item():.4f}, Accuracy = {acc.item():.4f}")

