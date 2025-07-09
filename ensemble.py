import os
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# ------------------ Model Definitions ------------------ #
class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate, bn_size=4, drop_rate=0.):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, bn_size*growth_rate, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size*growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size*growth_rate, growth_rate, 3, padding=1, bias=False)
        self.drop_rate = drop_rate
    def forward(self, x):
        out = self.conv1(self.relu1(self.bn1(x)))
        out = self.conv2(self.relu2(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_ch, growth_rate):
        super().__init__()
        layers, ch = [], in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.layers = nn.Sequential(*layers)
        self.out_ch = ch
    def forward(self, x):
        return self.layers(x)

class Transition(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2, 2)
        )
    def forward(self, x):
        return self.net(x)

class DenseCNN(nn.Module):
    def __init__(self, config, growth=12, init_feat=64, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, init_feat, 7, 2, 3, bias=False),
            nn.BatchNorm2d(init_feat),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )
        ch = init_feat
        self.blocks = nn.ModuleList()
        self.trans = nn.ModuleList()
        for i, n in enumerate(config):
            blk = DenseBlock(n, ch, growth)
            self.blocks.append(blk)
            ch = blk.out_ch
            if i < len(config) - 1:
                t_out = ch // 2
                self.trans.append(Transition(ch, t_out))
                ch = t_out
        self.bn_final = nn.BatchNorm2d(ch)
        self.classifier = nn.Linear(ch, num_classes)
    def forward(self, x):
        out = self.features(x)
        for i, blk in enumerate(self.blocks):
            out = blk(out)
            if i < len(self.trans):
                out = self.trans[i](out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, (1,1)).flatten(1)
        return self.classifier(out)

class EnsembleModel(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.models = nn.ModuleList([DenseCNN(cfg) for cfg in configs])
    def forward(self, x):
        logits = torch.stack([m(x) for m in self.models])
        return logits.mean(0)

# ------------------ Data Handling with 3-Plane Extraction ------------------ #
class MRIDataset(Dataset):
    def __init__(self, file_list, labels, device='cpu'):
        self.files = file_list
        self.labels = labels
        self.device = device
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        vol = nib.load(self.files[idx]).get_fdata().astype(np.float32)
        vol = (vol - vol.mean()) / (vol.std() + 1e-5)
        x_c, y_c, z_c = [d // 2 for d in vol.shape]
        axial   = vol[:, :, z_c]
        coronal = vol[:, y_c, :]
        sagittal= vol[x_c, :, :]
        slices = [axial, coronal, sagittal]
        out = []
        for sl in slices:
            t = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(112,112), mode='bilinear', align_corners=False)
            out.append(t.squeeze(0))
        x = torch.cat(out, dim=0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x.to(self.device), y.to(self.device)

def load_dataset(root):
    splits = {
        'train': ['AD_training', 'CN_training'],
        'validation': ['AD_validation', 'CN_validation'],
        'test': ['AD_testing', 'CN_testing']
    }
    data = {}
    for sp, folders in splits.items():
        files, labels = [], []
        for i, cls in enumerate(folders):
            cls_path = os.path.join(root, sp, cls)
            for f in os.listdir(cls_path):
                if f.endswith('.nii'):
                    files.append(os.path.join(cls_path, f))
                    labels.append(i)
        data[sp] = (files, labels)
    return data

from tqdm import tqdm

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    loop = tqdm(loader, desc="Training", leave=False)
    for x, y in loop:
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    preds, trues = [], []
    loop = tqdm(loader, desc="Evaluating", leave=False)
    for x, y in loop:
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds.append(probs)
        trues.append(y.cpu().numpy())
    y_true = np.concatenate(trues)
    y_score = np.concatenate(preds)
    auc = roc_auc_score(y_true, y_score)
    return total_loss / len(loader.dataset), correct / len(loader.dataset), auc, y_true, y_score

# 1. Upload your ZIP file (e.g. gm.zip or test_data.zip)
from google.colab import files
import zipfile, os, shutil

print("Please select and upload your dataset ZIP:")
uploaded = files.upload()  # choose your ZIP here

# 2. Identify uploaded filename and desired output folder
zip_fname = list(uploaded.keys())[0]                # e.g. 'test_data.zip'
output_dir = os.path.splitext(zip_fname)[0]        # e.g. 'test_data'

# 3. Create the output folder (if not exists) and extract
os.makedirs(output_dir, exist_ok=True)
with zipfile.ZipFile(zip_fname, 'r') as zf:
    # This will extract into a temp folder if the ZIP has an internal top‑level dir
    zf.extractall(output_dir)

# 4. Flatten nested folder if necessary
nested = os.path.join(output_dir, os.path.basename(output_dir))
if os.path.isdir(nested):
    for item in os.listdir(nested):
        src = os.path.join(nested, item)
        dst = os.path.join(output_dir, item)
        shutil.move(src, dst)
    os.rmdir(nested)

print(f"Contents of '{output_dir}/':", os.listdir(output_dir))

# 5. (Optional) Verify train/validation/test splits
for split in ['train','validation','test']:
    path = os.path.join(output_dir, split)
    if os.path.isdir(path):
        classes = os.listdir(path)
        print(f" ↳ {split}/ found classes: {classes}")
    else:
        print(f" ⚠️  Missing expected folder: {path}")

if __name__ == '__main__':
    data = load_dataset('split')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = DataLoader(MRIDataset(*data['train'], device), batch_size=16, shuffle=True)
    val_loader   = DataLoader(MRIDataset(*data['validation'], device), batch_size=8)
    test_loader  = DataLoader(MRIDataset(*data['test'], device), batch_size=8)

    configs = [[6,12,24,16], [6,12,32,32], [6,12,36,24]]
    model = EnsembleModel(configs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0001, weight_decay=0.06
    )

    best_loss, patience, counter = float('inf'), 7, 0
    train_history, val_history = [], []
    import os

    # Before training loop
    save_path = 'exp1/best.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(30):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_auc, _, _ = eval_epoch(model, val_loader, criterion, device)
        train_history.append((tr_loss, tr_acc))
        val_history.append((val_loss, val_acc))
        print(f"Epoch {epoch}: train_loss={tr_loss:.3f}, train_acc={tr_acc:.3f} | val_loss={val_loss:.3f}, val_acc={val_acc:.3f}, val_auc={val_auc:.3f}")
        if val_loss < best_loss:
            best_loss, counter = val_loss, 0
            torch.save(model.state_dict(), save_path)
        else:
            counter += 1
            if counter >= patience:
                print('Early stopping at epoch', epoch)
                break

    # Plot training curves
    epochs = range(len(train_history))
    plt.figure(); plt.plot(epochs, [t[0] for t in train_history], label='Train Loss')
    plt.plot(epochs, [v[0] for v in val_history], label='Val Loss'); plt.legend(); plt.show()
    plt.figure(); plt.plot(epochs, [t[1] for t in train_history], label='Train Acc')
    plt.plot(epochs, [v[1] for v in val_history], label='Val Acc'); plt.legend(); plt.show()

    # Final test evaluation
    model.load_state_dict(torch.load(save_path))
    test_loss, test_acc, test_auc, y_true, y_score = eval_epoch(model, test_loader, criterion, device)
    print(f"Test Loss={test_loss:.3f}, Test Acc={test_acc:.3f}, Test AUC={test_auc:.3f}")
    preds = (y_score > 0.5).astype(int)
    cm = confusion_matrix(y_true, preds)
    print("Confusion Matrix:\n", cm)
    plt.figure(); plt.imshow(cm); plt.colorbar(); plt.title('Confusion Matrix'); plt.show()
