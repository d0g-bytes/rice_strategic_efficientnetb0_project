import os, random, time, numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent.absolute()
DATA_PATH = SCRIPT_DIR / "Dhan-Shomadhan"

# Load images into a local array to optimize training speed
def load_images():
    typo_fixes = {"Browon Spot": "Brown_Spot", "Brown Spot": "Brown_Spot", 
                "Leaf Scaled": "Leaf_Scalded", "Rice Blast": "Rice_Blast", 
                "Rice Turgro": "Rice_Tungro", "Rice Tungro": "Rice_Tungro", 
                "Sheath Blight": "Sheath_Blight", "Shath Blight": "Sheath_Blight"}
    
    data = []
    class_names = sorted(list(set(typo_fixes.values())))
    c_to_idx = {name: i for i, name in enumerate(class_names)}

    all_files = []
    for root, _, files in os.walk(DATA_PATH):
        folder = os.path.basename(root)
        if folder in typo_fixes:
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    all_files.append((os.path.join(root, f), folder, root))

    for img_path, folder_name, full_root in tqdm(all_files, desc="Loading..."):
        try:
            bg = 0 if "White" in full_root else 1 
            label = c_to_idx[typo_fixes[folder_name]]
            img = Image.open(img_path).convert('RGB').resize((224,224))
            data.append({'img': img, 'label': label, 'bg': bg})
        except: 
            continue
    return data, class_names

# Confusion Matrix Plotting Utility
def plot_cm(y_true, y_pred, classes, run_idx):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation=45)
    plt.title(f"Confusion Matrix - Run {run_idx+1}")
    plt.tight_layout()
    plt.show()

    print(f"\nClass ACC (Run {run_idx+1})")
    cm_acc = cm.diagonal() / cm.sum(axis=1)
    for name, acc in zip(classes, cm_acc):
        print(f"{name:<15}: {acc*100:.2f}%")
    print("-" * 35)

class RiceDataset(Dataset):
    def __init__(self, items, is_train=False):
        self.items = items
        if is_train:
            self.tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(20), 
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        return self.tf(self.items[i]['img']), self.items[i]['label'], self.items[i]['bg']

def train_and_eval(all_data, class_names):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels_for_split = [x['label'] * 2 + x['bg'] for x in all_data]
    results = {'white': [], 'field': [], 'combined': []}

    for run in range(5):
        print(f"\n> RUN {run+1}/5")
        train_items, test_items = train_test_split(all_data, test_size=0.2, random_state=run, stratify=labels_for_split)
        
        train_loader = DataLoader(RiceDataset(train_items, True), batch_size=16, shuffle=True)
        test_loader = DataLoader(RiceDataset(test_items, False), batch_size=16)

        # 1. Load EfficientNet-B0 Base Model
        try:
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        except:
            model = models.efficientnet_b0(pretrained=True)

        # 2. Freeze the ENTIRE base model initially (Transfer Learning setup)
        for param in model.parameters():
            param.requires_grad = False

        # 3. Replace Classifier Head
        n_inputs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(n_inputs, len(class_names))
        )
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        # === Train classifier head ===
        print("   -> Stage 1: Training classifier head (frozen base)...")
        optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)
        
        for epoch in range(5): # Initial short training for head convergence
            model.train()
            pbar = tqdm(train_loader, desc=f"      Ep {epoch+1}/5", leave=False)
            for imgs, labels_batch, _ in pbar:
                imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels_batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pbar.set_postfix(loss=f"{loss.item():.3f}")

        # === Strategic Fine-Tuning ===
        # --- STAGE 2: Strategic Fine-Tuning ---
        print("   -> Stage 2: Strategic Fine-Tuning (Unfreezing final blocks & classifier for 5 epochs)...")
        
        # 1. Unfreeze the last 2 main blocks of model.features (Blocks 7 and 8) 
        # and the final classifier head. This represents the final high-level layers.
        for param in model.features[5:].parameters():
            param.requires_grad = True
            
        for param in model.classifier.parameters():
            param.requires_grad = True

        # 2. Re-initialize optimizer with the paper's reduced learning rate (1e-4)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
        
        FINE_TUNING_EPOCHS = 5  # Exactly 5 epochs as specified in the paper
        for epoch in range(FINE_TUNING_EPOCHS):
            model.train()
            running_corr = 0
            pbar = tqdm(train_loader, desc=f"      Fine-Tune Ep {epoch+1}/5", leave=False)
            
            for imgs, labels_batch, _ in pbar:
                imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels_batch)
                    _, preds = torch.max(outputs, 1)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_corr += torch.sum(preds == labels_batch.data)
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{(running_corr.double()/((pbar.n+1)*16))*100:.1f}%")

        # --- Evaluation Phase ---
        model.eval()
        stats = {0: {'c': 0, 't': 0}, 1: {'c': 0, 't': 0}}
        all_preds, all_true = [], []

        with torch.no_grad():
            for imgs, labels_batch, bgs in test_loader:
                imgs, labels_batch = imgs.to(device), labels_batch.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(labels_batch.cpu().numpy())
                for p, l, b in zip(preds, labels_batch, bgs):
                    stats[b.item()]['t'] += 1
                    if p == l: stats[b.item()]['c'] += 1

        acc_w = (stats[0]['c']/stats[0]['t']*100) if stats[0]['t'] > 0 else 0
        acc_f = (stats[1]['c']/stats[1]['t']*100) if stats[1]['t'] > 0 else 0
        acc_c = ((stats[0]['c'] + stats[1]['c']) / (stats[0]['t'] + stats[1]['t'])) * 100
        
        results['white'].append(acc_w)
        results['field'].append(acc_f)
        results['combined'].append(acc_c)
        print(f"* Run {run+1} Results -> White: {acc_w:.1f}% | Field: {acc_f:.1f}% | Both: {acc_c:.1f}%")

        plot_cm(all_true, all_preds, class_names, run)
        del model, optimizer
        torch.cuda.empty_cache()

    print("\n" + "="*55 + "\n=== FINAL RESULTS SUMMARY ===\n" + "="*55)
    print(f"{'Scenario':<25} | {'Mean Acc (%)':<15} | {'Std Dev':<10}")
    print("-" * 55)
    names = {'white': "White background", 'field': "Field background", 'combined': "Either white or field background"}
    for k in ['white', 'field', 'combined']:
        print(f"{names[k]:<25} | {np.mean(results[k]):<15.2f} | {np.std(results[k]):<10.2f}")
    print("="*55)

if __name__ == '__main__':
    all_data, class_names = load_images()
    train_and_eval(all_data, class_names)