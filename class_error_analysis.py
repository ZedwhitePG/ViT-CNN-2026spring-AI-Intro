import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix

from config import TrainConfig, CnnConfig, VitConfig
from models.cnn import CNN
from models.vit import ViT
from utils.dataset import build_transforms

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

# focus pairs specified in requirements
FOCUS_PAIRS = [('cat', 'dog'), ('automobile', 'truck'), 
               ('deer', 'horse'), ('bird', 'airplane')]

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_labels), np.array(all_preds)

def analyze_errors(labels, preds, model_name, f):
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    f.write(f"\n{'='*40}\n")
    f.write(f"Model: {model_name.upper()}\n")
    f.write(f"{'='*40}\n")
    
    f.write("Per-class Accuracy:\n")
    for idx, acc in enumerate(per_class_acc):
        f.write(f"  {CIFAR10_CLASSES[idx]:<10}: {acc*100:.2f}%\n")
        
    f.write("\nTop Confused Class Pairs:\n")
    confusions = []
    for i in range(10):
        for j in range(10):
            if i != j:
                confusions.append((cm[i, j], CIFAR10_CLASSES[i], CIFAR10_CLASSES[j]))
    
    confusions.sort(reverse=True, key=lambda x: x[0])
    for err_count, c_true, c_pred in confusions[:10]:
        f.write(f"  True: {c_true:<10} Pred: {c_pred:<10} -> {err_count} errors\n")
        
    f.write("\nFocus Pairs Error Analysis (sum of mutual errors):\n")
    for c1, c2 in FOCUS_PAIRS:
        i1, i2 = CIFAR10_CLASSES.index(c1), CIFAR10_CLASSES.index(c2)
        errs = cm[i1, i2] + cm[i2, i1]
        f.write(f"  {c1}/{c2}: {errs} errors\n")
        
    return per_class_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('figures', exist_ok=True)
    
    _, test_transform = build_transforms(augment=False)
    test_dataset = ImageFolder(TrainConfig.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=TrainConfig.batch_size, shuffle=False)
    
    checkpoints = {
        'CNN (100% Data)': ('checkpoints/cnn_ratio1.0_best.pth', CNN),
        'ViT (100% Data)': ('checkpoints/vit_ratio1.0_best.pth', ViT)
    }
    
    per_class_accs = {}
    
    with open('class_error_analysis.txt', 'w') as f:
        f.write("=== ViT & CNN Class Error Analysis ===\n")
        
        for name, (ckpt_path, ModelClass) in checkpoints.items():
            if not os.path.exists(ckpt_path):
                print(f"ȱ��: {ckpt_path}")
                continue
                
            if ModelClass == CNN:
                model = CNN(**CnnConfig.to_cnn_kwargs())
            else:
                model = ViT(**VitConfig.to_vit_kwargs())
                
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            print(f"Analyzing {name}...")
            labels, preds = evaluate_model(model, test_loader, device)
            acc = analyze_errors(labels, preds, name, f)
            per_class_accs[name] = acc

    # Plot Comparison
    if len(per_class_accs) == 2:
        names = list(per_class_accs.keys())
        acc1, acc2 = per_class_accs[names[0]], per_class_accs[names[1]]
        
        x = np.arange(len(CIFAR10_CLASSES))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, acc1 * 100, width, label=names[0], color='skyblue')
        ax.bar(x + width/2, acc2 * 100, width, label=names[1], color='lightcoral')
        
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title('Per-Class Accuracy Comparison (CNN vs ViT)', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(CIFAR10_CLASSES, rotation=45)
        ax.legend(fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig('figures/class_error_comparison.png', dpi=300)
        plt.close()
        print("����ͼ��: figures/class_error_comparison.png")
        print("���ɷ���: class_error_analysis.txt")

if __name__ == '__main__':
    main()
