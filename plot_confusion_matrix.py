import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from config import TrainConfig, CnnConfig, VitConfig
from models.cnn import CNN
from models.vit import ViT
from utils.dataset import build_transforms

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

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

def plot_cm(labels, preds, filename, title):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CIFAR10_CLASSES, 
                yticklabels=CIFAR10_CLASSES)
    plt.title(title, fontsize=14)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"����ͼ��: {filename}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('figures', exist_ok=True)
    
    # Dataset
    _, test_transform = build_transforms(augment=False)
    test_dataset = ImageFolder(TrainConfig.test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=TrainConfig.batch_size, 
                             shuffle=False, num_workers=0)
    
    # Models to evaluate
    checkpoints = {
        'cnn_ratio1.0': ('checkpoints/cnn_ratio1.0_best.pth', CNN),
        'vit_ratio1.0': ('checkpoints/vit_ratio1.0_best.pth', ViT)
    }
    
    for name, (ckpt_path, ModelClass) in checkpoints.items():
        if not os.path.exists(ckpt_path):
            print(f"ȱ�� checkpoint �ļ�: {ckpt_path}")
            continue
            
        print(f"�������� {name} ...")
        
        if ModelClass == CNN:
            model = CNN(**CnnConfig.to_cnn_kwargs())
        else:
            model = ViT(**VitConfig.to_vit_kwargs())
            
        try:
             checkpoint = torch.load(ckpt_path, map_location=device)
             model.load_state_dict(checkpoint['model_state_dict'])
        except TypeError:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        
        labels, preds = evaluate_model(model, test_loader, device)
        plot_cm(labels, preds, f'figures/confusion_matrix_{name}.png', f'Confusion Matrix - {name.upper()}')

if __name__ == '__main__':
    main()
