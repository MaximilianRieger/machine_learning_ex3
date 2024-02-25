# script to create a confusion matrix of the trained models

import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
import seaborn as sns
import matplotlib.pyplot as plt
from second_step.dataloading.CelebADataset import CelebADataset
from second_step.dataloading.CIFAR10Dataset import CIFAR10Dataset


from datasets.load_cifar10_samples import load_cifar10_samples
from datasets.load_celebA_samples import load_celebA_samples

from tqdm import tqdm

def dataloader_celebA(img_names, labels, args, transform):
    dataset = CelebADataset(img_names, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False)
    return dataloader

def dataloader_cifar10(img_names, labels, args, transform):
    dataset = CIFAR10Dataset(img_names, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args['batch_size'], shuffle=False)
    return dataloader

def confusion_matrix_plot(y_true, y_pred, title, name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title)
    ax.xaxis.set_tick_params(rotation=90)
    ax.yaxis.set_tick_params(rotation=0)
    plt.show()
    fig.savefig(f"confusion_matrix_{name}.png")

def reshape_tensor(x):
    # reshapes the custom cifar10 tensor to normal image tensor
    # x: tensor of shape (3072,)
    # returns: tensor of shape (3, 32, 32)
    x = torch.tensor(x, dtype=torch.float32)
    return x.view(3, 32, 32)

def main(args):
    if args['dataset'] == 'cifar10':
        _, _, data, labels = load_cifar10_samples()
        transform = transforms.Compose([
            transforms.Lambda(lambda x: reshape_tensor(x)),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # Normalize images
        ])
        dataloader = dataloader_cifar10
    elif args['dataset'] == 'celebA':
        _, _, data, labels = load_celebA_samples()
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize images
        ])
        dataloader = dataloader_celebA
    else:
        raise ValueError("Dataset not supported")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args['model_path'], map_location=device)
    model.to(device)
    model.eval()
    dataloader = dataloader(data, labels, args, transform)
    y_true = []
    y_pred = []
    # make progress bar
    for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluating model'):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = torch.argmax(output, dim=1)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    confusion_matrix_plot(y_true, y_pred, f"Confusion matrix for {args['dataset']}", args['dataset'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create confusion matrix for trained models')
    parser.add_argument('model_path', type=str, help='path to trained model')
    parser.add_argument('dataset', type=str, help='dataset used for training')
    parser.add_argument('-b', '--batch_size', default=256, type=int, help='batch size')
    args = parser.parse_args()
    args = vars(args)
    main(args)