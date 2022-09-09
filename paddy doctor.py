import os
from os import walk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

HP = {
    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 1e-2,
    'momentum': 0.9,
    'test_size': 0.15,
    'seed': 1
}

torch.manual_seed(HP['seed'])
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.backends.cudnn.benchmark = True
print(f'using {device} device')

dataset_dir = '/kaggle/input/paddy-disease-classification/train_images/'
submission_dir = '/kaggle/input/paddy-disease-classification/test_images/'
dataset_file = '/kaggle/input/paddy-disease-classification/train.csv'
submission_sample = '/kaggle/input/paddy-disease-classification/sample_submission.csv'
submission_output = '/kaggle/working/submission.csv'

df = pd.read_csv(dataset_file)
df = shuffle(df, random_state=HP['seed'])

print(f'count: {len(df)} \n')
df.head(5)

df['variety'] = pd.factorize(df['variety'])[0]
df.describe().T

idx_to_label = df['label'].unique()
label_to_idx = {idx: label for label, idx in enumerate(idx_to_label)}
print(label_to_idx)

train_df, test_df = train_test_split(df, test_size=HP['test_size'])
print(f'train len: {len(train_df)}, test len: {len(test_df)}')

train_df['label'].value_counts()

train_transform = transforms.Compose([
    transforms.RandomRotation(45),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomChoice([
        transforms.Pad(padding=10),
        transforms.CenterCrop(480),
        transforms.CenterCrop((576,432)),
    ]),
    transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0.1),
    transforms.RandomGrayscale(p=0.025),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
class PaddyDataset(Dataset):
    def __init__(self, dataset_dir, df, label_to_idx, transforms):
        self.df = df
        self.label_to_idx = label_to_idx
        self.transforms = transforms
        self.df['path'] = dataset_dir + '/' + self.df.label + '/' + self.df.image_id
        # 0: image_id, 1: label, 2: variety, 3: age, 4: path
        self.df = self.df.values.tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df[idx]
        image = Image.open(row[4])
        image = self.transforms(image)
        idx = self.label_to_idx[row[1]]
        return image, idx

train_dataset = PaddyDataset(dataset_dir, train_df, label_to_idx, train_transform)
test_dataset = PaddyDataset(dataset_dir, test_df, label_to_idx, test_transform)
train_dataloader = DataLoader(train_dataset, batch_size=HP['batch_size'], shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=HP['batch_size'], shuffle=True, pin_memory=True)

model = models.resnet50(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.1),
    nn.Linear(model.fc.in_features, len(label_to_idx))
)
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=HP['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=8,gamma=0.1)

def train(model, criterion, optimizer, train_dataloader, test_dataloader):
    total_train_loss = 0
    total_test_loss = 0

    model.train()
    with tqdm(train_dataloader, unit='batch', leave=False) as pbar:
        pbar.set_description(f'training')
        for images, idxs in pbar:
            images = images.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)
            output = model(images)

            loss = criterion(output, idxs)
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    model.eval()
    with tqdm(test_dataloader, unit='batch', leave=False) as pbar:
        pbar.set_description(f'testing')
        for images, idxs in pbar:
            images = images.to(device, non_blocking=True)
            idxs = idxs.to(device, non_blocking=True)

            output = model(images)
            loss = criterion(output, idxs)
            total_test_loss += loss.item()

    train_acc = total_train_loss / len(train_dataset)
    test_acc = total_test_loss / len(test_dataset)
    print(f'Train loss: {train_acc:.4f} Test loss: {test_acc:.4f} ')

%%time
for i in range(HP['epochs']):
    print(f"Epoch {i+1}/{HP['epochs']}")
    train(model, criterion, optimizer, train_dataloader, test_dataloader)

%%time
model.eval()
image_ids, labels = [], []
for (dirpath, dirname, filenames) in walk(submission_dir):
    for filename in filenames:
        image = Image.open(dirpath+filename)
        image = test_transform(image)
        image = image.unsqueeze(0).to(device)
        image_ids.append(filename)
        labels.append(idx_to_label[model(image).argmax().item()])
        
submission = pd.DataFrame({
    'image_id': image_ids,
    'label': labels,
})