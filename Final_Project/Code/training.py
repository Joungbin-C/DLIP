# dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

class MagnificationDataset(Dataset):
    def __init__(self, xa_dir, xb_dir, y_dir, transform=None):
        self.xa_files = sorted([os.path.join(xa_dir, f) for f in os.listdir(xa_dir) if f.endswith(".png")])
        self.xb_files = sorted([os.path.join(xb_dir, f) for f in os.listdir(xb_dir) if f.endswith(".png")])
        self.y_files  = sorted([os.path.join(y_dir,  f) for f in os.listdir(y_dir)  if f.endswith(".png")])
        self.transform = transform if transform else T.ToTensor()

    def __len__(self):
        return len(self.xa_files)

    def __getitem__(self, idx):
        xa = Image.open(self.xa_files[idx]).convert('RGB')
        xb = Image.open(self.xb_files[idx]).convert('RGB')
        y  = Image.open(self.y_files[idx]).convert('RGB')

        xa = self.transform(xa)
        xb = self.transform(xb)
        y  = self.transform(y)

        return xa, xb, y


def train(model, xa_dir, xb_dir, y_dir, device, num_epochs, batch_size, save_path):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    dataset = MagnificationDataset(xa_dir, xb_dir, y_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model = model.to(device)

    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for xa, xb, y in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)
            sf = torch.full((xa.size(0),), 20.0, device=device)

            optimizer.zero_grad()
            output = model(xa, xb, sf)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xa.size(0)

        epoch_loss = running_loss / len(dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.6f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved Best Model (Loss: {epoch_loss:.6f})")
