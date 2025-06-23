# ==== Import Libraries ====
import os                                      # For directory and file path handling
from PIL import Image                         # For image loading and format conversion
from torch.utils.data import Dataset          # Base class for PyTorch datasets
import torchvision.transforms as T            # For image preprocessing (resize, tensor conversion, etc.)
import torch                                   # PyTorch core library
from torch.utils.data import DataLoader       # For batch loading data
import torch.nn as nn                         # For loss functions and model components
from tqdm import tqdm                         # Progress bar for training loop


class MagnificationDataset(Dataset):
    def __init__(self, xa_dir, xb_dir, y_dir, transform=None):
        # Get sorted list of all file paths for Xa, Xb, and Y
        self.xa_files = sorted([os.path.join(xa_dir, f) for f in os.listdir(xa_dir) if f.endswith(".png")])
        self.xb_files = sorted([os.path.join(xb_dir, f) for f in os.listdir(xb_dir) if f.endswith(".png")])
        self.y_files = sorted([os.path.join(y_dir, f) for f in os.listdir(y_dir) if f.endswith(".png")])

        # If transform is not provided, default to converting to tensor only
        self.transform = transform if transform else T.ToTensor()

    def __len__(self):
        return len(self.xa_files)  # Number of samples in dataset (assumed equal for all three folders)

    def __getitem__(self, idx):
        # Load images and convert them to RGB format
        xa = Image.open(self.xa_files[idx]).convert('RGB')
        xb = Image.open(self.xb_files[idx]).convert('RGB')
        y = Image.open(self.y_files[idx]).convert('RGB')

        # Apply transform (e.g., Resize + ToTensor)
        xa = self.transform(xa)
        xb = self.transform(xb)
        y = self.transform(y)

        # Return 3-tuple (input A, input B, target output)
        return xa, xb, y


def train(model, xa_dir, xb_dir, y_dir, device, num_epochs, batch_size, save_path):
    # Define preprocessing transform (resize images and convert to tensor)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])

    # Initialize dataset and data loader
    dataset = MagnificationDataset(xa_dir, xb_dir, y_dir, transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()                                       # Use Mean Squared Error loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)     # Adam optimizer
    model = model.to(device)                                      # Move model to GPU or CPU

    best_loss = float('inf')                                      # Initialize best loss for checkpointing

    # === Training Loop ===
    for epoch in range(num_epochs):
        model.train()                                             # Set model to training mode
        running_loss = 0.0                                        # Track total loss for the epoch

        # === Iterate over training batches ===
        for xa, xb, y in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            xa, xb, y = xa.to(device), xb.to(device), y.to(device)    # Move data to device
            sf = torch.full((xa.size(0),), 20.0, device=device)       # Create a magnification factor tensor [B]

            optimizer.zero_grad()                                 # Reset gradients
            output = model(xa, xb, sf)                            # Forward pass through model
            loss = criterion(output, y)                           # Compute loss
            loss.backward()                                       # Backpropagation
            optimizer.step()                                      # Update model weights

            running_loss += loss.item() * xa.size(0)              # Accumulate batch loss (scaled by batch size)

        epoch_loss = running_loss / len(dataset)                  # Average loss for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.6f}")

        # === Save the model if it has improved ===
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)            # Save best model weights
            print(f"✅ Saved Best Model (Loss: {epoch_loss:.6f})")
