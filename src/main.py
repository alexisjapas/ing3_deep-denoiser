import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from DenoiserDataset import DenoiserDataset
from VDSR import VDSR
from train import train


#### MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
model = VDSR(18).to(device)
print(model)


#### TRAINING
# Dataset
train_data_path = "/home/obergam/Data/flir/images_thermal_train/"
crop_size = 82
noise_density = 0.5
train_dataset = DenoiserDataset(train_data_path, crop_size, noise_density)

# Dataloader
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8, shuffle=True)
for inputs, targets in train_dataloader: # DEBUG
    print(inputs.shape, targets.shape)
    break

# Train
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 100
for e in range(epochs):
    print(f"Epoch {e+1}")
    epoch_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    print(f"Epoch loss: {epoch_loss}")
    print("-----------------------------")
print("Training finished")


#### SAVING MODEL
torch.save(model, 'model.pth')


#### TESTING

