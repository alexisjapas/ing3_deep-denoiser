import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from DenoiserDataset import DenoiserDataset
from VDSR import VDSR
from train import train
from infer import infer


#### MODEL
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
model = VDSR(18).to(device)
print(model)


#### INFERING
# Dataset
infer_data_path = "/home/obergam/Data/flir/images_thermal_val/"
infer_dataset = DenoiserDataset(infer_data_path, 0, 0.1)

# Dataloader
infer_dataloader = DataLoader(infer_dataset, batch_size=1, num_workers=1)

# Infer
#infer(infer_dataloader, model, device)


#### TRAINING
# Dataset
train_data_path = "/home/obergam/Data/flir/images_thermal_train/"
crop_size = 82
noise_density = 0.1
train_dataset = DenoiserDataset(train_data_path, crop_size, noise_density)

# Dataloader
batch_size = 64
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
for inputs, targets in train_dataloader: # DEBUG
    print(inputs.shape, targets.shape)
    break

# Train
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 10
for e in range(epochs):
    print(f"Epoch {e+1}")
    epoch_loss = train(train_dataloader, model, loss_fn, optimizer, device)
    print(f"Epoch loss: {epoch_loss}")
    print("-----------------------------")
print("Training finished")


#### SAVING MODEL
torch.save(model, 'model.pth')


#### TESTING



#### INFERING
#infer(infer_dataloader, model, device)
