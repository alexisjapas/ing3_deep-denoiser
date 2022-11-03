import torch
import numpy as np

import transformers as trfs


def infer(image, model, device="cpu"):
    # Working on input image
    image = trfs.normalize(image)
    image = torch.as_tensor(np.array([image]), dtype=torch.float).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        output = output.to("cpu").detach().numpy()
        output = output.squeeze()
    return output

def inferandom(dataloader, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        X, _ = next(iter(dataloader))
        noisy_input = X.squeeze()
        X = X.to(device)
        output = model(X)
        output = output.to("cpu").detach().numpy()
        output = output.squeeze()

        return noisy_input, output

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from DenoiserDataset import DenoiserDataset
    from matplotlib import pyplot as plt


    model = torch.load('model.pth').to('cpu')
    infer_data_path = "/home/obergam/Data/flir/images_thermal_val/"
    infer_dataset = DenoiserDataset(infer_data_path, 0, 0.5)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, num_workers=1, shuffle=True)

    noisy_input, output = infer(infer_dataloader, model, 'cpu')
    plt.figure("Noisy input"); plt.imshow(noisy_input)
    plt.figure("Output"); plt.imshow(output)
    plt.show()
