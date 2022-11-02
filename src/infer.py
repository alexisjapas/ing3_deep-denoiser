import torch
from matplotlib import pyplot as plt


def infer(dataloader, model, device):
    model.eval()
    with torch.no_grad():
        X, _ = next(iter(dataloader))
        plt.figure(1)
        plt.imshow(X.squeeze())
        X = X.to(device)
        output = model(X)
        output = output.to('cpu').detach().numpy()
        print(output.shape)
        output = output.squeeze()
        print(output.shape)
        plt.figure(2)
        plt.imshow(output)
        plt.show()


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from DenoiserDataset import DenoiserDataset


    model = torch.load('model.pth').to('cpu')
    infer_data_path = "/home/obergam/Data/flir/images_thermal_val/"
    infer_dataset = DenoiserDataset(infer_data_path, 0, 0.1)
    infer_dataloader = DataLoader(infer_dataset, batch_size=1, num_workers=1, shuffle=True)

    infer(infer_dataloader, model, 'cpu')
