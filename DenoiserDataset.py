import torch
from torch.utils.data import Dataset
from matplotlib import image as mpimg
from os import listdir
from os.path import isfile, join
import numpy as np

import noises
import transformers


class DenoiserDataset(Dataset):
    def __init__(self, dataset_path, crop_size, noise_density):
        self.images_paths = [join(dataset_path, path) for path in listdir(dataset_path) if isfile(join(dataset_path, path))]
        self.crop_size = crop_size
        self.noise_density = noise_density

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        # Read input image
        image_path = self.images_paths[index]
        image = mpimg.imread(image_path).astype(float)

        # Crop
        if self.crop_size > 0:
            image = transformers.random_crop(image, self.crop_size, self.crop_size)

        # Normalize
        image = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Generate final input / target
        noisy_image = noises.salt_and_pepper(image, self.noise_density)
        target = image.copy()

        # Transform to tensor
        tensor_noisy_image = torch.as_tensor(np.array([noisy_image]), dtype=torch.float)
        tensor_target = torch.as_tensor(np.array([target]), dtype=torch.float)

        return tensor_noisy_image, tensor_target
