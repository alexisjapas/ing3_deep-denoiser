import torch
from torch.utils.data import Dataset
from matplotlib import image as mpimg
from os import listdir
from os.path import isfile, join
import numpy as np
from random import uniform

import noises
import transformers as trfs


class DenoiserDataset(Dataset):
    def __init__(self, dataset_path, crop_size, noise_density):
        self.targets_paths = [join(dataset_path, path) for path in listdir(dataset_path) if isfile(join(dataset_path, path))]
        self.crop_size = crop_size
        self.noise_density = noise_density

    def __len__(self):
        return len(self.targets_paths)

    def __getitem__(self, index):
        # Setup noise density if random
        if type(self.noise_density) == tuple:
            noise_density = uniform(self.noise_density[0], self.noise_density[1])
        else:
            noise_density = self.noise_density

        # Read input target
        target_path = self.targets_paths[index]
        target = mpimg.imread(target_path).astype(float)

        # Crop input target
        if self.crop_size > 0:
            target = trfs.random_crop(target, self.crop_size, self.crop_size)

        # Normalize input target
        target = trfs.normalize(target)

        # Generate final input / target
        noisy_input = target.copy()
        noisy_input = noises.salt_and_pepper(target, noise_density)

        # Transform to tensor
        tensor_noisy_input = torch.as_tensor(np.array([noisy_input]), dtype=torch.float)
        tensor_target = torch.as_tensor(np.array([target]), dtype=torch.float)

        return tensor_noisy_input, tensor_target
