import torch
import numpy as np
from sklearn.datasets import make_moons
from torch_geometric.data import Data, InMemoryDataset


def generate_moons(total_samples, max_samples, max_noise, random_length=True):
    data = []
    sampled_points = 0
    for n in range(total_samples):
            if random_length:
                samples = np.random.randint(max_samples // 4, max_samples // 2)
            else:
                samples = max_samples // 2

            sampled_points += samples
            noise = max(max_noise / 2, max_noise * np.random.rand())
            x_shift = 2
            y_shift = 2
            random_state = np.random.randint(1024)
            x, y = make_moons(samples, noise=noise, random_state=random_state)

            x, y = np.concatenate((x, x + (x_shift, y_shift))), np.concatenate((y, y + 2))
            x, y = torch.Tensor(x), torch.from_numpy(y)

            data.append(Data(x=x, y=y))

    return data


class GenerateDatasets(InMemoryDataset):
    def __init__(self, length: int, name: str = 'moons', noise: float = 0.05, max_samples: int = 300):
        super(GenerateDatasets, self).__init__('.', None, None, None)
        self.length = length
        self.name = name
        self.noise = noise
        self.max_samples = max_samples
        data = generate_moons(total_samples=length, max_samples=max_samples, max_noise=noise)
        self.data, self.slices = self.collate(data)
