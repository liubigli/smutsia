from typing import Union, Tuple, List
import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from torch_geometric.data import Data, InMemoryDataset


def sample_circles(n_samples: int, noise: float, factor: float = .5, random_state: Union[int, None] = None):

    x, y = make_circles(n_samples, factor=factor, noise=noise, random_state=random_state)
    return x, y


def sample_blobs(n_samples: int, random_state: Union[int, None] = None):

    x, y = make_blobs(n_samples, random_state=random_state)
    return x, y


def sample_varied(n_samples: int, cluster_std: Union[List, None] = None, random_state: Union[int, None] = None):

    if cluster_std is None:
        cluster_std = [1.0, 2.5, 0.5]

    x, y = make_blobs(n_samples, cluster_std=cluster_std, random_state=random_state)
    return x, y


def sample_aniso(n_samples: int, anisotropic_transf: Union[List, None] = None, random_state: Union[int, None] = None):

    if anisotropic_transf is None:
        anisotropic_transf = [[0.6, -0.6], [-0.4, 0.8]]
    x, y = make_blobs(n_samples, random_state=random_state)
    x = np.dot(x, anisotropic_transf)
    return x, y


def sample_moons(n_samples: int, noise: float, shift_x: float = 2.0, shift_y: float = 2.0,
                 random_state: Union[int, None] = None):

    x, y = make_moons(n_samples, noise=noise, random_state=random_state)

    x, y = np.concatenate((x, x + (shift_x, shift_y))), np.concatenate((y, y + 2))
    
    return x, y


def generate_dataset(name: str, total_samples: int, max_points: int, noise: Union[float, Tuple[float, float]],
                     random_length: bool = True, seed: int = -1):
    """
    Function that generate one of the five possible toy datasets

    Parameters
    ----------
        name: str, optional {'cicles', 'moons', 'blobs', 'varied', 'aniso'}
            Name of Dataset to sample

        total_samples: int
            Total number of samples to generate

        max_points: int
            Maximum number of points in each sample

        noise: Union[float, Tuple[float, float]]
            noise value to use or interval of noises to use

        random_length: bool
            if yes each sample contains a random number of points

        seed: int
            if seed > 0 then we use a seed to generate always the same dataset

    Returns
    -------
        data: List[Data]
            List of samples of the generated dataset
    """

    data = []

    if seed >= 0:
        np.random.seed(seed)
        seeds = np.random.randint(1024, size=total_samples)
    else:
        seeds = None

    for n in range(total_samples):
        if random_length:
            if seeds is not None:
                np.random.seed(seeds[n])
            num_points = np.random.randint(max_points // 2, max_points)
        else:
            num_points = max_points

        if isinstance(noise, Tuple):
            min_noise = min(noise)
            max_noise = max(noise)
            if seed >= 0:
                np.random.seed(seeds[n])
            sample_noise = max(min_noise, max_noise * np.random.rand())
        else:
            sample_noise = noise

        random_state = np.random.randint(1024) if seeds is None else seeds[n]

        # draw a sample according to the dataset name
        if name == 'circles':
            x, y = sample_circles(num_points, sample_noise, random_state=random_state)
        elif name == 'moons':
            # we halve samples befor passing because the points are copied inside the function
            x, y = sample_moons(num_points // 2, sample_noise, random_state=random_state)
        elif name == 'blobs':
            x, y = sample_blobs(num_points, random_state=random_state)
        elif name == 'varied':
            x, y = sample_varied(num_points, random_state=random_state)
        elif name == 'aniso':
            x, y = sample_aniso(num_points, random_state=random_state)
        else:
            raise KeyError(f"Dataset name {name} not known. "
                           f"Possible choices are 'circles', 'moons', 'blobs', 'varied', 'aniso'")

        x, y = torch.Tensor(x), torch.from_numpy(y)

        data.append(Data(x=x, y=y))

    return data


class ToyDatasets(InMemoryDataset):
    def __init__(self, length: int, name: str, noise: Union[float, Tuple[float, float]] = 0.05,
                 max_samples: int = 300, random_length: bool = True, seed: int = -1):
        """
        Class that generates an InMemoryDataset for Ultrametric Fitting tests

        Parameters
        ----------
            length: int
                Total number of samples to generate in the dataset

            name: str Optional {'cicles', 'moons', 'blobs', 'varied', 'aniso'}
                Name of the toy dataset to generate

            noise: Union[float, Tuple[float, float]]
                noise used to generate points

            max_samples: int
                Maximum number of points for each sample

            random_length: bool
                True if you want that each sample has a different number of points. False for samples with all the same
                length

            seed: int
                If you want generate always the same dataset use a value equal or greater than 0
        """
        super(ToyDatasets, self).__init__('.', None, None, None)
        self.length = length
        self.name = name.lower()
        self.noise = noise
        self.max_samples = max_samples
        self.random_length = random_length
        self.seed = seed

        data = generate_dataset(name=self.name, total_samples=self.length,
                                max_points=self.max_samples, noise=self.noise, random_length=self.random_length,
                                seed=self.seed)

        self.data, self.slices = self.collate(data)
