import random
from typing import Tuple, List

from torch.utils.data import Dataset

from classes.data.datasets.Canary import Canary
from classes.data.datasets.DementiaBank import DementiaBank


class DatasetRandomizer(Dataset):

    def __init__(self, datasets: List):
        self.__datasets = datasets
        self.__datasets_map = {
            "canary": Canary(),
            "dementiabank": DementiaBank(),
        }
        try:
            self.__datasets_map = {dataset: self.__datasets_map[dataset] for dataset in self.__datasets}
        except KeyError:
            print("One or more datasets in '{}' are not supported! Supported datasets are {}"
                  .format(self.datasets, self.__datasets_map.keys()))

    def __getitem__(self, idx: int) -> Tuple:
        dataset = random.choice(list(self.__datasets_map.values()))
        return dataset.__getitem__(idx % len(dataset))

    def __len__(self) -> int:
        return sum((lambda x: x.__len__, list(self.__datasets_map.values())))
