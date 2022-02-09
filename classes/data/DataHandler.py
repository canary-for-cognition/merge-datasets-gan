from multiprocessing import cpu_count
from typing import Tuple, List

from torch.utils.data import DataLoader

from classes.data.DatasetRandomizer import DatasetRandomizer


class DataHandler:
    def __init__(self, dataset_list: List):
        self.__dataset = DatasetRandomizer
        self.__dataset_list = dataset_list

    def train_test_loaders(self, fold: int) -> Tuple:
        training_loader = self.get_loader(train=True, fold=fold)
        test_loader = self.get_loader(train=False, fold=fold)
        return training_loader, test_loader

    def get_loader(self, train: bool, fold: int, batch_size=1) -> DataLoader:
        dataset = self.__dataset(self.__dataset_list)
        self._check_empty_set(dataset, train)
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=cpu_count(), drop_last=True)

    @staticmethod
    def _check_empty_set(dataset: DatasetRandomizer, train: bool):
        set_type = "TRAIN" if train else "TEST"
        if not len(dataset):
            raise ValueError("Empty {} set!".format(set_type))
        print(" {} dataset size: {}".format(set_type, len(dataset)))
