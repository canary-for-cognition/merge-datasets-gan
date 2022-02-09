import random

from torch.utils.data import Dataset, DataLoader

from classes.data.datasetModels.Canary import Canary
from classes.data.datasetModels.DementiaBank import DementiaBank
from classes.data.datasetModels.Ondri import Ondri


class DatasetRandomizer(Dataset):

    def __init__(self, datasets, sequence_length):
        self.__datasets = datasets
        self.__datasets_map = {
            "canary": Canary(sequence_length),
            "dementiabank": DementiaBank(sequence_length),
            "ondri": Ondri(),
        }
        # assume only 2 datasets to merge each time
        self.__datasets_labels_map = {
            datasets[0]: 0,
            datasets[1]: 1
        }
        try:
            self.__datasets_map = {dataset: self.__datasets_map[dataset] for dataset in self.__datasets}
        except KeyError:
            print("One or more datasets in '{}' are not supported! Supported datasets are {}"
                  .format(self.datasets, self.__datasets_map.keys()))

    def __getitem__(self, idx):
        # dataset = random.choice(list(self.__datasets_map.values()))
        # Q: why idx % len(dataset) instead of idx ?
        # return dataset.__getitem__(idx % len(dataset))
        random_dataset_name = random.choice(self.__datasets) 
        random_dataset = self.__datasets_map[random_dataset_name]
        random_datapoint = random_dataset.__getitem__(idx % len(random_dataset))
        random_datapoint["dataset_name"] = random_dataset_name
        random_datapoint["dataset_label"] = float(self.__datasets_labels_map[random_dataset_name])
        return random_datapoint

    def __len__(self):
        return sum([(lambda x: len(x))(x) for x in self.__datasets_map.values()])

# dataset = DatasetRandomizer(["canary", "dementiabank"])
# dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
# for i, data in enumerate(dataloader, 0):
#     print('i is: ')
#     print(i)
#     print('data is: ')
#     print(data)