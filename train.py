import argparse
import os
import time

from auxiliary.settings import RANDOM_SEED
from auxiliary.utils import make_deterministic, print_namespace, experiment_header
from classes.core.Trainer import Trainer
from classes.data.DataHandler import DataHandler


def main(ns: argparse.Namespace):
    epochs, dataset_list, folds = ns.epochs, ns.datasets, ns.folds
    path_to_log = os.path.join("logs", str(time.time()))
    os.makedirs(path_to_log)

    experiment_header("Training GAN on datasets: {}".format(dataset_list))

    for fold in range(folds):
        print("\t *** FOLD {}/{} ***".format(fold + 1, folds))
        train_data, val_data = DataHandler(dataset_list).train_test_loaders(fold)
        Trainer().train(train_data, val_data, epochs, path_to_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--lr", type=int, default=0.00005)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--datasets", type=str, default="canary dementiabank")
    parser.add_argument("--folds", type=int, default=10)
    namespace = parser.parse_args()
    make_deterministic(namespace.random_seed)
    print_namespace(namespace)
    main(namespace)
