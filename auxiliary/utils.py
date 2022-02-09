import argparse
import functools
import json
import os
import random
import re
from typing import Dict

import numpy as np
import torch

SEPARATOR = {"stars": "".join(["*"] * 100), "dashes": "".join(["-"] * 100), "dots": "".join(["."] * 100)}


def get_device(device_type: str) -> torch.device:
    if device_type == "cpu":
        print("\n Running on device 'cpu' \n")
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", device_type):
        if not torch.cuda.is_available():
            print("\n WARNING: running on cpu since device {} is not available \n".format(device_type))
            return torch.device("cpu")

        print("\n Running on device '{}' \n".format(device_type))
        return torch.device(device_type)

    raise ValueError("ERROR: {} is not a valid device! Supported device are 'cpu' and 'cuda:n'".format(device_type))


def make_deterministic(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False


def print_namespace(namespace: argparse.Namespace):
    print("\n" + SEPARATOR["dashes"])
    print("\n\t *** INPUT NAMESPACE PARAMETERS *** \n")
    for arg in vars(namespace):
        print(("\t - {} " + "".join(["."] * (25 - len(arg))) + " : {}").format(arg, getattr(namespace, arg)))
    print("\n" + SEPARATOR["dashes"] + "\n")


def infer_path(ns: argparse.Namespace) -> str:
    return os.path.join(ns.path_to_pretrained, ns.sal_dim, ns.sal_type + "_tccnet", ns.data_folder)


def save_settings(settings: argparse.Namespace, path_to_save: str):
    json.dump(vars(settings), open(os.path.join(path_to_save, "settings.json"), 'w'), indent=2)


def experiment_header(title: str):
    print("\n" + SEPARATOR["stars"])
    print("\t\t {}".format(title))
    print(SEPARATOR["stars"] + "\n")


def overload(func: callable) -> any:
    def fake_wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return fake_wrapper


def overloads(base: callable):
    def outer_wrapper(func):
        @functools.wraps(func)
        def inner_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return inner_wrapper

    return outer_wrapper


def _print_metrics(metrics: Dict, best_metrics: Dict):
    for mn, mv in metrics.items():
        print((" {} " + "".join(["."] * (15 - len(mn))) + " : {:.4f} (Best: {:.4f})")
              .format(mn.capitalize(), mv, best_metrics[mn]))

from ..classes.data.datasetModels.Canary import Canary
# import torch

class Util:
    def __init__(self):
        pass

# custom weights initialization called on Generator and Discriminator 
# (DCGAN paper authours specify that all model weights shall be randomly initialized from a Normal distribution with mean=0, stdev=0.02)
def weights_init(m):
    classname = m.__class__.__name__
    print(f'Weights Initialization Step: classname is: {classname}')
    if classname.find('Conv') != -1:
        nn.init.normal_()

# If time sequence is too long, training result would not be ideal
def sample_eye_tracking_sequences(sequence, output_len):
    seq_len = len(sequence)
    sample_frequency = seq_len / output_len
    sampled_sequence = torch.zeros(output_len, sequence.shape[1]) 
    for i in range(0, output_len, sample_frequency):
        sampled_sequence[i] = sequence[i]
    return sampled_sequence

# dataset = data.datasetModels.Canary.Canary() 
print("123")
