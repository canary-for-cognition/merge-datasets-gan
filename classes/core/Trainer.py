import os

import pandas as pd
from torch.utils.data import DataLoader

from auxiliary.utils import SEPARATOR
from classes.core.LossTracker import LossTracker
from classes.core.MetricsTracker import MetricsTracker
from classes.models.ModelAutoEncoder import ModelAutoEncoder
from classes.models.ModelDiscriminator import ModelDiscriminator


class Trainer:

    def __init__(self):
        self.__auto_encoder, self.__discriminator = ModelAutoEncoder(), ModelDiscriminator()

        self.__train_loss_auto_encoder, self.__val_loss_auto_encoder = LossTracker(), LossTracker()
        self.__train_loss_discriminator, self.__val_loss_discriminator = LossTracker(), LossTracker()
        self.__val_evaluator_auto_encoder, self.__val_evaluator_discriminator = MetricsTracker(), MetricsTracker()

        self.__best_val_loss = 1000
        self.__best_metrics_auto_encoder = self.__val_evaluator_auto_encoder.get_best_metrics()
        self.__best_metrics_discriminator = self.__val_evaluator_discriminator.get_best_metrics()

    @staticmethod
    def __print_heading(text: str):
        print("\n" + SEPARATOR["dashes"])
        print("\t\t {}".format(text))
        print(SEPARATOR["dashes"] + "\n")

    def __train_epoch(self, data: DataLoader, epoch: int, epochs: int):
        self.__auto_encoder.train_mode()
        self.__discriminator.train_mode()

        for i, (x, y) in data:
            (x, o1), o2 = self.__auto_encoder.predict(x), self.__discriminator.predict(x)
            l1, l2 = self.__auto_encoder.optimize(o1, x), self.__discriminator.optimize(o2, y)
            self.__train_loss_auto_encoder.update(l1)
            self.__train_loss_discriminator.update(l1)
            print(" Epoch {}/{}  - Loss: [ Auto Encoder: {:.4f} - Discriminator {:.4f}]"
                  .format(epoch + 1, epochs, l1, l2))

    def __validate_epoch(self, data: DataLoader, epoch: int, epochs: int):
        self.__auto_encoder.eval_mode()
        self.__discriminator.eval_mode()

        for i, (x, y) in data:
            (x, o1), o2 = self.__auto_encoder.predict(x), self.__discriminator.predict(x)
            l1, l2 = self.__auto_encoder.get_loss(o1, x).item(), self.__discriminator.get_loss(o2, y).item()
            self.__val_loss_auto_encoder.update(l1)
            self.__val_loss_discriminator.update(l1)
            self.__val_evaluator_auto_encoder.add_error((o1, x))
            self.__val_evaluator_discriminator.add_error((o2, y))
            print(" Epoch {}/{} - Loss: [ Decoder: {:.4f} - Discriminator {:.4f}]"
                  .format(epoch + 1, epochs, l1, l2))

    def __check_if_best_model(self, path_to_log: str):
        loss = self.__val_loss_auto_encoder.avg + self.__val_loss_discriminator.avg
        if 0 < loss < self.__best_val_loss:
            self.__best_val_loss = loss
            self.__best_metrics_auto_encoder = self.__val_evaluator_auto_encoder.update_best_metrics()
            self.__best_metrics_discriminator = self.__val_evaluator_discriminator.update_best_metrics()
            print("\n -> Saving new best model...")
            self.__auto_encoder.save(os.path.join(path_to_log, "auto_encoder.pth"))
            self.__discriminator.save(os.path.join(path_to_log, "discriminator.pth"))

    def __log_metrics(self, path_to_metrics: str):
        metrics_decoder = self.__val_evaluator_auto_encoder.get_metrics()
        metrics_discriminator = self.__val_evaluator_discriminator.get_metrics()
        log_data = pd.DataFrame({"train_loss_decoder": [self.__train_loss_auto_encoder.avg],
                                 "train_loss_discriminator": [self.__train_loss_discriminator.avg],
                                 "val_loss_decoder": [self.__val_loss_auto_encoder.avg],
                                 "val_loss_discriminator": [self.__val_loss_discriminator.avg],
                                 **{"best_" + k: [v] for k, v in self.__best_metrics_auto_encoder.items()},
                                 **{"best_" + k: [v] for k, v in self.__best_metrics_discriminator.items()},
                                 **{k: [v] for k, v in metrics_decoder.items()},
                                 **{k: [v] for k, v in metrics_discriminator.items()}})

        header = log_data.keys() if not os.path.exists(path_to_metrics) else False
        log_data.to_csv(path_to_metrics, mode='a', header=header, index=False)

    def train(self, train_data: DataLoader, val_data: DataLoader, epochs: int, path_to_log: str):

        for epoch in range(epochs):

            self.__print_heading("Training Epoch {}/{}".format(epoch + 1, epochs))
            self.__train_epoch(train_data, epoch, epochs)

            # --- Validation ---

            if not epoch % 5:
                self.__print_heading("Validating Epoch {}/{}".format(epoch + 1, epochs))
                self.__validate_epoch(val_data, epoch, epochs)
                self.__check_if_best_model(path_to_log)

            self.__log_metrics(path_to_metrics=os.path.join(path_to_log, "metrics.csv"))
