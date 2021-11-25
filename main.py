import argparse
import os
import time

import pandas as pd
from torch.utils.data import DataLoader

from auxiliary.settings import RANDOM_SEED
from auxiliary.utils import make_deterministic, print_namespace, experiment_header, SEPARATOR
from classes.core.LossTracker import LossTracker
from classes.core.MetricsTracker import MetricsTracker
from classes.data.DatasetRandomizer import DatasetRandomizer
from classes.models.ModelDecoder import ModelDecoder
from classes.models.ModelDiscriminator import ModelDiscriminator
from classes.models.ModelEncoder import ModelEncoder


def main(ns: argparse.Namespace):
    epochs, dataset_list, eval_freq = ns.epochs, ns.datasets, ns.eval_freq
    path_to_log = os.path.join("logs", str(time.time()))
    os.makedirs(path_to_log)
    path_to_metrics = os.path.join(path_to_log, "metrics.csv")

    dataset = DatasetRandomizer(dataset_list)
    dataloader = DataLoader(dataset, shuffle=True, drop_last=True)
    encoder, decoder, discriminator = ModelEncoder(), ModelDecoder(), ModelDiscriminator()

    experiment_header("Training GAN on datasets: {}".format(dataset_list))

    train_loss_decoder, val_loss_decoder = LossTracker(), LossTracker()
    train_loss_discriminator, val_loss_discriminator = LossTracker(), LossTracker()

    val_evaluator_decoder, val_evaluator_discriminator = MetricsTracker(), MetricsTracker()

    best_val_loss = 1000
    best_metrics_decoder = val_evaluator_decoder.get_best_metrics()
    best_metrics_discriminator = val_evaluator_discriminator.get_best_metrics()

    for epoch in range(epochs):

        print("\n" + SEPARATOR["dashes"])
        print("\t\t Training Epoch {}/{}".format(epoch + 1, epochs))
        print(SEPARATOR["dashes"] + "\n")

        # --- Training ---
        encoder.train_mode()
        decoder.train_mode()
        discriminator.train_mode()

        for i, (x, y) in dataloader:
            x = encoder.predict(x)
            o1, o2 = decoder.predict(x), discriminator.predict(x)
            l1, l2 = decoder.optimize(o1, x), discriminator.optimize(o2, y)
            train_loss_decoder.update(l1)
            train_loss_discriminator.update(l1)
            print(" Epoch {}/{}  - Loss: [ Decoder: {:.4f} - Discriminator {:.4f}]".format(epoch + 1, epochs, l1, l2))

        print("\n" + SEPARATOR["dashes"])
        print("\t\t Validating Epoch {}/{}".format(epoch + 1, epochs))
        print(SEPARATOR["dashes"] + "\n")

        # --- Validation ---

        if not epoch % eval_freq:
            encoder.eval_mode()
            decoder.eval_mode()
            discriminator.eval_mode()

            # Training
            for i, (x, y) in dataloader:
                x = encoder.predict(x)
                o1, o2 = decoder.predict(x), discriminator.predict(x)
                l1, l2 = decoder.get_loss(o1, x).item(), discriminator.get_loss(o2, y).item()
                val_loss_decoder.update(l1)
                val_loss_discriminator.update(l1)
                val_evaluator_decoder.add_error((o1, x))
                val_evaluator_discriminator.add_error((o2, y))
                print(" Epoch {}/{} - Loss: [ Decoder: {:.4f} - Discriminator {:.4f}]"
                      .format(epoch + 1, epochs, l1, l2))

        # --- Check best model ---

        loss = val_loss_decoder.avg + val_loss_discriminator.avg
        if 0 < loss < best_val_loss:
            best_val_loss = loss
            best_metrics_decoder = val_evaluator_decoder.update_best_metrics()
            best_metrics_discriminator = val_evaluator_discriminator.update_best_metrics()
            print("\n -> Saving new best model...")
            encoder.save(os.path.join(path_to_log, "encoder.pth"))
            decoder.save(os.path.join(path_to_log, "decoder.pth"))
            encoder.save(os.path.join(path_to_log, "discriminator.pth"))

        # --- Save metrics ---

        metrics_decoder = val_evaluator_decoder.get_metrics()
        metrics_discriminator = val_evaluator_discriminator.get_metrics()
        log_data = pd.DataFrame({"train_loss_decoder": [train_loss_decoder.avg],
                                 "train_loss_discriminator": [train_loss_discriminator.avg],
                                 "val_loss_decoder": [val_loss_decoder.avg],
                                 "val_loss_discriminator": [val_loss_discriminator.avg],
                                 **{"best_" + k: [v] for k, v in best_metrics_decoder.items()},
                                 **{"best_" + k: [v] for k, v in best_metrics_discriminator.items()},
                                 **{k: [v] for k, v in metrics_decoder.items()},
                                 **{k: [v] for k, v in metrics_discriminator.items()}})

        header = log_data.keys() if not os.path.exists(path_to_metrics) else False
        log_data.to_csv(path_to_metrics, mode='a', header=header, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=RANDOM_SEED)
    parser.add_argument("--lr", type=int, default=0.00005)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--datasets", type=str, default="canary dementiabank")
    parser.add_argument("--eval_freq", type=int, default=5)
    namespace = parser.parse_args()

    make_deterministic(namespace.random_seed)
    print_namespace(namespace)

    main(namespace)
