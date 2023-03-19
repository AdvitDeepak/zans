import os

import constants
from data.StartingDataset import StartingDataset
from train_functions.starting_train import starting_train
from importlib import import_module


def main():
    config = constants.params[constants.CURR_MODEL]
    epochs, batch_size, n_eval = config['EPOCHS'], config[
        'BATCH_SIZE'], config['N_EVAL']
    need_tgts = config['NEEDS_TGTS']
    use_cnn = config['USE_CNN']

    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "n_eval": n_eval,
    }

    print(f"Epochs: {epochs}\nBatch size: {batch_size}")

    # Initalize dataset
    train_dataset = StartingDataset(
        "train",
        create_tgts=need_tgts,
        use_cnn=use_cnn,
        trim_end=constants.DATA["TRIM_END"],
        aug_subsample_size=constants.DATA["AUG_SUBSAMPLE_SIZE"])
    val_dataset = StartingDataset(
        "val",
        create_tgts=need_tgts,
        use_cnn=use_cnn,
        trim_end=constants.DATA["TRIM_END"],
        aug_subsample_size=constants.DATA["AUG_SUBSAMPLE_SIZE"])

    # Initialize model
    network_class = import_module("networks." +
                                  constants.CURR_MODEL).__getattribute__(
                                      constants.CURR_MODEL)
    model = network_class(config)
    model = model.float()

    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        use_masking=need_tgts,
    )


if __name__ == "__main__":
    main()
