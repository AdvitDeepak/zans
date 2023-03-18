import os

import constants
from data.StartingDataset import StartingDataset
from train_functions.starting_train import starting_train
from importlib import import_module

# TODO: make a conda env to run everything 

def main():
    # Get command line arguments

    config = constants.params[constants.CURR_MODEL] 
    epochs, batch_size, n_eval = config['EPOCHS'], config['BATCH_SIZE'], config['N_EVAL']
    use_trn = constants.CURR_MODEL == "TRN"
    
    hyperparameters = {"epochs": epochs, "batch_size": batch_size, "n_eval": n_eval}

    print(f"Epochs: {epochs}\nBatch size: {batch_size}")

    # Initalize dataset 
    train_dataset = StartingDataset("train", 
                                    use_trn=use_trn,
                                    trim_end=constants.DATA["TRIM_END"],
                                    aug_subsample_size=constants.DATA["AUG_SUBSAMPLE_SIZE"])
    val_dataset = StartingDataset("val",
                                  use_trn=use_trn,
                                  trim_end=constants.DATA["TRIM_END"],
                                  aug_subsample_size=constants.DATA["AUG_SUBSAMPLE_SIZE"])

    # Initialize model 
    network_class = import_module("networks." + constants.CURR_MODEL).__getattribute__(constants.CURR_MODEL)
    model = network_class(config)
    model = model.float()

    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        use_masking=use_trn,
    )


if __name__ == "__main__":
    main()
