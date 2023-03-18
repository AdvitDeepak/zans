DATASET_PATH = "../../project/"
SUMMARY_PATH = "summaries/"

CURR_MODEL = "RNN"

DATA = {
    "TRIM_END" : 500, 
    "AUG_SUBSAMPLE_SIZE": 2, 
    "NUM_CLASSES": 4,
    "NUM_ELECTRODES": 22,
}

params = {
    "RNN" : {
        "EPOCHS" : 100, 
        "BATCH_SIZE": 128, 
        "N_EVAL": 1,
        "RNN_DROPOUT": 0.4,
        "FC_DROPOUT": 0.2,
        "N_LAYERS": 1,
        "HIDDEN_SIZE": 128,
    }, 
    "TRN": {
        "EPOCHS" : 100, 
        "BATCH_SIZE" : 128, 
        "N_EVAL" : 1,
        "N_HEADS" : 5,
        "N_LAYERS": 12,
    }
}
