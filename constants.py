DATASET_PATH = "../../project/"
SUMMARY_PATH = "summaries/"

CURR_MODEL = "RNN"

DATA = {
    "TRIM_END" : 500, 
    "AUG_SUBSAMPLE_SIZE": 2, 
}

params = {
    "RNN" : {
        "EPOCHS" : 100, 
        "BATCH_SIZE": 128, 
        "N_EVAL": 1 
    }, 
    "TRN": {
        "EPOCHS" : 100, 
        "BATCH_SIZE" : 128, 
        "N_EVAL" : 11,
        "NUM_HEADS" : 16,
    }
}
