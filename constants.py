DATASET_PATH = "project/"
SUMMARY_PATH = "summaries/"

CURR_MODEL = "CNN"

DATA = {
    "TRIM_END" : 500, 
    "AUG_SUBSAMPLE_SIZE": 2, 
    "NUM_CLASSES": 4,
    "NUM_ELECTRODES": 22,
}

params = {
    "CNN" : {
        "EPOCHS" : 100, 
        "BATCH_SIZE": 512, 
        "N_EVAL": 1,
        "USE_CNN": True,
        "NEEDS_TGTS": False,
        "TR_PERSON_IDX": None,
        "VAL_PERSON_IDX": None,
        "TRAIN_VAL_SPLIT": 0.1,

        "C_DROPOUT": [0.55, 0.65, 0.75],
        "OUT_CHANNELS": [32, 64, 128],
        "F_DROPOUT": 0.5,
        "HIDDEN_SIZES": [4096, 512, 64],
    }, 
    "CNNRNN" : {
        "EPOCHS" : 100, 
        "BATCH_SIZE": 128, 
        "N_EVAL": 1, 
        "USE_CNN": True,
        "NEEDS_TGTS": False,
        "TR_PERSON_IDX": None,
        "VAL_PERSON_IDX": None,
        "TRAIN_VAL_SPLIT": 0.1,

        # CNN
        "C_DROPOUT": 0.5,
        "HIDDEN_SIZE": 3200,
        "OUT_CHANNELS": [25, 50, 100, 200],

        # RNN
        "N_LAYERS": 1,
        "R_DROPOUT": 0.4,
        "FC_DROPOUT": 0.25,
        "HIDDEN_SIZE": 128,
    }, 
    "RNN" : {
        "EPOCHS" : 100, 
        "BATCH_SIZE": 128, 
        "N_EVAL": 1,
        "USE_CNN": False,
        "NEEDS_TGTS": False,
        "TR_PERSON_IDX": None,
        "VAL_PERSON_IDX": None,
        "TRAIN_VAL_SPLIT": 0.1,

        "RNN_DROPOUT": 0.4,
        "FC_DROPOUT": 0.2,
        "N_LAYERS": 1,
        "HIDDEN_SIZE": 128,
    }, 
    "TRN": {
        "EPOCHS" : 100, 
        "BATCH_SIZE" : 128, 
        "N_EVAL" : 1,
        "USE_CNN": False,
        "NEEDS_TGTS": True,
        "TR_PERSON_IDX": None,
        "VAL_PERSON_IDX": None,
        "TRAIN_VAL_SPLIT": 0.1,

        "N_HEADS" : 5,
        "N_LAYERS": 12,
    },
    "CTN": {
        "EPOCHS" : 100, 
        "BATCH_SIZE" : 128, 
        "N_EVAL" : 1,
        "USE_CNN": True,
        "NEEDS_TGTS": False,
        "TR_PERSON_IDX": None,
        "VAL_PERSON_IDX": None,
        "TRAIN_VAL_SPLIT": 0.1,

        "OUT_CHANNELS": 25,
        "TRANSFORMER_LAYERS": 12,
        "D_MODEL": 83,
        "N_HEADS": 1,
        "T_DROPOUT": 0.2,
        "C_DROPOUT": 0.5,
        "HIDDEN_SIZE": 2075,
    },
    "CTN_ALT": {
        "EPOCHS" : 100, 
        "BATCH_SIZE" : 128, 
        "N_EVAL" : 1,
        # technically uses CNN but doesn't require
        # input reshaping so set to False
        "USE_CNN": False,
        "NEEDS_TGTS": False,
        "TR_PERSON_IDX": None,
        "VAL_PERSON_IDX": None,
        "TRAIN_VAL_SPLIT": 0.1,

        "OUT_CHANNELS": 16,
        "TRANSFORMER_LAYERS": 12,
        "N_HEADS": 5,
        "D_MODEL": 125,
        "FC_DROPOUT": 0.4,
        "HIDDEN_SIZE_1": 1375,
        "HIDDEN_SIZE_2": 64,

    }
}
