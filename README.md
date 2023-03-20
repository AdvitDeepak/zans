# ZANS 😁

## Setup

1. Install the required packages. There may be some not included. 
```
pip install torch numpy scikit-learn tqdm
```
2. Download and unzip the project data. Place the `project` directory in the root directory of this repo.

## Model Training and Testing

You can modify `constants.py` with model hyperparameters and parameters between runs. Choose the model to run in `constants.py` by setting the `CURR_MODEL` field:

      CURR_MODEL = "CNN"

Model parameters are set in the `params` dictionary. Each model's parameter dictionary must contain the following fields:

      "CNN" : {
         # number of epochs to train for
         "EPOCHS" : 100,
         # number of examples per batch 
         "BATCH_SIZE": 128, 
         # evaluate on val data after N_EVAL epochs
         "N_EVAL": 1,
         # reshapes data to use CNN as first layer
         "USE_CNN": True,
         # adds target data to inputs in case of transformer
         "NEEDS_TGTS": False,
         # which participant to use for train data (None = all particpants)
         "TR_PERSON_IDX": None,
         # which participant to use for val data (None = all particpants)
        "VAL_PERSON_IDX": None,
      }


To start training and evaluation, run the following command:

```
python main.py
```

