import torch
import numpy as np

DATASET_PATH = "../project_data/"


class StartingDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split,
        train_val_split=0.2,
        trim_end=1000,
        maxpool_subsample=1,
        average_aug_subsample=0,
        average_aug_noise=0,
        subsample_aug_size=1,
        subsample_aug_noise=0
    ):
        if split == "train":
            self.X = np.load(DATASET_PATH + "X_train_valid.npy") # (2115, 22, 1000)
            self.X = self.X[int(train_val_split*len(self.X)):]
            #self.X = self.X.astype(np.float128)

            labels = np.load(DATASET_PATH + "y_train_valid.npy") - 769 # (2115,)
            labels = labels[int(train_val_split*len(labels)):]
            self.y = labels.astype(np.int64) 
            
            # self.y = np.zeros((labels.size, labels.max() + 1))
            # self.y[np.arange(labels.size), labels] = 1 # (2115, 4); converted into one-hot
            self.person = np.load(DATASET_PATH + "person_train_valid.npy") # (2115, 1)


        elif split == "val":
            self.X = np.load(DATASET_PATH + "X_train_valid.npy") # (2115, 22, 1000)
            self.X = self.X[:int(train_val_split*len(self.X))]


            labels = np.load(DATASET_PATH + "y_train_valid.npy") - 769 # (2115,)
            labels = labels[:int(train_val_split*len(labels))]
            self.y = labels.astype(np.int64)
            
            # self.y = np.zeros((labels.size, labels.max() + 1))
            # self.y[np.arange(labels.size), labels] = 1 # (2115, 4); converted into one-hot
            self.person = np.load(DATASET_PATH + "person_train_valid.npy") # (2115, 1)


        elif split == "test":
            self.X = np.load(DATASET_PATH + "X_test.npy") # (443, 22, 1000)


            labels = np.load(DATASET_PATH + "y_test.npy") - 769 # (443,)
            self.y = np.zeros((labels.size, labels.max() + 1))
            self.y[np.arange(labels.size), labels] = 1 # (443, 4); converted into one-hot
            self.person = np.load(DATASET_PATH + "person_test.npy") # (443, 1)
        else:
            raise Exception("Invalid split name")
    

        if split == "test": 
            self.X = torch.from_numpy(self.X).double() 
            self.length = self.X.shape[0]
            return 

        # Trimming the data (sample,22,1000) -> (sample,22,500)
        if trim_end in range(0, self.X.shape[2] + 1):
            self.X = self.X[:,:,0:trim_end]
        
        # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
        X_max = np.max(
            self.X.reshape(self.X.shape[0], self.X.shape[1], -1, maxpool_subsample),
            axis=3
        )
        
        total_X = X_max

        # Averaging + noise
        if average_aug_subsample > 1:
            X_average = np.mean(
                self.X.reshape(self.X.shape[0], self.X.shape[1], -1, average_aug_subsample),
                axis=3
            )
            X_average = X_average + np.random.normal(0.0, average_aug_noise, X_average.shape)
        
            total_X = np.vstack((total_X, X_average))
            self.y = np.hstack((self.y, self.y))
        
        # Subsampling
        if subsample_aug_size > 1:
            for i in range(subsample_aug_size):
                X_subsample = self.X[:, :, i::subsample_aug_size] + \
                    np.random.normal(0.0, subsample_aug_noise, self.X[:,:,i::subsample_aug_size].shape)
                total_X = np.vstack((total_X, X_subsample))
                self.y = np.hstack((self.y, self.y))
        
        self.X = total_X
        self.X = torch.from_numpy(self.X).double() 
        self.length = self.X.shape[0]

    def getParticipantData(self, participant):
        num_participants = self.person.max()
        if participant not in range(0, num_participants):
            raise Exception("Invalid participant number: choose between 0 and {}".format(num_participants - 1))
        participant_idxs = np.where(self.person == participant)
        return self.X[participant_idxs], self.y[participant_idxs]

    def __getitem__(self, index):
        inputs = self.X[index] 
        label = self.y[index]

        return inputs, label

    def __len__(self):
        return self.length

