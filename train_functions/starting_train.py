import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score 
from torch.utils import tensorboard

import constants 

global device 
device = torch.device('cpu') 

if torch.cuda.is_available():
    device = torch.device('cuda:0')


def starting_train(train_dataset, val_dataset, model, hyperparameters, use_masking=False):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        use_masking: True if using TRN to mask src data
    """

    # Get keyword arguments
    batch_size, epochs, n_eval = hyperparameters["batch_size"], hyperparameters["epochs"], hyperparameters["n_eval"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-1, eps=1e-7)
    loss_fn = nn.CrossEntropyLoss()

    model = model.to(device)

    # Initialize summary writer (for logging)
    summary_path = constants.SUMMARY_PATH
    tb_summary = None
    if summary_path is not None:
        tb_summary = tensorboard.SummaryWriter(summary_path)


    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):

            input_data, label_data = batch
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            if use_masking:
                batch_size = input_data.shape[0]
                src_mask = model.generate_square_subsequent_mask(22, batch_size)
                pred = model(input_data, src_mask)
            else:
                pred = model(input_data)

            loss = loss_fn(pred, label_data) 
            pred = pred.argmax(axis=1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"\n    Train Loss: {loss.item()}")

        # Periodically evaluate our model + log to Tensorboard
        if epoch % n_eval == 0:

            # Compute training loss and accuracy.
            train_accuracy = accuracy_score(label_data, pred)
            print(f"    Train Accu: {train_accuracy}")

            # Compute validation loss and accuracy.
            valid_loss, valid_accuracy = evaluate(val_loader, model, loss_fn)

            # # Log the results to Tensorboard
            if tb_summary:
                tb_summary.add_scalar('Loss (Training)', loss, epoch)
                tb_summary.add_scalar('Accuracy (Training)', train_accuracy, epoch)
                tb_summary.add_scalar('Loss (Validation)', valid_loss, epoch)
                tb_summary.add_scalar('Accuracy (Validation)', valid_accuracy, epoch)

            print(f"    Valid Loss: {valid_loss}")
            print(f"    Valid Accu: {valid_accuracy}")

            model.train()

            step += 1

        print()


 

def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset!
    """
    model.eval()
    model = model.to(device)

    loss, correct, count = 0, 0, 0
    with torch.no_grad(): 
        for batch in val_loader:
            input_data, label_data = batch

            # Move both images and labels to GPU, if available
            input_data = input_data.to(device)
            label_data = label_data.to(device)

            pred = model(input_data)
            loss += loss_fn(pred, label_data).mean().item()

            # Update both correct and count (use metrics for tensorboard)
            correct += (torch.argmax(pred, dim=1) == label_data).sum().item()
            count += len(label_data)

    return loss, correct/count
