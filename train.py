from trainer import *
from data import *
from plotter import Plotter

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import numpy as np
import time

# -----------------------------
# Helper function: create corrupted input by masking one element
def corrupt_input(x, seq_len):
    """
    Input:
        x: tensor of shape [batch_size, seq_len]
        seq_len: length of the sequence
    Output:
        x_corrupted: tensor of shape [batch_size, seq_len-1, 1], input sequences with one element removed and an added feature dimension
        mask_idx: tensor of shape [batch_size], indices of the removed elements in each sequence
    """
    batch_size = x.size(0)
    mask_idx = torch.randint(0, seq_len, (batch_size,), device=x.device)

    x_corrupted = []
    for i in range(batch_size):
        idx = mask_idx[i]
        # Concatenate all elements except the masked index to form corrupted input
        xi = torch.cat([x[i, :idx], x[i, idx+1:]], dim=0)
        x_corrupted.append(xi.unsqueeze(-1))  # add feature dimension
    x_corrupted = torch.stack(x_corrupted)  # shape: [batch_size, seq_len-1, 1]
    return x_corrupted, mask_idx

# -----------------------------
# Main function

def main():
    end_name = ''
    filename = "avgWires.csv"
    doTraining = True

    print('\n\nLoading data...')
    startT_data = time.time()

    # Read data
    events = read_file(filename)

    # Define plotter
    plotter = Plotter(print_dir="plots/", end_name=end_name)

    dataset = FeatureDataset(events)
    val_size = 200000
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    print('\n\nTrain size:', train_size)
    print('Test size:', val_size)

    train_loader = DataLoader(train_set, batch_size=32, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, num_workers=4, shuffle=False)

    X_sample = next(iter(train_loader))
    print('X_sample:', X_sample.shape)  # e.g. torch.Size([32, 6])

    endT_data = time.time()
    print(f'Loading data took {endT_data - startT_data:.2f}s \n\n')

    # Initialize model, assign seq_len attribute
    model = TransformerAutoencoder()
    model.seq_len = X_sample.shape[1]  # e.g. 6

    loss_tracker = LossTracker()

    trainer = pl.Trainer(
        max_epochs=100,
        enable_progress_bar=True,
        log_every_n_steps=1,
        enable_checkpointing=False,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        callbacks=[loss_tracker]
    )

    if doTraining:
        print('\n\nTraining...')
        startT_train = time.time()
        trainer.fit(model, train_loader, val_loader)
        endT_train = time.time()

        print(f'Training took {(endT_train - startT_train) / 60:.2f} minutes \n\n')

        plotter.plotTrainLoss(loss_tracker)

        # Save the model (create example input first)
        example_batch = next(iter(train_loader))
        x_corrupted, mask_idx = corrupt_input(example_batch, seq_len=model.seq_len)
        example_input = (x_corrupted, mask_idx)
        torchscript_model = torch.jit.trace(model, example_input)
        torchscript_model.save(f"nets/tmae{end_name}.pt")

    # Load the model and run inference
    model = torch.jit.load(f"nets/tmae{end_name}.pt")
    model.eval()

    all_preds = []
    all_targets = []

    startT_test = time.time()

    with torch.no_grad():
        for batch in val_loader:
            # batch shape: [batch_size, seq_len]
            x_corrupted, mask_idx = corrupt_input(batch, seq_len=6)
            y_true = batch[torch.arange(batch.size(0)), mask_idx]
            y_pred = model(x_corrupted, mask_idx)

            all_preds.append(y_pred.cpu())
            all_targets.append(y_true.cpu())

    endT_test = time.time()
    print(f'Test with {val_size} samples took {endT_test - startT_test:.2f}s \n\n')

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # Plot pred vs target
    plotter.plot_pred_target(all_preds, all_targets)

    # Plot diff
    plotter.plot_diff(all_preds, all_targets)

    print("Predictions shape:", all_preds.shape)  # should be [val_size]
    print("Targets shape:", all_targets.shape)

    print("First 10 predictions:", all_preds[:10].numpy())
    print("First 10 true values:", all_targets[:10].numpy())

if __name__ == "__main__":
    main()
