from trainer import *
from data import *
from plotter import Plotter

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import numpy as np
import time

import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Transformer Masked Autoencoder Training")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu", "auto"], default="auto",
                        help="Choose device: cpu, gpu, or auto (default: auto)")
    parser.add_argument("inputs", type=str, nargs="*", default=["avgWires.csv"],
                        help="One or more input CSV files (default: avgWires.csv)")
    parser.add_argument("--max_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--outdir", type=str, default="outputs",
                        help="Directory to save models and plots")
    parser.add_argument("--end_name", type=str, default="",
                        help="Optional suffix to append to output files (default: none)")
    parser.add_argument("--d_model", type=int, default=32,
                        help="Transformer embedding dimension (must be divisible by nhead)")
    parser.add_argument("--nhead", type=int, default=4,
                        help="Number of attention heads in the transformer (default: 4)")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer encoder layers (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for optimizer (default: 1e-3)")
    parser.add_argument("--no_train", action="store_true",
                        help="Skip training and only run inference using a saved model")
    return parser.parse_args()

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
    args = parse_args()

    inputs = args.inputs if args.inputs else ["avgWires.csv"]
    outDir = args.outdir
    maxEpochs = args.max_epochs
    batchSize = args.batch_size
    end_name = args.end_name
    doTraining = not args.no_train

    os.makedirs(args.outdir, exist_ok=True)


    print('\n\nLoading data...')
    startT_data = time.time()

    # Read data
    events = []
    for fname in inputs:
        print(f"Loading data from {fname} ...")
        events.extend(read_file(fname))

    # Define plotter
    plotter = Plotter(print_dir=outDir, end_name=end_name)

    dataset = FeatureDataset(events)
    val_size = 200000
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    print('\n\nTrain size:', train_size)
    print('Test size:', val_size)

    train_loader = DataLoader(train_set, batch_size=batchSize, num_workers=4, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batchSize, num_workers=4, shuffle=False)

    X_sample = next(iter(train_loader))
    print('X_sample:', X_sample.shape)  # e.g. torch.Size([32, 6])

    endT_data = time.time()
    print(f'Loading data took {endT_data - startT_data:.2f}s \n\n')

    # Initialize model, assign seq_len attribute
    if args.d_model % args.nhead != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by nhead ({args.nhead})")

    model = TransformerAutoencoder(
        seq_len=X_sample.shape[1],
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        lr=args.lr
    )

    loss_tracker = LossTracker()

    if doTraining:
        # Device selection
        # --------------------
        if args.device == "cpu":
            accelerator = "cpu"
            devices = 1

        elif args.device == "gpu":
            if torch.cuda.is_available():
                accelerator = "gpu"
                devices = 1
            else:
                print("GPU requested but not available. Falling back to CPU.")
                accelerator = "cpu"
                devices = 1

        elif args.device == "auto":
            if torch.cuda.is_available():
                accelerator = "gpu"
                devices = "auto"  # use all visible GPUs
            else:
                accelerator = "cpu"
                devices = 1

        else:
            raise ValueError(f"Unknown device option: {args.device}")

        print(f"Using accelerator={accelerator}, devices={devices}")

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            strategy="ddp" if accelerator == "gpu" else "auto",  # auto picks the right thing for 1 GPU vs multi-GPU
            max_epochs=maxEpochs,
            enable_progress_bar=True,
            log_every_n_steps=1000,
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            callbacks=[loss_tracker]
        )

        print('\n\nTraining...')
        startT_train = time.time()
        trainer.fit(model, train_loader, val_loader)
        endT_train = time.time()

        print(f'Training took {(endT_train - startT_train) / 60:.2f} minutes \n\n')

        plotter.plotTrainLoss(loss_tracker)

        # Save the model
        model.to("cpu")
        torchscript_model = torch.jit.script(model)
        torchscript_model.save(f"{outDir}/tmae_{end_name}.pt")

    # Load the model and run inference
    if doTraining:
        model = torch.jit.load(f"{outDir}/tmae_{end_name}.pt")
    else:
        model = torch.jit.load(f"nets/tmae_default.pt")
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
