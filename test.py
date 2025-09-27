import torch

# Assume you already have the model and data
model = TransformerAutoencoder()
model.eval()

# Simulate a batch of 4 samples, each with 6 features (complete data)
x_full = torch.tensor([
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
    [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
    [0.4, 0.3, 0.2, 0.1, 0.0, 0.9],
], dtype=torch.float32)

# Specify the missing feature index for each sample, e.g., 0th, 2nd, 5th, 1st
mask_idx = torch.tensor([0, 2, 5, 1])

# Call corrupt_input to generate input with missing features
x_corrupted, mask_idx = model.corrupt_input(x_full, mask_idx=mask_idx)

print("Original full data:\n", x_full)
print("Mask indices:\n", mask_idx)
print("Input after masking:\n", x_corrupted.squeeze(-1))

# Pass through the model to predict the missing features
with torch.no_grad():
    y_pred = model(x_corrupted, mask_idx)

print("Model prediction of missing features:\n", y_pred)

# Ground truth of the missing values
y_true = x_full[torch.arange(x_full.size(0)), mask_idx]
print("Ground truth of missing features:\n", y_true)


