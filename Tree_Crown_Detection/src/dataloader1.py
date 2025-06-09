import os
import random
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import ToTensor
from dataset import TreeCrownDataset  # your TreeCrownDataset implementation

# Reproducibility
random.seed(42)

# Paths (ensure these point to the RGB subfolders)
train_image_dir  = "/kaggle/input/tree-crown-dataset/neon-dataset/train"
val_image_dir    = "/kaggle/input/tree-crown-dataset/neon-dataset/val"
annotation_dir   = "/kaggle/input/tree-crown-dataset/neon-dataset/annotations"
training_dir     = "/kaggle/input/tree-crown-dataset/neon-dataset/training/RGB"
evaluation_dir   = "/kaggle/input/tree-crown-dataset/neon-dataset/evaluation/RGB"

# Full datasets
full_val_dataset   = TreeCrownDataset(val_image_dir, annotation_dir, training_dir, evaluation_dir, transforms=ToTensor())
full_train_dataset = TreeCrownDataset(train_image_dir, annotation_dir, training_dir, evaluation_dir, transforms=ToTensor())

# Subsample for quick iteration
val_indices   = random.sample(range(len(full_val_dataset)),   k=min(100,   len(full_val_dataset)))
train_indices = random.sample(range(len(full_train_dataset)), k=min(10000, len(full_train_dataset)))

val_dataset   = Subset(full_val_dataset,   val_indices)
train_dataset = Subset(full_train_dataset, train_indices)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collate fn (to batch variable‐length targets)
def collate_fn(batch):
    return tuple(zip(*batch))

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=2,
    collate_fn=collate_fn,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    collate_fn=collate_fn,
    persistent_workers=True
)

print(f"Loader 1 → Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
