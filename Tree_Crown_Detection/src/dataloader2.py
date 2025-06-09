import os
import random
import torch
from torch.utils.data import Subset, ConcatDataset, DataLoader
from torchvision.transforms import ToTensor
from dataset import TreeCrownDataset  # your TreeCrownDataset implementation

# Reproducibility
random.seed(42)

# Paths (we use the train folder here to mix Tree vs Larch/Other)
base_dir       = "/kaggle/input/tree-crown-dataset/neon-dataset/train"
annotation_dir = "/kaggle/input/tree-crown-dataset/neon-dataset/annotations"
training_dir   = "/kaggle/input/tree-crown-dataset/neon-dataset/training/RGB"
evaluation_dir = "/kaggle/input/tree-crown-dataset/neon-dataset/evaluation/RGB"

# Base dataset (all crops in train/)
val_ds = TreeCrownDataset(
    image_dir=base_dir,
    annotation_dir=annotation_dir,
    training_dir=training_dir,
    evaluation_dir=evaluation_dir,
    transforms=ToTensor()
)

# Define prefixes for stratification
larch_other_prefixes = ("B01", "B02", "B03", "B07")
tree_prefixes        = ("2018", "2019")

# Collect indices by class
larch_other_idxs = [i for i, fn in enumerate(val_ds.image_files) if fn.startswith(larch_other_prefixes)]
tree_idxs        = [i for i, fn in enumerate(val_ds.image_files) if fn.startswith(tree_prefixes)]

# Create class‐specific subsets
larch_other_ds = Subset(val_ds, larch_other_idxs)
tree_ds        = Subset(val_ds, tree_idxs)

# Concatenate to enforce one-to-one ratio
combined_ds = ConcatDataset([tree_ds, larch_other_ds])

# Subsample for training and validation
n_train = min(5000, len(combined_ds))
n_val   = min(100,  len(combined_ds))
train_indices = random.sample(range(len(combined_ds)), k=n_train)
val_indices   = random.sample(range(len(combined_ds)),   k=n_val)

train_dataset = Subset(combined_ds, train_indices)
val_dataset   = Subset(combined_ds,   val_indices)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Collate fn
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

# Sanity check
print(f"Loader 2 → Tree samples: {len(tree_ds)}, Larch/Other samples: {len(larch_other_ds)}")
print(f" → Combined dataset: {len(combined_ds)}")
print(f" → Train batches   : {len(train_loader)}")
print(f" → Val batches     : {len(val_loader)}")
