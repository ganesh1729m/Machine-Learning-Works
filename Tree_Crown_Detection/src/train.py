# train.py

import os
import re
import random
import torch
import torchvision
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import TreeCrownDataset
from model_utils import load_backbone_and_replace_head

# ----------------------------
# Configuration & Paths
# ----------------------------
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data paths (RGB subfolders)
TRAIN_DIR      = "/kaggle/input/tree-crown-dataset/neon-dataset/train"
VAL_DIR        = "/kaggle/input/tree-crown-dataset/neon-dataset/val"
ANNOT_DIR      = "/kaggle/input/tree-crown-dataset/neon-dataset/annotations"
TRAINING_RGB   = "/kaggle/input/tree-crown-dataset/neon-dataset/training/RGB"
EVAL_RGB       = "/kaggle/input/tree-crown-dataset/neon-dataset/evaluation/RGB"

# Checkpoint directory (for resume)
CHECKPOINT_DIR = "/kaggle/input/tree-crown-paths"
NUM_CLASSES    = 4      # background + 3 tree classes
NUM_EPOCHS     = 50
BATCH_SIZE     = 4
LR             = 0.005
STEP_SIZE      = 3
GAMMA          = 0.1

# ----------------------------
# Prepare Datasets & Loaders
# ----------------------------
full_train_ds = TreeCrownDataset(
    TRAIN_DIR, ANNOT_DIR, TRAINING_RGB, EVAL_RGB, transforms=ToTensor()
)
full_val_ds = TreeCrownDataset(
    VAL_DIR, ANNOT_DIR, TRAINING_RGB, EVAL_RGB, transforms=ToTensor()
)

train_idxs = random.sample(range(len(full_train_ds)), k=min(10000, len(full_train_ds)))
val_idxs   = random.sample(range(len(full_val_ds)),   k=min(100,   len(full_val_ds)))

train_ds = Subset(full_train_ds, train_idxs)
val_ds   = Subset(full_val_ds,   val_idxs)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, collate_fn=collate_fn, persistent_workers=True)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                          num_workers=2, collate_fn=collate_fn, persistent_workers=True)

print(f"Loaded train ({len(train_ds)}) and val ({len(val_ds)}) samples.")

# ----------------------------
# Model, Optimizer, Scheduler
# ----------------------------
# Load pretrained backbone + new head
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.roi_heads.box_predictor = FastRCNNPredictor(
    model.roi_heads.box_predictor.cls_score.in_features,
    NUM_CLASSES
)
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

# ----------------------------
# Resume from Latest Checkpoint
# ----------------------------
def extract_epoch(fn):
    m = re.search(r"epoch(\d+)\.pth$", fn)
    return int(m.group(1)) if m else -1

ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pth")]
start_epoch = 0
if ckpts:
    best = max(ckpts, key=extract_epoch)
    epoch0 = extract_epoch(best)
    path0 = os.path.join(CHECKPOINT_DIR, best)
    model.load_state_dict(torch.load(path0, map_location=device))
    start_epoch = epoch0 + 1
    print(f"Resumed from checkpoint '{best}' (epoch {epoch0})")
else:
    print("No checkpoints found. Starting from scratch.")

# ----------------------------
# Training & Validation Loop
# ----------------------------
for epoch in range(start_epoch, NUM_EPOCHS):
    # Train
    model.train()
    total_train_loss = 0.0
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]", leave=False)
    for images, targets in train_bar:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        train_bar.set_postfix(loss=loss.item())

    lr_scheduler.step()
    avg_train_loss = total_train_loss / len(train_loader)

    # Validation
    model.train()  # keep dropout/batchnorm behavior consistent for loss computation
    total_val_loss = 0.0
    val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val]", leave=False)
    with torch.no_grad():
        for images, targets in val_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            total_val_loss += loss.item()
            val_bar.set_postfix(val_loss=loss.item())

    avg_val_loss = total_val_loss / len(val_loader)

    # Summary
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} → "
          f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Checkpoint
    if (epoch + 1) % 10 == 0 or (epoch + 1) == NUM_EPOCHS:
        ckpt_name = f"fasterrcnn_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_name)
        print(f"Saved checkpoint: {ckpt_name}")
