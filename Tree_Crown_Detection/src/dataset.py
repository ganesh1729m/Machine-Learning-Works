import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor
from PIL import Image
import xml.etree.ElementTree as ET
from tqdm import tqdm
import random


# Map class-names to integer labels
LABEL_MAP = {"Tree": 1, "Larch": 2, "Other": 3}  # background = 0

class TreeCrownDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, training_dir, evaluation_dir, transforms=None):
        print()
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.training_dir = training_dir
        self.evaluation_dir = evaluation_dir
        self.transforms = transforms or ToTensor()
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(".tif")]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename)
        image = Image.open(img_path).convert("RGB")
        # print(img_filename)

        # 1) Strip ".tif" from filename
        name_noext = img_filename[:-4]

        # 1a) Try "as-is" under training_dir / evaluation_dir
        base_name = name_noext
        crop_index = 0
        is_cropped = False

        full_img_path = os.path.join(self.training_dir, base_name + ".tif")
        if not os.path.exists(full_img_path):
            full_img_path = os.path.join(self.evaluation_dir, base_name + ".tif")
        if os.path.exists(full_img_path):
            # Found full image directly
            is_cropped = False
            crop_index = 0
        else:
            # 1b) Strip trailing "_<number>"
            parts = name_noext.rsplit("_", 1)
            if len(parts) != 2 or not parts[1].isdigit():
                raise FileNotFoundError(
                    f"Cannot find full image for '{img_filename}', and filename does not end with '_<number>'."
                )
            base_name = parts[0]
            crop_index = int(parts[1])
            is_cropped = True

            # Retry lookup for base_name
            full_img_path = os.path.join(self.training_dir, base_name + ".tif")
            if not os.path.exists(full_img_path):
                full_img_path = os.path.join(self.evaluation_dir, base_name + ".tif")
            if not os.path.exists(full_img_path):
                raise FileNotFoundError(f"After stripping '_<number>', could not find full image: {base_name}.tif")

        # Load full image to get dimensions
        full_img = Image.open(full_img_path)
        full_width, full_height = full_img.size

        # 2) Compute (crop_x, crop_y) if is_cropped, else (0, 0)
        if is_cropped:
            P, O = 256, 64
            S = P - O
            x_offsets = list(range(0, full_width - P + 1, S))
            last_x = full_width - P
            if not x_offsets or x_offsets[-1] != last_x:
                x_offsets.append(last_x)
            y_offsets = list(range(0, full_height - P + 1, S))
            last_y = full_height - P
            if not y_offsets or y_offsets[-1] != last_y:
                y_offsets.append(last_y)

            crops = []
            idx_ctr = 0
            for x0 in x_offsets:
                for y0 in y_offsets:
                    crops.append((idx_ctr, x0, y0))
                    idx_ctr += 1
            if crop_index < 0 or crop_index >= len(crops):
                raise ValueError(f"crop_index {crop_index} is out of range [0, {len(crops)-1}]")
            _, crop_x, crop_y = crops[crop_index]
        else:
            crop_x, crop_y = 0, 0

        # 3) Parse annotation XML for base_name
        xml_path = os.path.join(self.annotation_dir, base_name + ".xml")
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Annotation not found: {xml_path}")

        boxes = []
        labels = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            # Try <tree> first, then <name>, default to "Other"
            text = None
            tree_node = obj.find("tree")
            if tree_node is not None and tree_node.text and tree_node.text.strip():
                text = tree_node.text.strip()
            else:
                name_node = obj.find("name")
                if name_node is not None and name_node.text and name_node.text.strip():
                    text = name_node.text.strip()
            cls_name = text.title() if text else "Other"
            label = LABEL_MAP.get(cls_name, LABEL_MAP["Other"])
        
            bnd = obj.find("bndbox")
            xmin = int(float(bnd.find("xmin").text))
            ymin = int(float(bnd.find("ymin").text))
            xmax = int(float(bnd.find("xmax").text))
            ymax = int(float(bnd.find("ymax").text))


            if is_cropped:
                # Skip if box fully outside this 256×256 tile
                if xmax <= crop_x or xmin >= crop_x + 256:
                    continue
                if ymax <= crop_y or ymin >= crop_y + 256:
                    continue
                # Clip and shift into crop coords
                clipped_xmin = max(xmin, crop_x) - crop_x
                clipped_ymin = max(ymin, crop_y) - crop_y
                clipped_xmax = min(xmax, crop_x + 256) - crop_x
                clipped_ymax = min(ymax, crop_y + 256) - crop_y
                # Skip degenerate boxes
                if (clipped_xmax - clipped_xmin) <= 0 or (clipped_ymax - clipped_ymin) <= 0:
                    continue
                boxes.append([clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax])
                labels.append(label)
            else:
                # Skip degenerate full-image boxes
                if (xmax - xmin) <= 0 or (ymax - ymin) <= 0:
                    continue
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(label)

        # Convert to tensors (ensure correct shape if empty)
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }

        image = self.transforms(image)
        return image, target

