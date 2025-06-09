import torch
import numpy as np
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou, batched_nms

# Label maps
LABEL_MAP = {"Tree": 1, "Larch": 2, "Other": 3}
INV_LABELS = {v: k for k, v in LABEL_MAP.items()}


def load_backbone_and_replace_head(checkpoint_path: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    """
    Load backbone weights from an existing checkpoint and reinitialize the box predictor head.

    Args:
        checkpoint_path (str): Path to a .pth checkpoint file
        num_classes (int): Number of classes (including background)
        device (torch.device): Device to map model and tensors

    Returns:
        model (torch.nn.Module): Faster R-CNN model with pretrained backbone and new head
    """
    # Initialize model with pretrained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Extract and load backbone weights
    backbone_state = {k.replace('backbone.', ''): v for k, v in state_dict.items() if k.startswith('backbone.')}
    missing, unexpected = model.backbone.load_state_dict(backbone_state, strict=False)
    print(f"Loaded backbone: {len(missing)} missing, {len(unexpected)} unexpected keys")

    return model.to(device)


def soft_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.3, sigma: float = 0.5, score_thresh: float = 0.001) -> list:
    """
    Perform Soft-NMS on a single class of boxes.
    Returns indices of boxes to keep.
    """
    boxes_np  = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    idxs = list(range(len(scores_np)))
    keep = []
    while idxs:
        i = max(idxs, key=lambda x: scores_np[x])
        if scores_np[i] < score_thresh:
            break
        keep.append(i)
        idxs.remove(i)
        x1, y1, x2, y2 = boxes_np[i]
        area_i = (x2 - x1) * (y2 - y1)
        for j in idxs[:]:
            xx1 = max(x1, boxes_np[j, 0])
            yy1 = max(y1, boxes_np[j, 1])
            xx2 = min(x2, boxes_np[j, 2])
            yy2 = min(y2, boxes_np[j, 3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            area_j = (boxes_np[j, 2] - boxes_np[j, 0]) * (boxes_np[j, 3] - boxes_np[j, 1])
            iou = inter / (area_i + area_j - inter + 1e-6)
            scores_np[j] *= np.exp(-(iou * iou) / sigma)
            if scores_np[j] < score_thresh:
                idxs.remove(j)
    return keep


def merge_overlapping_boxes(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thresh: float = 0.3):
    """
    Merge overlapping boxes of the same class by computing their bounding union and retaining max score.
    Returns merged (boxes, scores, labels).
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    iou_mat = box_iou(boxes, boxes)
    used = set()
    out_boxes, out_scores, out_labels = [], [], []
    for i in range(len(boxes)):
        if i in used:
            continue
        same_cls = [j for j in (iou_mat[i] > iou_thresh).nonzero(as_tuple=True)[0].tolist()
                    if labels[j] == labels[i]]
        for j in same_cls:
            used.add(j)
        grp = boxes[same_cls]
        x_min = grp[:, 0].min().item()
        y_min = grp[:, 1].min().item()
        x_max = grp[:, 2].max().item()
        y_max = grp[:, 3].max().item()
        out_boxes.append([x_min, y_min, x_max, y_max])
        grp_scores = scores[same_cls]
        out_scores.append(grp_scores.max().item())
        out_labels.append(labels[i].item())

    return (
        torch.tensor(out_boxes, dtype=torch.float32, device=boxes.device),
        torch.tensor(out_scores, dtype=torch.float32, device=scores.device),
        torch.tensor(out_labels, dtype=torch.int64, device=labels.device)
    )


def prune_overlaps_keep_all(boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, iou_thresh: float = 0.1):
    """
    Final hard NMS using torchvision.ops.batched_nms
    """
    keep = batched_nms(boxes, scores, labels, iou_threshold=iou_thresh)
    return boxes[keep], scores[keep], labels[keep]
