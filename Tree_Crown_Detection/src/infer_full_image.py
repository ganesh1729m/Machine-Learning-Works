import os
from PIL import Image
import torch
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import xml.etree.ElementTree as ET

from model_utils import soft_nms, merge_overlapping_boxes, prune_overlaps_keep_all, INV_LABELS


def visualize_full_image_pruned(img_path: str,
                                annotation_dir: str,
                                model: torch.nn.Module,
                                P: int = 256,
                                O: int = 96,
                                score_thresh: float = 0.6,
                                soft_iou: float = 0.3,
                                prune_iou: float = 0.1) -> None:
    """
    Perform tiled inference on a full-resolution image, post-process, and visualize results.

    Args:
        img_path: Path to the full-resolution RGB image.
        annotation_dir: Directory containing corresponding XML annotations.
        model: Trained Faster R-CNN model.
        P: Patch size (px).
        O: Overlap between patches (px).
        score_thresh: Confidence threshold for initial filtering.
        soft_iou: IoU threshold for Soft-NMS.
        prune_iou: IoU threshold for final Hard-NMS pruning.
    """
    device = next(model.parameters()).device
    model.eval()
    to_tensor = ToTensor()

    # Load full image
    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    S = P - O

    # Collect detections from each tile
    all_boxes, all_scores, all_labels = [], [], []
    x_offsets = list(range(0, W - P + 1, S)) + ([W - P] if (W - P) % S else [])
    y_offsets = list(range(0, H - P + 1, S)) + ([H - P] if (H - P) % S else [])

    with torch.no_grad():
        for x0 in x_offsets:
            for y0 in y_offsets:
                crop = img.crop((x0, y0, x0 + P, y0 + P))
                inp = to_tensor(crop).to(device).unsqueeze(0)
                out = model(inp)[0]
                mask = out['scores'] > score_thresh
                b, s, l = out['boxes'][mask].cpu(), out['scores'][mask].cpu(), out['labels'][mask].cpu()
                if b.numel() == 0:
                    continue
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                all_boxes.append(b)
                all_scores.append(s)
                all_labels.append(l)

    if not all_boxes:
        print(f"No detections in {os.path.basename(img_path)}")
        return

    # Concatenate
    boxes = torch.cat(all_boxes)
    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    # Soft-NMS per class
    refined_boxes, refined_scores, refined_labels = [], [], []
    for cls in labels.unique().tolist():
        idxs = (labels == cls).nonzero(as_tuple=True)[0]
        b_cls, s_cls = boxes[idxs], scores[idxs]
        keep_idxs = soft_nms(b_cls, s_cls, iou_thresh=soft_iou)
        if keep_idxs:
            refined_boxes.append(b_cls[keep_idxs])
            refined_scores.append(s_cls[keep_idxs])
            refined_labels.extend([cls] * len(keep_idxs))

    if not refined_boxes:
        print("No boxes after Soft-NMS")
        return

    boxes_sn = torch.cat(refined_boxes)
    scores_sn = torch.cat(refined_scores)
    labels_sn = torch.tensor(refined_labels, dtype=torch.int64)

    # Merge overlapping boxes
    boxes_merged, scores_merged, labels_merged = merge_overlapping_boxes(
        boxes_sn, scores_sn, labels_sn, iou_thresh=soft_iou
    )

    # Final pruning
    boxes_pr, scores_pr, labels_pr = prune_overlaps_keep_all(
        boxes_merged, scores_merged, labels_merged, iou_thresh=prune_iou
    )

    # Sort by score
    order = scores_pr.argsort(descending=True)
    boxes_pr, scores_pr, labels_pr = boxes_pr[order], scores_pr[order], labels_pr[order]

    # Load ground truth
    base = os.path.splitext(os.path.basename(img_path))[0]
    gt_boxes, gt_labels = [], []
    xml_path = os.path.join(annotation_dir, base + ".xml")
    if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            name_node = obj.find("tree") or obj.find("name")
            cls_name = (name_node.text.strip().title() if name_node is not None else "Other")
            bnd = obj.find("bndbox")
            xmin = float(bnd.find("xmin").text)
            ymin = float(bnd.find("ymin").text)
            xmax = float(bnd.find("xmax").text)
            ymax = float(bnd.find("ymax").text)
            if xmax > xmin and ymax > ymin:
                gt_boxes.append((xmin, ymin, xmax, ymax))
                gt_labels.append(cls_name)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)

    # Plot GT in green
    for (xmin, ymin, xmax, ymax), cls_name in zip(gt_boxes, gt_labels):
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       edgecolor='lime', facecolor='none', lw=2))
        ax.text(xmin, ymin - 5, cls_name,
                color='white', fontsize=8,
                bbox=dict(facecolor='lime', pad=1))

    # Plot Predictions in red
    for (xmin, ymin, xmax, ymax), lbl, sc in zip(boxes_pr.tolist(), labels_pr.tolist(), scores_pr.tolist()):
        cls_name = INV_LABELS.get(lbl, str(lbl))
        ax.add_patch(patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       edgecolor='red', facecolor='none', lw=2))
        ax.text(xmin, ymin - 5, f"{cls_name} {sc:.2f}",
                color='white', fontsize=8,
                bbox=dict(facecolor='red', pad=1))

    ax.set_title(os.path.basename(img_path))
    ax.axis('off')
    plt.show()


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 3:
        print("Usage: python infer_full_image.py <image_path> <annotation_dir>")
        sys.exit(1)
    image_path = sys.argv[1]
    ann_dir = sys.argv[2]
    # Load your model here (e.g., via load_backbone_and_replace_head)
    # model = load_backbone_and_replace_head(...)
    visualize_full_image_pruned(image_path, ann_dir, model)
