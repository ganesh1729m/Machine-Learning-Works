import os
import torch
import random
import numpy as np
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
from torchvision.ops import box_iou, batched_nms
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET
from tqdm import tqdm
from typing import List

from model_utils import soft_nms, merge_overlapping_boxes, prune_overlaps_keep_all, LABEL_MAP, INV_LABELS

# --- Inference pipeline ---
def infer_full_image(img: Image.Image, model: torch.nn.Module,
                     P=256, O=96,
                     score_thresh=0.8,
                     soft_iou=0.3,
                     merge_iou=0.3,
                     prune_iou=0.1):
    dev = next(model.parameters()).device
    model.eval()
    to_tensor = ToTensor()

    W, H = img.size
    S = P - O
    all_b, all_s, all_l = [], [], []

    x_off = list(range(0, W-P+1, S)) + ([W-P] if (W-P)%S else [])
    y_off = list(range(0, H-P+1, S)) + ([H-P] if (H-P)%S else [])

    with torch.no_grad():
        for x0 in x_off:
            for y0 in y_off:
                patch = img.crop((x0, y0, x0+P, y0+P))
                inp = to_tensor(patch).to(dev).unsqueeze(0)
                out = model(inp)[0]
                m = out['scores'] > score_thresh
                b, s, l = out['boxes'][m].cpu(), out['scores'][m].cpu(), out['labels'][m].cpu()
                if b.numel() == 0:
                    continue
                b[:, [0, 2]] += x0
                b[:, [1, 3]] += y0
                all_b.append(b)
                all_s.append(s)
                all_l.append(l)

    if not all_b:
        return torch.zeros((0,4)), torch.tensor([]), torch.tensor([])

    boxes = torch.cat(all_b)
    scores = torch.cat(all_s)
    labels = torch.cat(all_l)

    # Soft-NMS per class
    bs, ss, ls = [], [], []
    for cls in labels.unique().tolist():
        idxs = (labels == cls).nonzero(as_tuple=True)[0]
        keep = soft_nms(boxes[idxs], scores[idxs], iou_thresh=soft_iou)
        if keep:
            bs.append(boxes[idxs][keep])
            ss.append(scores[idxs][keep])
            ls.extend([cls] * len(keep))
    if not bs:
        return torch.zeros((0,4)), torch.tensor([]), torch.tensor([])

    boxes = torch.cat(bs)
    scores = torch.cat(ss)
    labels = torch.tensor(ls)

    # Merge overlapping boxes
    boxes, scores, labels = merge_overlapping_boxes(boxes, scores, labels, iou_thresh=merge_iou)

    # Final prune
    boxes, scores, labels = prune_overlaps_keep_all(boxes, scores, labels, iou_thresh=prune_iou)

    return boxes, scores, labels

# --- Evaluation on list of full-image files ---
def evaluate_full_images(image_paths: List[str], annotation_dir: str, model: torch.nn.Module, device: torch.device, iou_match: float = 0.5):
    true_all, pred_all = [], []
    iou_list, ssim_list, psnr_list, mse_list = [], [], [], []

    for img_path in tqdm(image_paths, desc="Evaluating Full Images"):
        img = Image.open(img_path).convert("RGB")
        pred_b, _, pred_l = infer_full_image(img, model)

        # Load GT boxes and labels
        base = os.path.splitext(os.path.basename(img_path))[0]
        xmlp = os.path.join(annotation_dir, base + ".xml")
        gt_b, gt_l = [], []
        if os.path.exists(xmlp):
            tree = ET.parse(xmlp)
            for obj in tree.getroot().findall("object"):
                bnd = obj.find("bndbox")
                coords = list(map(float, [bnd.find(x).text for x in ["xmin","ymin","xmax","ymax"]]))
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    gt_b.append(coords)
                    name_node = obj.find("tree") or obj.find("name")
                    raw = (name_node.text.strip() if name_node is not None else None)
                    cls = raw.title() if raw else "Other"
                    gt_l.append(LABEL_MAP.get(cls, 3))
        gt_b = torch.tensor(gt_b) if gt_b else torch.zeros((0,4))
        gt_l = torch.tensor(gt_l) if gt_l else torch.tensor([], dtype=torch.int64)

        # Match & compute IoU-based metrics
        if pred_b.numel() and gt_b.numel():
            ious = box_iou(pred_b, gt_b)
            matched_p, matched_g = set(), set()
            for i in range(pred_b.size(0)):
                max_iou, j = ious[i].max(0)
                if max_iou >= iou_match:
                    true_all.append(gt_l[j].item()); pred_all.append(pred_l[i].item())
                    matched_p.add(i); matched_g.add(j.item()); iou_list.append(max_iou.item())
            # false negatives
            for j in range(gt_b.size(0)):
                if j not in matched_g:
                    true_all.append(gt_l[j].item()); pred_all.append(0)
            # false positives
            for i in range(pred_b.size(0)):
                if i not in matched_p:
                    true_all.append(0); pred_all.append(pred_l[i].item())
        else:
            # all GT are FNs, all preds are FPs
            for j in range(gt_b.size(0)):
                true_all.append(gt_l[j].item()); pred_all.append(0)
            for i in range(pred_b.size(0)):
                true_all.append(0); pred_all.append(pred_l[i].item())

        # Compute mask-based metrics
        w,h = img.size
        gt_mask = Image.new("L", (w,h), 0); pr_mask = Image.new("L", (w,h), 0)
        draw_gt = ImageDraw.Draw(gt_mask); draw_pr = ImageDraw.Draw(pr_mask)
        for cb in gt_b.tolist(): draw_gt.rectangle(cb, fill=255)
        for cb in pred_b.tolist(): draw_pr.rectangle(cb, fill=255)
        A, B = np.array(gt_mask), np.array(pr_mask)
        ssim_list.append(ssim(A, B, data_range=255)); psnr_list.append(psnr(A, B, data_range=255))
        mse_list.append(np.mean((A - B) ** 2))

    # Aggregate metrics
    iou_avg = np.mean(iou_list)
    precision = precision_score(true_all, pred_all, labels=[1,2,3], average="macro", zero_division=0)
    recall = recall_score(true_all, pred_all, labels=[1,2,3], average="macro", zero_division=0)
    accuracy = accuracy_score(true_all, pred_all)
    ssim_avg = np.mean(ssim_list); psnr_avg = np.mean(psnr_list); mse_avg = np.mean(mse_list)

    print("\n--- Full Image Evaluation Metrics ---")
    print(f"IoU (avg):    {iou_avg:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"mAP (approx): {iou_avg:.4f}")
    print(f"SSIM (avg):   {ssim_avg:.4f}")
    print(f"PSNR (avg):   {psnr_avg:.4f}")
    print(f"MSE (avg):    {mse_avg:.4f}")

    # Confusion matrix
    cm = confusion_matrix(true_all, pred_all, labels=[1,2,3])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[INV_LABELS[i] for i in [1,2,3]],
                yticklabels=[INV_LABELS[i] for i in [1,2,3]])
    plt.title("Full Image Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.show()

# If run as script
if __name__ == "__main__":
    # Example usage:
    # python evaluate.py /path/to/full/images /path/to/annotations model_checkpoint.pth
    import sys
    img_dir, ann_dir, ckpt = sys.argv[1:4]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_backbone_and_replace_head(ckpt, num_classes=4, device=device)
    image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.tif')]
    evaluate_full_images(image_paths, ann_dir, model, device)
