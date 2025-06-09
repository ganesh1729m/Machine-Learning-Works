import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.ops import box_iou
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr


def evaluate_model(model: torch.nn.Module,
                   dataset,
                   device: torch.device,
                   iou_match: float = 0.5,
                   score_thresh: float = 0.6):
    """
    Evaluate detection performance on cropped images using various metrics.

    Args:
        model: Trained object detection model (Faster R-CNN).
        dataset: PyTorch Dataset yielding (image, target) pairs.
        device: Torch device ('cpu' or 'cuda').
        iou_match: IoU threshold for matching predicted and ground-truth boxes.
        score_thresh: Minimum score for predicted boxes.
    """
    model.eval()

    true_labels, pred_labels = [], []
    iou_scores, ssim_scores, psnr_scores, mse_scores = [], [], [], []

    for img, target in tqdm(dataset, desc="Evaluating Crops"):
        # Move to device
        img_tensor = img.to(device)
        targets = {k: v.to(device) for k, v in target.items()}

        with torch.no_grad():
            out = model([img_tensor])[0]

        # Filter predictions
        keep = out['scores'] > score_thresh
        pred_b = out['boxes'][keep].cpu()
        pred_l = out['labels'][keep].cpu()

        true_b = targets['boxes'].cpu()
        true_l = targets['labels'].cpu()

        # IoU-based matching
        if true_b.numel() and pred_b.numel():
            ious = box_iou(pred_b, true_b)
            matched_p, matched_t = set(), set()
            for i in range(pred_b.size(0)):
                max_iou, j = ious[i].max(0)
                if max_iou >= iou_match:
                    true_labels.append(true_l[j].item())
                    pred_labels.append(pred_l[i].item())
                    matched_p.add(i)
                    matched_t.add(j.item())
                    iou_scores.append(max_iou.item())
            # False negatives
            for j in range(true_b.size(0)):
                if j not in matched_t:
                    true_labels.append(true_l[j].item())
                    pred_labels.append(0)
            # False positives
            for i in range(pred_b.size(0)):
                if i not in matched_p:
                    true_labels.append(0)
                    pred_labels.append(pred_l[i].item())
        else:
            # No overlap: all GT are FN, preds are FP
            for j in range(true_b.size(0)):
                true_labels.append(true_l[j].item()); pred_labels.append(0)
            for i in range(pred_b.size(0)):
                true_labels.append(0); pred_labels.append(pred_l[i].item())

        # Mask-based SSIM, PSNR, MSE
        h, w = img_tensor.shape[1:]
        gt_mask = Image.new('L', (w, h), 0)
        pr_mask = Image.new('L', (w, h), 0)
        draw_gt = ImageDraw.Draw(gt_mask)
        draw_pr = ImageDraw.Draw(pr_mask)
        for box in true_b.tolist():
            draw_gt.rectangle(box, fill=255)
        for box in pred_b.tolist():
            draw_pr.rectangle(box, fill=255)
        A, B = np.array(gt_mask), np.array(pr_mask)
        ssim_scores.append(ssim(A, B, data_range=255))
        psnr_scores.append(psnr(A, B, data_range=255))
        mse_scores.append(np.mean((A - B)**2))

    # Compute classification metrics
    precision = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
    recall    = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
    accuracy  = accuracy_score(true_labels, pred_labels)
    cm        = confusion_matrix(true_labels, pred_labels)

    # Summary
    print("\n--- Crop Evaluation Metrics ---")
    print(f"IoU (avg):    {np.mean(iou_scores):.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall:       {recall:.4f}")
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"mAP (approx): {np.mean(iou_scores):.4f}")
    print(f"SSIM (avg):   {np.mean(ssim_scores):.4f}")
    print(f"PSNR (avg):   {np.mean(psnr_scores):.4f}")
    print(f"MSE (avg):    {np.mean(mse_scores):.4f}")
    print("\nConfusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Crop Confusion Matrix')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.show()
