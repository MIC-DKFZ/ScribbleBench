import numpy as np
import os
from tqdmp import tqdmp
from pathlib import Path
import argparse
from medvol import MedVol


def evaluate(gt_dir, pred_dir, num_classes, num_processes=None):    
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)
    names_gt = [path.name[:-7] for path in Path(gt_dir).rglob("*.nii.gz")]
    names_pred = [path.name[:-7] for path in Path(pred_dir).rglob("*.nii.gz")]

    if set(names_gt) != set(names_pred):
        raise RuntimeError(f"The set of GT segmentations is different to the set of predictions. Do you have missing predictions?")

    if isinstance(num_processes, str):
        num_processes = int(num_processes)

    dice_scores = tqdmp(evaluate_prediction, names_gt, num_processes, gt_dir=gt_dir, pred_dir=pred_dir, num_classes=num_classes, desc="Evaluating")

    mean_dice_score = float(np.mean(dice_scores))

    print(f"Mean Dice Score: {mean_dice_score}")
        

def evaluate_prediction(name, gt_dir, pred_dir, num_classes, foreground_only=True):
    gt_filepath = gt_dir / f"{name}.nii.gz"
    pred_filepath = pred_dir / f"{name}.nii.gz"
    if not os.path.exists(pred_filepath):
        raise RuntimeError(f"Prediction ({name}) does not exist.")
    gt = MedVol(str(gt_filepath)).array
    pred = MedVol(str(pred_filepath)).array
    gt = np.rint(np.asarray(gt)).astype(np.uint8)
    pred = np.rint(np.asarray(pred)).astype(np.uint8)
    if gt.shape != pred.shape:
        raise RuntimeError("Prediction and GT do not have the same shape.")
    gt = gt.flatten()
    pred = pred.flatten()
    dice_score = comp_dice(pred, gt, num_classes, foreground_only)
    return dice_score


def comp_dice(pred, gt, num_classes, foreground_only=True, ignore_mask=None):
    class_labels = list(range(num_classes))
    if foreground_only:
        class_labels = class_labels[1:]

    dice_score = []
    for label in class_labels:
        tp, fp, fn, tn = compute_tp_fp_fn_tn(gt == label, pred == label, ignore_mask)
        if tp + fp + fn != 0:
            class_dice_score = float(2 * tp / (2 * tp + fp + fn))
        else:
            class_dice_score = np.nan
        dice_score.append(class_dice_score)

    dice_score = np.nanmean(dice_score)
    dice_score = float(dice_score)
    return dice_score


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-gt', "--gt_dir", required=True, help="Path to the dense GT segmentations folder.")
    parser.add_argument('-pred', "--pred_dir", required=True, help="Path to the dense prediction segmentation folder.")
    parser.add_argument('-l', "--num_labels", required=True, type=int, help="The number of segmentation labels.")
    parser.add_argument('-p', "--processes", required=False, default=None, help="Number of multiprocessing processes.")
    args = parser.parse_args()

    evaluate(args.gt_dir, args.gt_dir, args.num_labels)
