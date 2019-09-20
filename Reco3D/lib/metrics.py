import numpy as np


def evaluate_voxel_prediction(preds, gt, thresh):
    preds_occupy = preds[:, :, :, 1] >= thresh
    diff = np.sum(np.logical_xor(preds_occupy, gt[:, :, :, 1]))
    intersection = np.sum(np.logical_and(preds_occupy, gt[:, :, :, 1]))
    union = np.sum(np.logical_or(preds_occupy, gt[:, :, :, 1]))
    num_fp = np.sum(np.logical_and(preds_occupy, gt[:, :, :, 0]))  # false positive
    num_fn = np.sum(np.logical_and(np.logical_not(preds_occupy), gt[:, :, :, 0]))  # false negative
    return np.array([diff, intersection, union, num_fp, num_fn])
