import tensorflow as tf
import torch
import numpy as np
from scipy.stats import spearmanr
from scipy.ndimage import gaussian_filter
import csv
import os

def postprocess_mask(mask):
    p = 0.99
    mask = np.clip(mask, np.percentile(img, p), np.percentile(mask, 100-p))
    smooth = 2
    mask = gaussian_filter(mask, sigma = 2)
    return mask

def spearman_correlation(a, b):
    """
    Computes the Spearman correlation between two sets of heatmaps.
    Parameters
    ----------
    heatmaps_a
        First set of heatmaps.
        Expected shape (N, W, H).
    heatmaps_b
        Second set of heatmaps.
        Expected shape (N, W, H).
    Returns
    -------
    spearman_correlations
        Array of Spearman correlation score between the two sets of heatmaps.
    """
    assert a.shape == b.shape, "The two sets of images must" \
                                                 "have the same shape."
    assert len(a.shape) == 4, "The two sets of heatmaps must have shape (1, 1, W, H)."

    rho, _ = spearmanr(a.flatten(), b.flatten())
    return rho

def tf2torch(t): # a batch of image tensors (N, H, W, 3)
    t = tf.cast(t, tf.float32).numpy()
    if t.shape[-1] in [1, 3]:
        t = torch.from_numpy(t.transpose(0, 3, 1, 2)) # torch.from_numpy(np_array.transpose(0, 3, 1, 2)) 
        return t
    return torch.from_numpy(t) # (N, 3, H, W)

# Image normalization
def img_normalize(imgs):
    imgs = imgs - imgs.min()
    imgs = imgs / imgs.max()
    return imgs

def linf_loss(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=np.inf)

def l2_loss(x, y):
    return np.linalg.norm(x.flatten() - y.flatten(), ord=2)

def write_csv_all(record, path):
    header = ['model', 'img', 'label', 'pred', 'eps', 'l2', 'linf', 'spearman']
    file_exists = os.path.isfile(path)

    with open(path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(record)

def write_csv_avg(record, path):
    header = ['model', 'num_correct', 
              'avg_eps', 'std_eps', 
              'avg_l2', 'std_l2', 
              'avg_linf', 'std_linf', 
              'avg_spearman', 'std_spearman']
    file_exists = os.path.isfile(path)

    with open(path, mode='a+', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(record)
if __name__ == "__main__":
    pass
