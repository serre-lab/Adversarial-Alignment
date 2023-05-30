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

'''
class Dataset:
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.AUTO = tf.data.AUTOTUNE

        self._feature_description = {
            "image"       : tf.io.FixedLenFeature([], tf.string, default_value=''),
            "heatmap"     : tf.io.FixedLenFeature([], tf.string, default_value=''),
            "label"       : tf.io.FixedLenFeature([], tf.int64, default_value=0),
        }

    def parse_prototype(self, prototype, training=False):
        data    = tf.io.parse_single_example(prototype, self._feature_description)

        image   = tf.io.decode_raw(data['image'], tf.float32)
        image   = tf.reshape(image, (224, 224, 3))
        image   = tf.cast(image, tf.float32)

        heatmap = tf.io.decode_raw(data['heatmap'], tf.float32)
        heatmap = tf.reshape(heatmap, (224, 224, 1))

        label   = tf.cast(data['label'], tf.int32)
        label   = tf.one_hot(label, 1_000)

        return image, heatmap, label

    def get_dataset(self, batch_size, training=False):
        deterministic_order = tf.data.Options()
        deterministic_order.experimental_deterministic = True

        dataset = tf.data.TFRecordDataset([self.data_path], num_parallel_reads=self.AUTO)
        dataset = dataset.with_options(deterministic_order) 
        
        dataset = dataset.map(self.parse_prototype, num_parallel_calls=self.AUTO)
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(self.AUTO)

        return dataset
'''
'''
class Models:
    def __init__(self):
        self.model_families = [
            "vit", "vgg", "resnet", "efficientnet", "convnext", "mobilenet",
            "inception", "densenet", "regnet", "xception", "mobilevit",
            "swin", "mixnet", "dpn", "darknet", "maxvit", "beit", "vit_clip",
            "volo", "deit", "nfnet", "cait", "xcit", "tinynet", "lcnet", "dla", 
            "mnasnet", "coatnet", "csp", 
            "resize_group_0", "resize_group_1", "resize_group_2", "resize_group_3", "resize_group_4",
            "test0", "test1", "test2", "paper2018"
        ]

    def get_num_models(self, name):
        
        return len(self.get_model_families(name))

    def get_model_families(self, name):

        if name == "vit":
            return [
                'vit_base_patch8_224', 'vit_base_patch16_224',  'vit_base_patch32_224',
                'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_small_patch32_224', 'vit_large_patch16_224', 
            ]

        if name == "vgg":
            return ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']

        if name == "resnet":
            return [
                'resnet18', 'resnet26', 'resnet34', 'resnet50', 'resnet101', 
                'resnet152', 'resnetv2_50', 'resnetv2_101']

        if name == "efficientnet":
            return ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4']

        if name == "convnext":
            return ['convnext_base', 'convnext_tiny', 'convnext_small', 'convnext_large', "convnext_xlarge_in22ft1k"]

        if name == "mobilenet":
            return [
                'mobilenetv2_050', 'mobilenetv2_100', 'mobilenetv2_110d', 'mobilenetv2_120d', 'mobilenetv2_140', 
                'mobilenetv3_large_100', 'mobilenetv3_rw', 'mobilenetv3_small_050', 'mobilenetv3_small_075', 'mobilenetv3_small_100']

        if name == "inception":
            return ['inception_resnet_v2', 'inception_v3', 'inception_v4']

        if name == "densenet":
            return ['densenet121', 'densenet161', 'densenet169', 'densenet201']

        if name == "regnet":
            return [
                'regnetv_040', 'regnetv_064', 
                'regnetx_002', 'regnetx_040', 'regnetx_064', 'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320', 
                'regnety_002', 'regnety_040', 'regnety_064', 'regnety_080', 'regnety_120', 'regnety_160', 'regnety_320', 
                'regnetz_040', 'regnetz_040h', 'regnetz_d8', 'regnetz_d8_evos', 'regnetz_d32', 'regnetz_e8']

        if name == "xception":
            return ['xception', 'xception41', 'xception65', 'xception71']

        if name == "mixnet":
            return 'mixnet_l', 'mixnet_m', 'mixnet_s', 'mixnet_xl'

        if name == "swin":
            return [
                'swin_tiny_patch4_window7_224', 'swin_small_patch4_window7_224', 'swin_base_patch4_window7_224', 'swin_large_patch4_window7_224', 
                'swin_s3_base_224', 'swin_s3_small_224', 'swin_s3_tiny_224']

        if name == "mobilevit":
            return [
                'mobilevit_s', 'mobilevit_xs', 'mobilevit_xxs', 'mobilevitv2_050', 'mobilevitv2_075', 
                'mobilevitv2_100', 'mobilevitv2_125', 'mobilevitv2_150', 'mobilevitv2_175', 'mobilevitv2_200']

        if name == "dpn":
            return ['dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131']

        if name == "darknet":
            return ["darknet53"]
            
        if name == "maxvit":
            return ['maxvit_rmlp_small_rw_224', 'maxvit_tiny_rw_224']

        if name == "volo":
            return ['volo_d1_224', 'volo_d2_224', 'volo_d3_224','volo_d4_224', 'volo_d5_224']

        if name == "deit":
            return [
                'deit3_base_patch16_224', 'deit3_base_patch16_224_in21ft1k', 'deit3_huge_patch14_224', 
                'deit3_huge_patch14_224_in21ft1k', 'deit3_large_patch16_224', 'deit3_large_patch16_224_in21ft1k', 
                'deit3_medium_patch16_224', 'deit3_medium_patch16_224_in21ft1k', 'deit3_small_patch16_224', 
                'deit3_small_patch16_224_in21ft1k', 'deit_base_distilled_patch16_224', 'deit_base_patch16_224', 
                'deit_small_distilled_patch16_224', 'deit_small_patch16_224', 'deit_tiny_distilled_patch16_224', 
                'deit_tiny_patch16_224']

        if name == "nfnet":
            return [
                'dm_nfnet_f0', 'dm_nfnet_f1', 'dm_nfnet_f2', 'dm_nfnet_f3', 'dm_nfnet_f4', 'dm_nfnet_f5', 'dm_nfnet_f6',
                'eca_nfnet_l0', 'eca_nfnet_l1', 'eca_nfnet_l2', 'nfnet_l0']

        if name == "cait":
            return ['cait_s24_224', 'cait_xxs24_224', 'cait_xxs36_224']

        if name == "xcit":
            return ['xcit_large_24_p8_224', 'xcit_large_24_p16_224', 'xcit_medium_24_p8_224', 'xcit_medium_24_p16_224', 
            'xcit_nano_12_p8_224', 'xcit_nano_12_p16_224', 'xcit_small_12_p8_224', 'xcit_small_12_p16_224', 'xcit_small_24_p8_224', 
            'xcit_small_24_p16_224', 'xcit_tiny_12_p8_224', 'xcit_tiny_12_p16_224', 'xcit_tiny_24_p8_224', 'xcit_tiny_24_p16_224']

        if name == "tinynet":
            return ['tinynet_a', 'tinynet_b', 'tinynet_c', 'tinynet_d', 'tinynet_e']

        if name == "lcnet":
            return ['lcnet_050', 'lcnet_075', 'lcnet_100']

        if name == "dla":
            return [
                'dla34', 'dla46_c', 'dla46x_c', 'dla60', 'dla60_res2net', 'dla60_res2next', 
                'dla60x', 'dla60x_c', 'dla102', 'dla102x', 'dla102x2', 'dla169']

        if name == "mnasnet":
            return ['mnasnet_100', 'mnasnet_small', 'semnasnet_075', 'semnasnet_100']

        if name == "coatnet":
            return ['coat_lite_mini', 'coat_lite_small', 'coat_lite_tiny', 'coat_mini', 'coat_tiny', 'coatnet_0_rw_224', 
            'coatnet_1_rw_224', 'coatnet_bn_0_rw_224', 'coatnet_nano_rw_224', 'coatnet_rmlp_1_rw_224', 
            'coatnet_rmlp_2_rw_224', 'coatnet_rmlp_nano_rw_224', 'coatnext_nano_rw_224']

        if name == "csp":
            return ['cspdarknet53', 'cspresnet50', 'cspresnext50']

        if name == "resize_group_0":
            return [
                'maxvit_nano_rw_256', 'maxvit_rmlp_nano_rw_256', 'maxvit_rmlp_pico_rw_256', 'maxvit_rmlp_tiny_rw_256',
                'deit3_base_patch16_384', 'deit3_base_patch16_384_in21ft1k', 'deit3_large_patch16_384', 'deit3_large_patch16_384_in21ft1k', 
                'deit3_small_patch16_384', 'deit3_small_patch16_384_in21ft1k', 'deit_base_distilled_patch16_384', 'deit_base_patch16_384']

        if name == "resize_group_1":
            return [
                'swin_base_patch4_window12_384', 'swin_large_patch4_window12_384', 'swinv2_base_window8_256', 
                'swinv2_base_window16_256', 'swinv2_cr_small_224', 'swinv2_cr_small_ns_224', 'swinv2_cr_tiny_ns_224', 
                'swinv2_small_window8_256', 'swinv2_small_window16_256', 'swinv2_tiny_window8_256', 
                'swinv2_tiny_window16_256']

        if name == "resize_group_2":
            return ['volo_d1_384', 'volo_d2_384', 'volo_d3_448', 'volo_d4_448', 'volo_d5_448', 'volo_d5_512']

        if name == "resize_group_3":
            return ['cait_m36_384', 'cait_m48_448', 'cait_s24_384', 'cait_s36_384', 'cait_xs24_384', 'cait_xxs24_384','cait_xxs36_384']

        if name == "resize_group_4":
            return [
                'xcit_large_24_p8_384_dist', 'xcit_large_24_p16_384_dist', 'xcit_medium_24_p8_384_dist', 'xcit_medium_24_p16_384_dist', 
                'xcit_nano_12_p8_384_dist', 'xcit_nano_12_p16_384_dist',  'xcit_small_12_p8_384_dist',  'xcit_small_12_p16_384_dist', 
                'xcit_small_24_p8_384_dist', 'xcit_small_24_p16_384_dist', 'xcit_tiny_12_p8_384_dist', 'xcit_tiny_12_p16_384_dist', 
                'xcit_tiny_24_p8_384_dist', 'xcit_tiny_24_p16_384_dist']

        # if name == "test0":
        #     return [
        #         "tinynet_e", "mobilenetv3_small_050", "resnet50", "resnet101",
        #         "vit_base_patch8_224"]   

        # if name == "test1":
        #     return [
        #         "convnext_xlarge_in22ft1k"] 

        # if name == "test2":
        #     return [
        #         "cait_m36_384"
        #     ]  
        
        # if name == "paper2018":
        #     return [
        #         'resnet50', 'resnet101', 'resnet152', 'inception_resnet_v2', 'inception_v3', 
        #         'inception_v4', 'vgg16', 'vgg19', 'mobilenetv2_050', 'mobilenetv2_100',
        #         'densenet121', 'densenet161', 'densenet169', 
        #     ]

        return []
'''
'''
class Parameters:
    def __init__(self):
        # FGSM, FFGSM
        self.epsilons = [ 
            0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006, 0.00007, 0.00008, 0.00009,
            0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009,
            0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
            0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 
            0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0,
            # 1.2, 1.4, 1.6, 1.8, 2.0,
            # 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
        ]
        self.alpha = 10/255

        # C&W
        self.consts = [ 
            0.01, 0.025, 0.05, 0.075, 0.1,
            0.25, 0.5, 0.75, 1,
            1.5, 2.0, 2.5, 3.0
        ]

        self.const = 0.1

        self.kappa = 0

        self.steps = 10

        self.lr = 0.0075

        self.lrs = [
            0.001, 0.002, 0.004, 0.008, 
            0.01, 0.02, 0.04, 0.08,
            0.1, 0.2, 0.4, 0.8, 1
        ]

        self.overshoot = 0.02

    def get_alpha(self):
        return self.alpha

    def get_epsilons(self):
        return self.epsilons

    def get_consts(self):
        return self.consts
    
    def get_const(self):
        return self.const

    def get_kappa(self):
        return self.kappa

    def get_steps(self):
        return self.steps

    def get_lr(self):
        return self.lr

    def get_lrs(self):
        return self.lrs
'''
if __name__ == "__main__":
    pass
