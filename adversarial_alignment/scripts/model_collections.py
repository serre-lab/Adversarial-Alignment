class Models:
    def __init__(self):
        self.model_families = [
            "vit", "vgg", "resnet", "efficientnet", "convnext", "mobilenet",
            "inception", "densenet", "regnet", "xception", "mobilevit",
            "swin", "mixnet", "dpn", "darknet", "maxvit", "beit", "vit_clip",
            "volo", "deit", "nfnet", "cait", "xcit", "tinynet", "lcnet", "dla", 
            "mnasnet", "coatnet", "csp", 
            "0", "1", "2", "3", "4",
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

        return []

if __name__ == "__main__":
    model = Models()
    cnt = 0
    for m in model.model_families:
        print(m)
        cnt += model.get_num_models(m)
    print(cnt)


