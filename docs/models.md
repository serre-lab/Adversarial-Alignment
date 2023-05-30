# Model Information

- In our experiment, 283 models have been tested and the Top-1 ImageNet accuracy for each model refers to [Hugging Face results](https://github.com/huggingface/pytorch-image-models/blob/main/results/results-imagenet.csv).
    - **·** 125 PyTorch CNN models from [timm library](https://timm.fast.ai/)
    - **·** 121 PyTorch ViT models from [timm library](https://timm.fast.ai/)
    - **·** 15 PyTorch ViT/CNN hybrid architectures from [timm library](https://timm.fast.ai/)
    - **·** 14 Tensorflow Harmonized models from [harmonizatin library](https://serre-lab.github.io/Harmonization/)
    - **·** 4 Baseline models
    - **·** 4 models that were trained for robustness to adversarial example

| Architecture | Model        | Versions |
|:------------:|:------------:|:--------:|
|CNN           | VGG          | 8        |
|CNN           | ResNet       | 8        |
|CNN           | EfficientNet | 7        |
|CNN           | ConvNext     | 6        |
|CNN           | MobileNet    | 10       |
|CNN           | Inception    | 3        |
|CNN           | DenseNet     | 4        |
|CNN           | RegNet       | 22       |
|CNN           | Xception     | 4        |
|CNN           | MixNet       | 4        |
|CNN           | DPN          | 6        |
|CNN           | DarkNet      | 1        |
|CNN           | NFNet        | 11       |
|CNN           | TinyNet      | 5        |
|CNN           | LCNet        | 3        |
|CNN           | DLA          | 12       |
|CNN           | MnasNet      | 4        |
|CNN           | CSPNet       | 3        |
|ViT           | General ViT  | 8        |
|ViT           | MobileViT    | 10       |
|ViT           | Swin         | 22       |
|ViT           | MaxViT       | 14       |
|ViT           | DeiT         | 24       |
|ViT           | CaiT         | 10       |
|ViT           | XCiT         | 28       |
|ViT           | EVA          | 5        |
| Hybrid       | VOLO         | 8        |
| Hybrid       | CoAtNet      | 13       |

