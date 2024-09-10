# Efficient Multimodal Semantic Segmentation via Dual-Prompt Learning (DPLNet)
🔮 Welcome to the official code repository for [Efficient Multimodal Semantic Segmentation via Dual-Prompt Learning](https://arxiv.org/pdf/2312.00360.pdf). We're excited to share our work with you, please bear with us as we prepare the code and demo. Stay tuned for the reveal!

🔮 Our work has been accepted by **IROS 2024 (Oral presentation)**!


## Illustration of Idea
💡 Previous multimodal methods often need to fully fine-tune the entire network, which are training-costly due to massive parameter updates in the feature extraction and fusion, and thus increases the deployment burden of multimodal semantic segmentation. In this paper, we propose a novel and simple yet effective dual-prompt learning paradigm, dubbed DPLNet, for training-efficient multimodal semantic segmentation.

<img src="https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/fig1.png" alt="Editor" width="500">

## Framework
![Framework](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/framework.png)
Overview architecture of the proposed DPLNet, which adapts a frozen pre-trained model using two specially designed prompting learning modules, MPG for multimodal prompt generation and MFA for multimodal feature adaption, with only a few learnable parameters to achieve multimodal semantic segmentation in a training-efficient way. More details can be seen in the [**paper**](https://arxiv.org/abs/2401.01578).


## Implementation

### Requirements
The code has been tested and verified using PyTorch 1.12.0 and CUDA 11.8. However, compatibility with other versions is also likely.

### Dataset Preparation
NYUDv2 dataset can be download here [NYUDv2](https://drive.google.com/drive/folders/1tief3fgaTe2hown8FRnrb9ZtsMeoWtlv). # change the data root in ./RGBD/configs/nyuv2.json

### Pretrained Model Weights
We provide our trained checkpoints for results reproducibility.
| Dataset | url |mIoU(SS/MS)| 
|:----:|:-----:|:-----:|
| NYUv2 | [Model](https://drive.google.com/drive/folders/1f7o1t3ShAqXiYAhgHPJTlLtaDCT2aTIN)  | 58.3/59.3 |

Put the [segformer pre-trained weight](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxieenze%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsegformer%2Ftrained%5Fmodels&ga=1). in the following file (We use segformer.b5.640x640.ade.160k.pth in our paper).
```shell
vim ./RGBD/toolbox/models/segformermodels/backbones/mix_transformer_ourprompt_proj.py   # line 457
```


### Training
```shell
# run for NYUV2
cd ./RGBD
python train.py
```

### Evaluation
```shell
# run for NYUV2,put the pretrained weight on your folder.
# example: python evaluate.py --logdir /mnt/DATA/shaohuadong/DPLNet/NYUDv2
cd ./RGBD
python evaluate.py --logdir "MODEL PATH"
```

## Experiments
🎏 DPLNet achieves state-of-the-art performance on challenging tasks, including RGB-D Semantic Segmentation, RGB-T Semantic Segmentation, RGB-T Video Semantic Segmentation, RGB-D SOD and RGB-T SOD. Note that 'SS' and 'MS' refer to single-scale and multi-scale testing, respectively. Additional results can be found in our [paper](https://arxiv.org/pdf/2312.00360.pdf).

### Results on NYUDv2 (RGB-D Semantic Segmentation)
|  Methods   | Backbone | Total Params | Learnable Params |mIoU|
|  ----:  | :----:  | :----: | :----: |:----: |
|CMX-B5 (MS) | MiT-B5 | 181.1 |   181.1 | 56.9|
|CMXNeXt (MS) | MiT-B4 |  119.6 |  119.6 |  56.9|
|DFormer-L (MS) | DFormer-L | 39.0 | 39.0 | 57.2|
|DPLNet (SS) (Ours) | MiT-B5 | **7.15** | **7.15** | 58.3|
|DPLNet (MS) (Ours)| MiT-B5 | **7.15** | **7.15** | **59.3**|

### Results on SUN RGB-D (RGB-D Semantic Segmentation)
|  Methods   | Backbone | Total Params | Learnable Params |mIoU|
|  ----:  | :----:  | :----: | :----: |:----: |
|CMX-B4 (MS) | MiT-B4 | 139.9 |   139.9 | 52.1|
|CMX-B5 (MS) | MiT-B5 | 181.1 |   181.1 | 52.4|
|CMXNeXt (MS) | MiT-B4 |  119.6 |  119.6 |   51.9|
|DFormer-B (MS) | DFormer-B | 29.5 | 29.5 | 51.2|
|DFormer-L (MS) | DFormer-L | 39.0 | 39.0 | 52.5|
|DPLNet (SS) (Ours) | MiT-B5 | **7.15** | **7.15** | 52.1|
|DPLNet (MS) (Ours)| MiT-B5 | **7.15** | **7.15** | **52.8**|

### Results on MFNet (RGB-T Semantic Segmentation)
|  Methods   | Backbone | Total Params | Learnable Params |mIoU|
|  ----:  | :----:  | :----: | :----: |:----: |
|EGFNet| ResNet-152 | 201.3 |   201.3 | 54.8|
|MTANet| ResNet-152 | 121.9 |   121.9 | 56.1|
|GEBNet| ConvNeXt-S|  - |  - |  56.2|
|CMX-B2| MiT-B2| 66.6 | 66.6 |  58.2|
|CMX-B4 | MiT-B4|  139.9 | 139.9 | 59.7|
|CMNeXt| MiT-B4 |  119.6 |  119.6 | **59.9**|
|DPLNet (Ours)| MiT-B5 | **7.15** | **7.15** | 59.3|

### Results on PST900 (RGB-T Semantic Segmentation)
|  Methods   | Backbone | Total Params | Learnable Params |mIoU|
|  ----:  | :----:  | :----: | :----: |:----: |
|EGFNet| ResNet-152 | 201.3 |   201.3 | 78.5|
|MTANet| ResNet-152 | 121.9 |   121.9 | 78.6|
|GEBNet| ConvNeXt-S|  - |  - |  81.2|
|EGFNet-ConvNext| ConvNeXt-B | - |   - | 85.4|
|CACFNet| ConvNeXt-B |  198.6 |  198.6 | 86.6|
|DPLNet (Ours)| MiT-B5 | **7.15** | **7.15** | **86.7**|

### Results on MVSeg (RGB-T Video Semantic Segmentation)
|  Methods   | Backbone | Total Params | Learnable Params |mIoU|
|  ----:  | :----:  | :----: | :----: |:----: |
|EGFNet| ResNet-152 | 201.3 |   201.3 | 53.4|
|MVNet| - | 88.4 |   88.4 | 54.5|
|DPLNet (Ours)| MiT-B5 | **7.15** | **7.15** | **57.9**|

## Acknowledgement
This repository is partially based on our previous open-source release [EGFNet](https://github.com/ShaohuaDong2021/EGFNet).

## Citation
⭐ If you find this repository useful, please consider giving it a star and citing it:
```
@article{dong2023efficient,
  title={Efficient multimodal semantic segmentation via dual-prompt learning},
  author={Dong, Shaohua and Feng, Yunhe and Yang, Qing and Huang, Yan and Liu, Dongfang and Fan, Heng},
  journal={arXiv preprint arXiv:2312.00360},
  year={2023}
}
```

