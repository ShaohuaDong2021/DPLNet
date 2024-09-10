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

### Dataset Preparation
The used datasets are placed in `data` folder with the following structure.
```
data
|_ vidstg
|  |_ videos
|  |  |_ [video name 0].mp4
|  |  |_ [video name 1].mp4
|  |  |_ ...
|  |_ vstg_annos
|  |  |_ train.json
|  |  |_ ...
|  |_ sent_annos
|  |  |_ train_annotations.json
|  |  |_ ...
|  |_ data_cache
|  |  |_ ...
|_ hc-stvg2
|  |_ v2_video
|  |  |_ [video name 0].mp4
|  |  |_ [video name 1].mp4
|  |  |_ ...
|  |_ annos
|  |  |_ hcstvg_v2
|  |  |  |_ train.json
|  |  |  |_ test.json
|  |  data_cache
|  |  |_ ...
|_ hc-stvg
|  |_ v1_video
|  |  |_ [video name 0].mp4
|  |  |_ [video name 1].mp4
|  |  |_ ...
|  |_ annos
|  |  |_ hcstvg_v1
|  |  |  |_ train.json
|  |  |  |_ test.json
|  |  data_cache
|  |  |_ ...
```

The download link for the above-mentioned document is as follows:

**hc-stvg**: [v1_video](https://intxyz-my.sharepoint.com/:f:/g/personal/zongheng_picdataset_com/EgIzBzuHYPtItBIqIq5hNrsBBE9cnhJDWjXuorxXMhMZGQ?e=qvsBjE), [annos](https://intxyz-my.sharepoint.com/:f:/g/personal/zongheng_picdataset_com/EgIzBzuHYPtItBIqIq5hNrsBBE9cnhJDWjXuorxXMhMZGQ?e=qvsBjE), [data_cache](https://disk.pku.edu.cn/link/AA66258EA52A1E435B815C4BC10E88925D)

**hc-stvg2**: [v2_video](https://intxyz-my.sharepoint.com/:f:/g/personal/zongheng_picdataset_com/ErqA01jikPZKnudZe6-Za9MBe17XXAxJr9ODn65Z2qGKkw?e=7vKw1U), [annos](https://intxyz-my.sharepoint.com/:f:/g/personal/zongheng_picdataset_com/ErqA01jikPZKnudZe6-Za9MBe17XXAxJr9ODn65Z2qGKkw?e=7vKw1U), [data_cache]()

**vidstg**: [videos](https://disk.pku.edu.cn/link/AA93DEAF3BBC694E52ACC5A23A9DC3D03B), [vstg_annos](https://disk.pku.edu.cn/link/AA9BD598C845DC43A4B6A0D35268724E4B), [sent_annos](https://github.com/Guaranteer/VidSTG-Dataset), [data_cache](https://disk.pku.edu.cn/link/AAA0FA082DEB3D47FCA92F3BF8775EA3BC) 



### Model Preparation
The used datasets are placed in `model_zoo` folder

[ResNet-101](https://zenodo.org/record/4721981/files/pretrained_resnet101_checkpoint.pth?download=1), 
[VidSwin-T](https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_tiny_patch244_window877_kinetics400_1k.pth),
[roberta-base](https://huggingface.co/FacebookAI/roberta-base)

### Requirements
The code has been tested and verified using PyTorch 2.0.1 and CUDA 11.7. However, compatibility with other versions is also likely. To install the necessary requirements, please use the commands provided below:

```shell
pip3 install -r requirements.txt
apt install ffmpeg -y
```

### Training
Please utilize the script provided below:
```shell
# run for HC-STVG
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train_net.py \
    --config-file "experiments/hcstvg.yaml" \
    INPUT.RESOLUTION 420 \
    OUTPUT_DIR output/hcstvg \
    TENSORBOARD_DIR output/hcstvg

# run for HC-STVG2
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train_net.py \
    --config-file "experiments/hcstvg2.yaml" \
    INPUT.RESOLUTION 420 \
    OUTPUT_DIR output/hcstvg2 \
    TENSORBOARD_DIR output/hcstvg2

# run for VidSTG
python3 -m torch.distributed.launch \
    --nproc_per_node=8 \
    scripts/train_net.py \
    --config-file "experiments/vidstg.yaml" \
    INPUT.RESOLUTION 420 \
    OUTPUT_DIR output/vidstg \
    TENSORBOARD_DIR output/vidstg
```
For additional training options, such as utilizing different hyper-parameters, please adjust the configurations as needed:
`experiments/hcstvg.yaml`, `experiments/hcstvg2.yaml` and `experiments/vidstg.yaml`.

### Evaluation
Please utilize the script provided below:
```shell
# run for NYUV2
python evaluate_nyuv2.py --logdir "MODEL PATH"
```

### Pretrained Model Weights
We provide our trained checkpoints for results reproducibility.

| Dataset | resolution | url | m_vIoU/vIoU@0.3/vIoU@0.5 | size |
|:----:|:-----:|:-----:|:-----:|:-----:|
| HC-STVG | 420 | [Model](https://drive.google.com/drive/folders/1f7o1t3ShAqXiYAhgHPJTlLtaDCT2aTIN)  | 38.4/61.5/36.3 | 3.4 GB |
| HC-STVG2 | 420 | [Model](https://huggingface.co/Gstar666/CGSTVG/resolve/main/hcstvg2.pth?download=true)  | 39.5/64.5/36.3 | 3.4 GB |
| VidSTG | 420 | [Model](https://huggingface.co/Gstar666/CGSTVG/resolve/main/vidstg.pth?download=true)  | 34.0/47.7/33.1 | 3.4 GB |


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

