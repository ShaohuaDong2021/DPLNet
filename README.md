
# DPLNet (Efficient Multimodal Semantic Segmentation via Dual-Prompt Learning)


Welcome to the official code repository for [Efficient Multimodal Semantic Segmentation via Dual-Prompt Learning](https://arxiv.org/pdf/2312.00360.pdf). We're excited to share our work with you, please bear with us as we prepare the code and demo. Stay tuned for the reveal!


## Motivation
Previous multimodal methods often need to fully fine-tune the entire network, which are training-costly due to massive parameter updates in the feature extraction and fusion, and thus increases the deployment burden of multimodal semantic segmentation. In this paper, we propose a novel and simple yet effective dual-prompt
learning paradigm, dubbed DPLNet, for training-efficient multimodal semantic segmentation.

<img src="https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/fig1.png" alt="Editor" width="500">

## Framework
Overview architecture of the proposed DPLNet, which adapts a frozen pre-trained model using two specially designed prompting learning modules, MPG for multimodal prompt generation and MFA for multimodal feature adaption, with only a few learnable parameters to achieve multimodal semantic segmentation in a training-efficient way.
![Framework](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/framework.png)

## Visualization
<video src='https://github.com/ShaohuaDong2021/DPLNet/blob/main/Visualization_video.mp4' width="500"></video>
  
## RGBD Semantic Segmentation Results
### NYU-V2
![Results](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/nyuv2.png)

### SUN-RGBD
![Results](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/sunrgbd.png)

## RGBT Semantic Segmentation Results
### MFNet
![Results](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/mfnet.png)

### PST900
![Results](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/PST900.png)

## RGB-D SOD Results
![Results](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/RGBDSOD.png)

## RGB-T SOD Results
![Results](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/RGBTSOD.png)

## RGB-T Video Semantic Segmentation Results
![Results](https://github.com/ShaohuaDong2021/DPLNet/blob/main/figs/MVSeg.png)



[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_HCSTVGv1-orange?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-hc-stvg1)
[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_HCSTVGv2-pink?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-hc-stvg2)
[![PWC](https://img.shields.io/badge/State_of_the_Art-STVG_on_VidSTG-yellow?logo=AMP)](https://paperswithcode.com/sota/spatio-temporal-video-grounding-on-vidstg)

# Context-Guided Spatio-Temporal Video Grounding
🔮 Welcome to the official code repository for [**CG-STVG: Context-Guided Spatio-Temporal Video Grounding**](https://arxiv.org/abs/2401.01578). We're excited to share our work with you, please bear with us as we prepare code. Stay tuned for the reveal!

## Illustration of Idea
💡 ***A picture is worth a thousand words!*** <br>Can we explore visual context from videos to enhance target localization for STVG? Yes!

![CG-STVG](https://github.com/HengLan/CGSTVG/blob/main/assets/idea.png)
**Figure:** Illustration of and comparison between (a) existing methods that localize the target using object information from text query and (b) our CG-STVG
that enjoys object information from text query and guidance from instance context for STVG. 

## Framework
![CG-STVG](https://github.com/HengLan/CGSTVG/blob/main/assets/framework.png)
**Figure:** Overview of our method, which consists of a multimodal encoder for feature extraction and a context-guided decoder by cascading
a set of decoding stages for grounding. In each decoding stage, instance context is mined (by ***ICG*** and ***ICR***) to guide query learning for better localization. More details can be seen in the [**paper**](https://arxiv.org/abs/2401.01578).

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
# run for HC-STVG
python3 -m torch.distributed.launch \
 --nproc_per_node=8 \
 scripts/test_net.py \
 --config-file "experiments/hcstvg.yaml" \
 INPUT.RESOLUTION 420 \
 MODEL.WEIGHT [Pretrained Model Weights] \
 OUTPUT_DIR output/hcstvg
 
# run for HC-STVG2
python3 -m torch.distributed.launch \
 --nproc_per_node=8 \
 scripts/test_net.py \
 --config-file "experiments/hcstvg2.yaml" \
 INPUT.RESOLUTION 420 \
 MODEL.WEIGHT [Pretrained Model Weights] \
 OUTPUT_DIR output/hcstvg2

# run for VidSTG
python3 -m torch.distributed.launch \
 --nproc_per_node=8 \
 scripts/test_net.py \
 --config-file "experiments/vidstg.yaml" \
 INPUT.RESOLUTION 420 \
 MODEL.WEIGHT [Pretrained Model Weights] \
 OUTPUT_DIR output/vidstg
```

### Pretrained Model Weights
We provide our trained checkpoints for results reproducibility.

| Dataset | resolution | url | m_vIoU/vIoU@0.3/vIoU@0.5 | size |
|:----:|:-----:|:-----:|:-----:|:-----:|
| HC-STVG | 420 | [Model](https://huggingface.co/Gstar666/CGSTVG/resolve/main/hcstvg.pth?download=true)  | 38.4/61.5/36.3 | 3.4 GB |
| HC-STVG2 | 420 | [Model](https://huggingface.co/Gstar666/CGSTVG/resolve/main/hcstvg2.pth?download=true)  | 39.5/64.5/36.3 | 3.4 GB |
| VidSTG | 420 | [Model](https://huggingface.co/Gstar666/CGSTVG/resolve/main/vidstg.pth?download=true)  | 34.0/47.7/33.1 | 3.4 GB |


## Experiments
🎏 CG-STVG achieves state-of-the-art performance on three challenging benchmarks, including [**HCSTVG-v1**](https://github.com/tzhhhh123/HC-STVG), [**HCSTVG-v2**](https://github.com/tzhhhh123/HC-STVG), and [**VidSTG**](https://github.com/Guaranteer/VidSTG-Dataset), as shown below. Note that, the baseline is our CG-STVG without context generation and refinement.

### Results on HCSTVG-v1
|  Methods   | M_tIoU | m_vIoU | vIoU@0.3 | vIoU@0.5  |
|  ----:  | :----:  | :----: | :----: | :----: |
|STGVT<sub>[TCSVT'2021]</sub> | - |  18.2 | 26.8 | 9.5|
|STVGBert<sub>[ICCV'2021]</sub> | - | 20.4 | 29.4 |  11.3|
|TubeDETR<sub>[CVPR'2022]</sub> | 43.7 | 32.4 | 49.8 | 23.5|
|STCAT<sub>[NeurIPS'2022]</sub> | 49.4 | 35.1 | 57.7 | 30.1|
|CSDVL<sub>[CVPR'2023]</sub> | - | 36.9 | **62.2** | 34.8|
|Baseline (ours) | 50.4 | 36.5 | 58.6 | 32.3 |
|CG-STVG (ours)|**52.8**<sub>(+2.4)</sub> | **38.4**<sub>(+1.9)</sub> | 61.5<sub>(+2.9)</sub> | **36.3**<sub>(+4.0)</sub>|

### Results on HCSTVG-v2
|  Methods   | M_tIoU | m_vIoU | vIoU@0.3 | vIoU@0.5  |
|  ----:  | :----:  | :----: | :----: | :----: |
|PCC<sub>[arxiv'2021]</sub> | - |  30.0 | - | - |
|2D-Tan<sub>[arxiv'2021]</sub>  | - | 30.4 |  50.4 | 18.8 |
|MMN<sub>[AAAI'2022]</sub> | - | 30.3 | 49.0 | 25.6|
|TubeDETR<sub>[CVPR'2022]</sub> | - | 36.4 | 58.8 | 30.6|
|CSDVL<sub>[CVPR'2023]</sub> | 58.1 | 38.7 | **65.5** | 33.8|
|Baseline (ours) | 58.6 | 37.8 | 62.4 | 32.1|
|CG-STVG (ours) | **60.0**<sub>(+1.4)</sub> | **39.5**<sub>(+1.7)</sub> | 64.5<sub>(+2.1)</sub> | **36.3**<sub>(+4.2)</sub>|

### Results on VidSTG
<table>
  <tr>
    <td rowspan="2" align="right"><b>Methods</b></td>
    <td colspan="4" align="center"><b>Declarative Sentences</b></td>
    <td colspan="4" align="center"><b>Interrogative Sentences</b></td>
  </tr>
  <tr>
    <td align="center">&nbsp;&nbsp;&nbsp;<b>M_tIoU</b>&nbsp;&nbsp;&nbsp;</td>
    <td align="center">&nbsp;&nbsp;&nbsp;<b>m_vIoU</b>&nbsp;&nbsp;&nbsp;</td>
    <td align="center">&nbsp;&nbsp;&nbsp;<b>vIoU@0.3</b>&nbsp;&nbsp;&nbsp;</td>
    <td align="center">&nbsp;&nbsp;&nbsp;<b>vIoU@0.5</b>&nbsp;&nbsp;&nbsp;</td>
    <td align="center">&nbsp;&nbsp;&nbsp;<b>M_tIoU</b>&nbsp;&nbsp;&nbsp;</td>
    <td align="center">&nbsp;&nbsp;&nbsp;<b>m_vIoU</b>&nbsp;&nbsp;&nbsp;</td>
    <td align="center">&nbsp;&nbsp;&nbsp;<b>vIoU@0.3</b>&nbsp;&nbsp;&nbsp;</td>
    <td align="center">&nbsp;&nbsp;&nbsp;<b>vIoU@0.5</b>&nbsp;&nbsp;&nbsp;</td>
  </tr>
  <tr>
    <td align="right">STGRN<sub>[CVPR'2020]</sub></td>
    <td align="center">48.5</td>
    <td align="center">19.8</td>
    <td align="center">25.8</td>
    <td align="center">14.6</td>
    <td align="center">47.0 </td>    
    <td align="center">18.3</td>
    <td align="center">21.1</td>
    <td align="center">12.8</td>
  </tr>
  <tr>
    <td align="right">OMRN<sub>[IJCAI'2020]</sub></td>
    <td align="center">50.7</td>
    <td align="center">23.1</td>
    <td align="center">32.6</td>
    <td align="center">16.4</td>
    <td align="center">49.2</td>    
    <td align="center">20.6</td>
    <td align="center">28.4</td>
    <td align="center">14.1</td>
  </tr>
  <tr>
    <td align="right">STGVT<sub>[TCSVT'2021]</sub></td>
    <td align="center">-</td>
    <td align="center">21.6</td>
    <td align="center">29.8</td>
    <td align="center">18.9</td>
    <td align="center">-</td>    
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
  <tr>
    <td align="right">STVGBert<sub>[ICCV'2021]</sub></td>
    <td align="center">-</td>
    <td align="center">24.0</td>
    <td align="center">30.9</td>
    <td align="center">18.4</td>
    <td align="center">-</td>    
    <td align="center">22.5</td>
    <td align="center">26.0</td>
    <td align="center">16.0</td>
  </tr>
  <tr>
    <td align="right">TubeDETR<sub>[CVPR'2022]</sub></td>
    <td align="center">48.1</td>
    <td align="center">30.4</td>
    <td align="center">42.5</td>
    <td align="center">28.2</td>
    <td align="center">46.9</td>    
    <td align="center">25.7</td>
    <td align="center">35.7</td>
    <td align="center">23.2</td>
  </tr>
  <tr>
    <td align="right">STCAT<sub>[NeurIPS'2022]</sub></td>
    <td align="center">50.8</td>
    <td align="center">33.1</td>
    <td align="center">46.2</td>
    <td align="center">32.6</td>
    <td align="center">49.7</td>    
    <td align="center">28.2</td>
    <td align="center">39.2</td>
    <td align="center">26.6</td>
  </tr>
  <tr>
    <td align="right">CSDVL<sub>[CVPR'2023]</sub></td>
    <td align="center">-</td>
    <td align="center">33.7</td>
    <td align="center">47.2</td>
    <td align="center">32.8</td>
    <td align="center">-</td>    
    <td align="center">28.5</td>
    <td align="center">39.9</td>
    <td align="center">26.2</td>
  </tr>
  <tr>
    <td align="right">Baseline (ours)</td>
    <td align="center">49.7</td>
    <td align="center">32.4</td>
    <td align="center">45.0</td>
    <td align="center">31.4</td>
    <td align="center">48.8</td>    
    <td align="center">27.7</td>
    <td align="center">38.7</td>
    <td align="center">25.6</td>
  </tr>
  <tr>
    <td align="right">CG-STVG (ours)</td>
    <td align="center"><b>51.4</b> <sub>(+1.7)</sub></td>
    <td align="center"><b>34.0</b> <sub>(+1.6)</sub></td>
    <td align="center"><b>47.7</b> <sub>(+2.7)</sub></td>
    <td align="center"><b>33.1</b> <sub>(+1.7)</sub></td>
    <td align="center"><b>49.9</b> <sub>(+1.1)</sub></td>    
    <td align="center"><b>29.0</b> <sub>(+1.3)</sub></td>
    <td align="center"><b>40.5</b> <sub>(+1.8)</sub></td>
    <td align="center"><b>27.5</b> <sub>(+1.9)</sub></td>
  </tr>
</table>

## Acknowledgement
This repo is partly based on our previous open-source release from [EGFNet](https://github.com/ShaohuaDong2021/EGFNet).

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

