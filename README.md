
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
<video src='https://github.com/ShaohuaDong2021/DPLNet/blob/main/Visualization_video.mp4' width=180/>


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
