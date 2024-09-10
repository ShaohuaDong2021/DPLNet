from .metrics import averageMeter, runningScore
from .log import get_logger
from .loss import MscCrossEntropyLoss
from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, adjust_lr
from .ranger.ranger import Ranger
from .ranger.ranger913A import RangerVA
from .ranger.rangerqh import RangerQH
import torch.nn as nn

def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'sunrgbd224']

    # if cfg['dataset'] == 'nyuv2':
    #     from .datasets.nyuv2_yang import NYUv2
    #     return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'nyuv2':
        from .datasets.nyuv2 import NYUv2
        return NYUv2(cfg, mode='train'), NYUv2(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd':
        from .datasets.sunrgbd import SUNRGBD
        return SUNRGBD(cfg, mode='train'), SUNRGBD(cfg, mode='test')
    if cfg['dataset'] == 'sunrgbd224':
        from .datasets.sunrgbd224 import SUNRGBD224
        return SUNRGBD224(cfg, mode='train'), SUNRGBD224(cfg, mode='test')


def get_model(cfg):
    if cfg['model_name'] == 'rednet':
        from .models.rednet import RedNet
        return RedNet(num_classes=cfg['n_classes'], pretrained=True)

    if cfg['model_name'] == 'bbsnet':
        from .models.BBSnetmodel.BBSnet import BBSNet
        return BBSNet(n_class=cfg['n_classes'])

    if cfg['model_name'] == 'edgeformer_prompt121':
        from .models.edgeformer_prompt121 import edgeformer_prompt121
        return edgeformer_prompt121()

    if cfg['model_name'] == 'edgeformer_prompt121sun':
        from .models.edgeformer_prompt121sun import edgeformer_prompt121sun
        return edgeformer_prompt121sun()

    if cfg['model_name'] == 'edgeformer_prompt121sun_20m':
        from .models.edgeformer_prompt121sun_20m import edgeformer_prompt121sun_20m
        return edgeformer_prompt121sun_20m()

    if cfg['model_name'] == 'edgeformer_prompt121nyu_20m':
        from .models.edgeformer_prompt121nyu_20m import edgeformer_prompt121nyu_20m
        return edgeformer_prompt121nyu_20m()


    if cfg['model_name'] == 'nyuv2_token10':
        from .models.nyuv2_token10 import nyuv2_token10
        return nyuv2_token10()

    if cfg['model_name'] == 'nyuv2_token20':
        from .models.nyuv2_token20 import nyuv2_token20
        return nyuv2_token20()

    if cfg['model_name'] == 'nyuv2_token40':
        from .models.nyuv2_token40 import nyuv2_token40
        return nyuv2_token40()

    if cfg['model_name'] == 'nyuv2_dim8':
        from .models.nyuv2_dim8 import nyuv2_dim8
        return nyuv2_dim8()

    if cfg['model_name'] == 'nyuv2_dim16':
        from .models.nyuv2_dim16 import nyuv2_dim16
        return nyuv2_dim16()

    if cfg['model_name'] == 'nyuv2_dim64':
        from .models.nyuv2_dim64 import nyuv2_dim64
        return nyuv2_dim64()

    if cfg['model_name'] == 'nyuv2_dim128':
        from .models.nyuv2_dim128 import nyuv2_dim128
        return nyuv2_dim128()

    if cfg['model_name'] == 'nyuv2_backbone0':
        from .models.nyuv2_backbone0 import nyuv2_backbone0
        return nyuv2_backbone0()

    if cfg['model_name'] == 'nyuv2_backbone1':
        from .models.nyuv2_backbone1 import nyuv2_backbone1
        return nyuv2_backbone1()

    if cfg['model_name'] == 'nyuv2_backbone1t2':
        from .models.nyuv2_backbone1t2 import nyuv2_backbone1t2
        return nyuv2_backbone1t2()

    if cfg['model_name'] == 'nyuv2_backbone1t3':
        from .models.nyuv2_backbone1t3 import nyuv2_backbone1t3
        return nyuv2_backbone1t3()

    if cfg['model_name'] == 'nyuv2_backbone4':
        from .models.nyuv2_backbone4 import nyuv2_backbone4
        return nyuv2_backbone4()

    if cfg['model_name'] == 'nyuv2_backbone4t3':
        from .models.nyuv2_backbone4t3 import nyuv2_backbone4t3
        return nyuv2_backbone4t3()

    if cfg['model_name'] == 'nyuv2_backbone4t2':
        from .models.nyuv2_backbone4t2 import nyuv2_backbone4t2
        return nyuv2_backbone4t2()

    if cfg['model_name'] == 'nyuv2_m0':
        from .models.nyuv2_m0 import nyuv2_m0
        return nyuv2_m0()

    if cfg['model_name'] == 'nyuv2_m1':
        from .models.nyuv2_m1 import nyuv2_m1
        return nyuv2_m1()

    if cfg['model_name'] == 'nyuv2_m1t2':
        from .models.nyuv2_m1t2 import nyuv2_m1t2
        return nyuv2_m1t2()

    if cfg['model_name'] == 'nyuv2_m1t3':
        from .models.nyuv2_m1t3 import nyuv2_m1t3
        return nyuv2_m1t3()

    if cfg['model_name'] == 'nyuv2_m4':
        from .models.nyuv2_m4 import nyuv2_m4
        return nyuv2_m4()

    if cfg['model_name'] == 'nyuv2_m4t3':
        from .models.nyuv2_m4t3 import nyuv2_m4t3
        return nyuv2_m4t3()

    if cfg['model_name'] == 'nyuv2_m4t2':
        from .models.nyuv2_m4t2 import nyuv2_m4t2
        return nyuv2_m4t2()

    if cfg['model_name'] == 'nyuv2_full':
        from .models.nyuv2_full import nyuv2_full
        return nyuv2_full()

    if cfg['model_name'] == 'nyuv2_deep':
        from .models.nyuv2_deep import nyuv2_deep
        return nyuv2_deep()

    if cfg['model_name'] == 'nyuv2_head':
        from .models.nyuv2_head import nyuv2_head
        return nyuv2_head()

    if cfg['model_name'] == 'nyuv2_adapterformer':
        from .models.nyuv2_adapterformer import nyuv2_adapterformer
        return nyuv2_adapterformer()

    if cfg['model_name'] == 'MITB5':
        from toolbox.models.segformermodels.backbones.mix_transformer_ourprompt_proj_pvt import MITB5
        return MITB5()

    if cfg['model_name'] == 'mamba_rgbd':
        from .models.mamba_rgbd import mamba_rgbd
        return mamba_rgbd()





