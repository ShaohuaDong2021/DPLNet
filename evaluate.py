import os
import time
from tqdm import tqdm
from PIL import Image
import json

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from toolbox import get_dataset
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import class_to_RGB, load_ckpt


def evaluate(logdir, save_predict=False):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')
    # device = torch.device('cpu')

    testset = get_dataset(cfg)[-1]
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    model = get_model(cfg).to(device)
    model = load_ckpt(logdir, model, kind='best')

    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()

    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    with torch.no_grad():
        model.eval()
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            time_start = time.time()
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)

            predict = model(image, depth)

            predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
            label = label.cpu().numpy()
            running_metrics_val.update(label, predict)

            time_meter.update(time.time() - time_start, n=image.size(0))

            if save_predict:
                predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                predict = class_to_RGB(predict, N=len(testset.cmap), cmap=testset.cmap)  # 如果数据集没有给定cmap,使用默认cmap
                predict = Image.fromarray(predict)
                predict.save(os.path.join(save_path, sample['label_path'][0]))

    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)
    # for k, v in metrics[1].items():
    #     print(k, v)
    print('inference time per image: ', time_meter.avg)
    print('inference fps: ', 1 / time_meter.avg)


def msc_evaluate(logdir, save_predict=False):
    # 加载配置文件cfg
    cfg = None
    for file in os.listdir(logdir):
        if file.endswith('.json'):
            with open(os.path.join(logdir, file), 'r') as fp:
                cfg = json.load(fp)
    assert cfg is not None

    device = torch.device('cuda')

    testset = get_dataset(cfg)[-1]
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg['num_workers'])

    model = get_model(cfg).to(device)
    model = load_ckpt(logdir, model, kind='best')

    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    time_meter = averageMeter()

    save_path = os.path.join(logdir, 'predicts')
    if not os.path.exists(save_path) and save_predict:
        os.mkdir(save_path)

    with torch.no_grad():
        model.eval()
        eval_scales = tuple(float(i) for i in cfg['eval_scales'].split(' '))
        eval_flip = cfg['eval_flip'] == 'true'
        for i, sample in tqdm(enumerate(test_loader), total=len(test_loader)):
            time_start = time.time()
            depth = sample['depth'].to(device)
            image = sample['image'].to(device)
            label = sample['label'].to(device)
            # resize
            h, w = image.size(2), image.size(3)

            predicts = torch.zeros((1, cfg['n_classes'], h, w), requires_grad=False).to(device)
            for scale in eval_scales:
                newHW = (int((h * scale // 32) * 32), int((w * scale // 32) * 32))
                new_image = F.interpolate(image, newHW, mode='bilinear', align_corners=True)
                new_depth = F.interpolate(depth, newHW, mode='bilinear', align_corners=True)
                out = model(new_image, new_depth)
                out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                prob = F.softmax(out, 1)
                predicts += prob
                if eval_flip:
                    out = model(torch.flip(new_image, dims=(3,)), torch.flip(new_depth, dims=(3,)))
                    out = torch.flip(out, dims=(3,))
                    out = F.interpolate(out, (h, w), mode='bilinear', align_corners=True)
                    prob = F.softmax(out, 1)
                    predicts += prob
            predict = predicts.max(1)[1].cpu().numpy()
            label = label.cpu().numpy()
            running_metrics_val.update(label, predict)

            time_meter.update(time.time() - time_start, n=image.size(0))

            if save_predict:
                predict = predict.squeeze(0)  # [1, h, w] -> [h, w]
                predict = class_to_RGB(predict, N=len(testset.cmap), cmap=testset.cmap)  # 如果数据集没有给定cmap,使用默认cmap
                predict = Image.fromarray(predict)
                predict.save(os.path.join(save_path, sample['label_path'][0]))

    metrics = running_metrics_val.get_scores()
    for k, v in metrics[0].items():
        print(k, v)
    # for k, v in metrics[1].items():
    #     print(k, v)
    print('inference time per image: ', time_meter.avg)
    print('inference fps: ', 1 / time_meter.avg)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="evaluate")
    parser.add_argument("--logdir", type=str, help="run logdir")
    parser.add_argument("-s", type=bool, default=False, help="save predict or not")

    args = parser.parse_args()

    evaluate(args.logdir, save_predict=args.s)
    # msc_evaluate(args.logdir, save_predict=args.s)
