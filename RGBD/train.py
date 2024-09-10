import os
import shutil
import json
import time
from torch.cuda import amp
import torch.distributed as dist
import torch
import torch.backends.cudnn as cudnn

from torch.optim.lr_scheduler import LambdaLR, StepLR
from torch.utils.data import DataLoader

from toolbox import MscCrossEntropyLoss
from toolbox import get_dataset
from toolbox import get_logger
from toolbox import get_model
from toolbox import averageMeter, runningScore
from toolbox import ClassWeight, save_ckpt,load_ckpt

torch.manual_seed(123)
cudnn.benchmark = True

def run(args):
    with open(args.config, 'r') as fp:
        cfg = json.load(fp)

    # set where you save the model weight
    logdir = f'run/{time.strftime("%Y-%m-%d-%H-%M")}'

    #logdir = 'run/2020-12-23-18-38'
    args.logdir = logdir

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    if args.local_rank == 0:
        logger.info(f'Conf | use logdir {logdir}')

    model = get_model(cfg)
    #model = load_ckpt(logdir,model,kind='end')
    trainset, *testset = get_dataset(cfg)

    # Which gpu you want to use,
    device = torch.device('cuda:1')
    args.distributed = False

    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
        if args.local_rank == 0:
            print(f"WORLD_SIZE is {os.environ['WORLD_SIZE']}")

    train_sampler = None
    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()

        # model = apex.parallel.convert_syncbn_model(model)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    model.to(device)
    train_loader = DataLoader(trainset, batch_size=cfg['ims_per_gpu'], shuffle=(train_sampler is None),
                              num_workers=cfg['num_workers'], pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(testset[0], batch_size=cfg['ims_per_gpu'], shuffle=False,num_workers=cfg['num_workers'],pin_memory=True)
    params_list = model.parameters()
    # wd_params, non_wd_params = model.get_params()
    # params_list = [{'params': wd_params, },
    #                {'params': non_wd_params, 'weight_decay': 0}]

    # optimizer = torch.optim.Adam(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'])
    optimizer = torch.optim.SGD(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])
    Scaler = amp.GradScaler()
    #optimizer = Ranger(params_list, lr=cfg['lr_start'], weight_decay=cfg['weight_decay']
    scheduler = LambdaLR(optimizer, lr_lambda=lambda ep: (1 - ep / cfg['epochs']) ** 0.9)

    # model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)
    if args.distributed:
      model = torch.nn.parallel.DistributedDataParallel(model)

    if hasattr(trainset, 'class_weight'):
        print('using classweight in dataset')
        class_weight = trainset.class_weight
    else:
        classweight = ClassWeight(cfg['class_weight'])
        class_weight = classweight.get_weight(train_loader, cfg['n_classes'])
    class_weight = torch.from_numpy(class_weight).float().to(device)

    class_weight[cfg['id_unlabel']] = 0

    criterion = MscCrossEntropyLoss(weight=class_weight).to(device)

    train_loss_meter = averageMeter()
    val_loss_meter = averageMeter()
    running_metrics_val = runningScore(cfg['n_classes'], ignore_index=cfg['id_unlabel'])
    flag = True
    miou = 0
    for ep in range(cfg['epochs']):
        if args.distributed:
            train_sampler.set_epoch(ep)
        # training
        model.train()
        train_loss_meter.reset()

        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            ################### train edit #######################
            depth = sample['depth'].to(device)

            image = sample['image'].to(device)

            label = sample['label'].to(device)
            # label = sample['labelcxk'].to(device)
            # print(i,set(label.cpu().reshape(-1).tolist()),'label')
            with amp.autocast():
                predict = model(image, depth)

                loss = criterion(predict, label)

            ####################################################
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            Scaler.scale(loss).backward()
            Scaler.step(optimizer)
            Scaler.update()
            # optimizer.step()

            if args.distributed:
                reduced_loss = loss.clone()
                dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                reduced_loss /= args.world_size
            else:
                reduced_loss = loss
            train_loss_meter.update(reduced_loss.item())

        scheduler.step(ep)

        # val
        with torch.no_grad():
            model.eval()
            running_metrics_val.reset()

            val_loss_meter.reset()
            for i, sample in enumerate(val_loader):
                depth = sample['depth'].to(device)
                image = sample['image'].to(device)
                label = sample['label'].to(device)

                predict = model(image, depth)

                loss = criterion(predict, label)
                val_loss_meter.update(loss.item())


                predict = predict.max(1)[1].cpu().numpy()  # [1, h, w]
                label = label.cpu().numpy()
                running_metrics_val.update(label, predict)

        if args.local_rank == 0:
            logger.info(
                f'Iter | [{ep + 1:3d}/{cfg["epochs"]}] train/val loss={train_loss_meter.avg:.5f}/{val_loss_meter.avg:.5f}, '
                f'miou={running_metrics_val.get_scores()[0]["mIou: "]:.3f}')
            save_ckpt(logdir, model, kind='end')
            newmiou = running_metrics_val.get_scores()[0]["mIou: "]

            # we want to save the best mIoU
            if newmiou>miou:
                save_ckpt(logdir,model,kind='best')
                miou = newmiou

    save_ckpt(logdir, model,kind='end')  # save last weight

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/nyuv2.json",
        help="Configuration file to use",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--opt_level",
        type=str,
        default='O1',
    )
    args = parser.parse_args()
    run(args)