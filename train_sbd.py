import os
import random
import sys
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, OptimizerParamsHandler, WeightsHistHandler
from ignite.engine import Events, create_supervised_evaluator, create_supervised_trainer
from ignite.metrics import RunningAverage, Loss
from torchvision.datasets import SBDataset

from googlenet_fcn.datasets.transforms.transforms import Compose, ToTensor, \
    Normalize, RandomHorizontalFlip, RandomAffine, RandomGaussionBlur, ColorJitter, RandomGaussionNoise
from googlenet_fcn.datasets.voc import VOC
from googlenet_fcn.metrics.confusion_matrix import ConfusionMatrix, IoU
from googlenet_fcn.model.googlenet_fcn import GoogLeNetFCN
from googlenet_fcn.utils import save, freeze_batchnorm, collate_fn


def get_data_loaders(data_dir, batch_size, val_batch_size, num_workers, download, augmentations):
    if augmentations:
        print("Using data augmentations")
        transform = Compose([
            RandomHorizontalFlip(),
            RandomAffine(translate=(0.1, 0.1), scale=(0.7, 2.0), shear=(-10, 10)),
            RandomGaussionBlur(radius=2.0),
            ColorJitter(0.1, 0.1, 0.1, 0.1),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomGaussionNoise()
        ])
    else:
        transform = Compose([
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    val_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = data.DataLoader(SBDataset(root=os.path.join(data_dir, 'sbd'), image_set='train', mode='segmentation',
                                             download=download, transforms=transform),
                                   batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                   collate_fn=collate_fn, pin_memory=True)

    val_loader = data.DataLoader(VOC(root=data_dir, download=download, transforms=val_transform),
                                 batch_size=val_batch_size, shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader


def run(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    num_classes = 21
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNetFCN(num_classes)
    model.init_from_googlenet()

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Using %d GPU(s)" % device_count)
        model = nn.DataParallel(model)
        args.batch_size = device_count * args.batch_size
        args.val_batch_size = device_count * args.val_batch_size

    model = model.to(device)

    train_loader, val_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.val_batch_size,
                                                args.num_workers, args.download, args.augmentations)

    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='sum')

    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if name.endswith('weight')]},
                           {'params': [param for name, param in model.named_parameters() if name.endswith('bias')],
                            'lr': args.lr * 2, 'weight_decay': 0}],
                          lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_iou = checkpoint['bestIoU']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (Epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            sys.exit()

    if args.freeze_bn:
        print("Freezing batch norm")
        model = freeze_batchnorm(model)

    trainer = create_supervised_trainer(model, optimizer, criterion, device, non_blocking=True)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    # attach progress bar
    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names=['loss'])

    cm = ConfusionMatrix(num_classes)
    evaluator = create_supervised_evaluator(model, metrics={'loss': Loss(criterion),
                                                            'IoU': IoU(cm)},
                                            device=device, non_blocking=True)

    pbar2 = ProgressBar(persist=True, desc='Eval Epoch')
    pbar2.attach(evaluator)

    def _global_step_transform(engine, event_name):
        return trainer.state.iteration

    tb_logger = TensorboardLogger(args.log_dir)
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag='training',
                                               metric_names=['loss']),
                     event_name=Events.ITERATION_COMPLETED)

    tb_logger.attach(trainer,
                     log_handler=OptimizerParamsHandler(optimizer),
                     event_name=Events.ITERATION_STARTED)

    tb_logger.attach(trainer,
                     log_handler=WeightsHistHandler(model),
                     event_name=Events.EPOCH_COMPLETED)

    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='validation',
                                               metric_names=['loss', 'IoU'],
                                               global_step_transform=_global_step_transform),
                     event_name=Events.EPOCH_COMPLETED)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        iou = engine.state.metrics['IoU'] * 100.0
        mean_iou = iou.mean()

        is_best = mean_iou.item() > trainer.state.best_iou
        trainer.state.best_iou = max(mean_iou.item(), trainer.state.best_iou)

        name = 'epoch{}_mIoU={:.1f}.pth'.format(trainer.state.epoch, mean_iou)
        file = {'model': model.state_dict(), 'epoch': trainer.state.epoch, 'iteration': engine.state.iteration,
                'optimizer': optimizer.state_dict(), 'args': args, 'bestIoU': trainer.state.best_iou}

        save(file, args.output_dir, 'checkpoint_{}'.format(name))
        if is_best:
            save(model.state_dict(), args.output_dir, 'model_{}'.format(name))

    @trainer.on(Events.STARTED)
    def initialize(engine):
        if args.resume:
            engine.state.epoch = args.start_epoch
            engine.state.iteration = args.start_epoch * len(engine.state.dataloader)
            engine.state.best_iou = best_iou
        else:
            engine.state.best_iou = 0.0

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        pbar.log_message("Start Validation - Epoch: [{}/{}]".format(engine.state.epoch, engine.state.max_epochs))
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        loss = metrics['loss']
        iou = metrics['IoU']
        mean_iou = iou.mean()

        pbar.log_message("Validation results - Epoch: [{}/{}]: Loss: {:.2e}, mIoU: {:.1f}"
                         .format(engine.state.epoch, engine.state.max_epochs, loss, mean_iou * 100.0))

    print("Start training")
    trainer.run(train_loader, max_epochs=args.epochs)
    tb_logger.close()


if __name__ == '__main__':
    parser = ArgumentParser('GoogLeNet-FCN with PyTorch')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='input batch size for training')
    parser.add_argument('--val-batch-size', type=int, default=8,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', type=float, default=5e-4,
                        help='momentum')
    parser.add_argument('--freeze-bn', action='store_true',
                        help='freeze batch norm during training')
    parser.add_argument('--seed', type=int, default=123, help='manual seed')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--log-dir', type=str, default='logs',
                        help='log directory for Tensorboard log output')
    parser.add_argument('--dataset-dir', type=str, default='data/cityscapes',
                        help='location of the dataset')
    parser.add_argument('--download', action='store_true',
                        help='download dataset')
    parser.add_argument('--augmentations', action='store_true',
                        help='apply extensive data augmentations')

    run(parser.parse_args())
