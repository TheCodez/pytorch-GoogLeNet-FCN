import os
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar, TensorboardLogger
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Events, create_supervised_evaluator, Engine
from ignite.metrics import RunningAverage, Loss
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from googlenet_fcn.datasets.cityscapes import CityscapesDataset
from googlenet_fcn.datasets.transforms.transforms import Compose, ColorJitter, ToTensor, \
    RandomHorizontalFlip, ConvertIdToTrainId, RandomGaussionBlur, RandomAffine, RandomApply, Normalize
from googlenet_fcn.metrics.confusion_matrix import ConfusionMatrix, IoU
from googlenet_fcn.model.googlenet_fcn import GoogLeNetFCN
from googlenet_fcn.utils import save


def get_data_loaders(data_dir, batch_size, val_batch_size, num_workers):
    joint_transforms = Compose([
        RandomHorizontalFlip(),
        RandomApply([RandomAffine(scale=(0.9, 1.2), shear=(-10, 10))]),
        ColorJitter(0.2, 0.2, 0.2),
        RandomGaussionBlur(),
        ToTensor(),
        ConvertIdToTrainId(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_joint_transforms = Compose([
        ToTensor(),
        ConvertIdToTrainId(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(CityscapesDataset(root=data_dir, split='train', transforms=joint_transforms),
                              batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(CityscapesDataset(root=data_dir, split='val', transforms=val_joint_transforms),
                            batch_size=val_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def run(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)

    num_classes = CityscapesDataset.num_classes()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GoogLeNetFCN(num_classes)
    model.init_from_googlenet()

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Using %d GPU(s)" % device_count)
        model = nn.DataParallel(model)
        args.batch_size = device_count * args.batch_size
        args.val_batch_size = device_count * args.val_batch_size

    train_loader, val_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.val_batch_size,
                                                args.num_workers)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    optimizer = optim.SGD([{'params': [param for name, param in model.named_parameters() if name.endswith('weight')],
                            'lr': args.lr, 'weight_decay': 5e-4},
                           {'params': [param for name, param in model.named_parameters() if name.endswith('bias')],
                            'lr': args.lr * 2}], momentum=args.momentum, lr=args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            args.start_iteration = checkpoint.get('iteration', 0)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("Loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'".format(args.resume))

    def _prepare_batch(batch, non_blocking=True):
        image, target = batch

        return (convert_tensor(image, device=device, non_blocking=non_blocking),
                convert_tensor(target, device=device, non_blocking=non_blocking))

    def _update(engine, batch):
        model.train()

        if engine.state.iteration % args.grad_accum == 0:
            optimizer.zero_grad()
        image, target = _prepare_batch(batch)
        pred = model(image)
        loss = criterion(pred, target) / args.grad_accum
        loss.backward()
        if engine.state.iteration % args.grad_accum == 0:
            optimizer.step()

        return loss.item()

    trainer = Engine(_update)

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

    tb_logger.attach(evaluator,
                     log_handler=OutputHandler(tag='validation',
                                               metric_names=['loss', 'IoU'],
                                               global_step_transform=_global_step_transform),
                     event_name=Events.EPOCH_COMPLETED)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint(engine):
        iou = engine.state.metrics['IoU'] * 100.0
        mean_iou = iou.mean()

        name = 'epoch{}_mIoU={:.1f}.pth'.format(trainer.state.epoch, mean_iou)
        file = {'model': model.state_dict(), 'epoch': trainer.state.epoch, 'iteration': engine.state.iteration,
                'optimizer': optimizer.state_dict(), 'args': args}

        save(file, args.output_dir, 'checkpoint_{}'.format(name))
        save(model.state_dict(), args.output_dir, 'model_{}'.format(name))

    @trainer.on(Events.STARTED)
    def initialize(engine):
        if args.resume:
            engine.state.epoch = args.start_epoch
            engine.state.iteration = args.start_iteration

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
    parser.add_argument('--val-batch-size', type=int, default=4,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument('--epochs', type=int, default=150,
                        help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-10,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='momentum')
    parser.add_argument('--seed', type=int, default=123, help='manual seed')
    parser.add_argument('--output-dir', default='checkpoints',
                        help='directory to save model checkpoints')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--start-iteration', default=0, type=int, metavar='N',
                        help='manual iteration number (useful on restarts)')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='how many batches to wait before logging training status')
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="log directory for Tensorboard log output")
    parser.add_argument("--dataset-dir", type=str, default="data/cityscapes",
                        help="location of the dataset")
    parser.add_argument('--grad-accum', type=int, default=1,
                        help='grad accumulation')

    run(parser.parse_args())
