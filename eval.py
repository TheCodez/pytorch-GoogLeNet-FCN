from argparse import ArgumentParser

import torch
import torch.nn as nn
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, Engine
from ignite.metrics import Loss
from ignite.utils import convert_tensor
from torch.utils.data import DataLoader

from googlenet_fcn.datasets.cityscapes import CityscapesDataset
from googlenet_fcn.datasets.transforms.transforms import Compose, ToTensor, ConvertIdToTrainId, Normalize
from googlenet_fcn.metrics.confusion_matrix import ConfusionMatrix, IoU
from googlenet_fcn.model.googlenet_fcn import GoogLeNetFCN, googlenet_fcn


def get_data_loaders(data_dir, batch_size, num_workers):
    val_transforms = Compose([
        ToTensor(),
        ConvertIdToTrainId(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_loader = DataLoader(CityscapesDataset(root=data_dir, split='val', transforms=val_transforms),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return val_loader


def run(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = CityscapesDataset.num_classes()
    if args.checkpoint:
        model = GoogLeNetFCN(num_classes)
        model.load_state_dict(torch.load(args.checkpoint))
    else:
        model = googlenet_fcn(pretrained=True, num_classes=num_classes)

    device_count = torch.cuda.device_count()
    if device_count > 1:
        print("Using %d GPU(s)" % device_count)
        model = nn.DataParallel(model)
        args.batch_size = device_count * args.batch_size

    val_loader = get_data_loaders(args.dataset_dir, args.batch_size, args.num_workers)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    def _prepare_batch(batch, non_blocking=True):
        image, target = batch

        return (convert_tensor(image, device=device, non_blocking=non_blocking),
                convert_tensor(target, device=device, non_blocking=non_blocking))

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            image, target = _prepare_batch(batch)
            pred = model(image)

            return pred, target

    evaluator = Engine(_inference)
    cm = ConfusionMatrix(num_classes)
    IoU(cm).attach(evaluator, 'IoU')
    Loss(criterion).attach(evaluator, 'loss')

    pbar = ProgressBar(persist=True, desc='Eval')
    pbar.attach(evaluator)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def run_validation(engine):
        metrics = engine.state.metrics
        loss = metrics['loss']
        iou = metrics['IoU'] * 100.0
        mean_iou = iou.mean()

        pbar.log_message("Validation results:\nLoss: {:.2e}\nmIoU: {:.1f}"
                         .format(loss, mean_iou))

    print("Start validation")
    evaluator.run(val_loader, max_epochs=1)


if __name__ == '__main__':
    parser = ArgumentParser('GoogLeNet FCN Eval with PyTorch')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='input batch size for validation')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    parser.add_argument("--dataset-dir", type=str, default="data/cityscapes",
                        help="location of the dataset")
    parser.add_argument("--checkpoint", type=str,
                        help="the checkpoint to eval")

    run(parser.parse_args())
