import os
import sys
import logging
import argparse
import datetime
import itertools
import torch

from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from vision.ssd.ssd import MatchPrior
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.mobilenet_v3_ssd_lite import create_mobilenetv3_ssd_lite
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

from vision.datasets.voc_dataset import VOCDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.utils.misc import Timer, freeze_net_layers, store_labels

from eval_ssd import MeanAPEvaluator


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With PyTorch')

# Params for datasets
parser.add_argument('--datasets', '--data', nargs='+',
                    default=["data"], help='Dataset directory path')

# Params for network
parser.add_argument('--net', default="mb1-ssd",
                    help="The network architecture, it can be mb1-ssd, mb1-ssd-lite, mb2-ssd-lite or mb3-ssd-lite.")
parser.add_argument('--freeze-base-net', action='store_true',
                    help="Freeze base net layers.")
parser.add_argument('--freeze-net', default=False, action='store_true',
                    help="Freeze all the layers except the prediction head.")

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')

# Params for ASGD
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate')
parser.add_argument('--weight-decay', default=5e-4, type=float,
                    help='Weight decay for SGD')

# Train params
parser.add_argument('--batch-size', '--batch', default=4, type=int,
                    help='Batch size for training')
parser.add_argument('--num-epochs', '--epoch', default=30, type=int,
                    help='the number epochs')
parser.add_argument('--num-workers', '--workers', default=2, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--validation-mean-ap', action='store_true',
                    help='Perform computation of Mean Average Precision (mAP) during validation')
parser.add_argument('--debug-steps', default=10, type=int,
                    help='Set the debug log output frequency.')
parser.add_argument('--use-cuda', default=True, action='store_true',
                    help='Use CUDA to train model')
parser.add_argument('--checkpoint-folder', '--model', default='models/',
                    help='Directory for saving checkpoint models')

args = parser.parse_args()

# Set logging
logging.basicConfig(stream=sys.stdout, level=getattr(logging, 'INFO', logging.INFO),
                    format='%(asctime)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

# Tensorboard
tensorboard = SummaryWriter(log_dir=os.path.join(
    args.checkpoint_folder, "tensorboard"))

# CPU or GPU training
DEVICE = torch.device("cuda:0" if torch.cuda.is_available()
                      and args.use_cuda else "cpu")
if args.use_cuda and torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    logging.info("Using CUDA...")

# Training function


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)

    train_loss = 0.0
    train_regression_loss = 0.0
    train_classification_loss = 0.0

    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    num_batches = 0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(
            confidence, locations, labels, boxes)
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_regression_loss += regression_loss.item()
        train_classification_loss += classification_loss.item()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logging.info(
                f"Epoch: {epoch}, Step: {i}/{len(loader)}, " +
                f"Avg Loss: {avg_loss:.4f}, " +
                f"Avg Regression Loss {avg_reg_loss:.4f}, " +
                f"Avg Classification Loss: {avg_clf_loss:.4f}"
            )
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

        num_batches += 1

    train_loss /= num_batches
    train_regression_loss /= num_batches
    train_classification_loss /= num_batches

    logging.info(
        f"Epoch: {epoch}, " +
        f"Training Loss: {train_loss:.4f}, " +
        f"Training Regression Loss {train_regression_loss:.4f}, " +
        f"Training Classification Loss: {train_classification_loss:.4f}"
    )

    tensorboard.add_scalar('Loss/train', train_loss, epoch)
    tensorboard.add_scalar('Regression Loss/train',
                           train_regression_loss, epoch)
    tensorboard.add_scalar('Classification Loss/train',
                           train_classification_loss, epoch)


def test(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(
                confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

    return running_loss / num, running_regression_loss / num, running_classification_loss / num


if __name__ == '__main__':
    timer = Timer()

    logging.info(args)

    # Make sure that the checkpoint output dir exists
    if args.checkpoint_folder:
        args.checkpoint_folder = os.path.expanduser(args.checkpoint_folder)
        if not os.path.exists(args.checkpoint_folder):
            os.mkdir(args.checkpoint_folder)

    # Select the network architecture and config
    if args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
        pretrained_ssd = 'pretrained/mobilenet-v1-ssd-mp-0_675.pth'
        config.set_image_size(30)
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
        pretrained_ssd = 'pretrained/mobilenet-v1-ssd-mp-0_675.pth'
    elif args.net == 'mb2-ssd-lite':
        def create_net(num): return create_mobilenetv2_ssd_lite(
            num, width_mult=1.0)
        config = mobilenetv1_ssd_config
        pretrained_ssd = 'pretrained/mb2-ssd-lite-mp-0_686.pth'
    elif args.net == 'mb3-ssd-lite':
        def create_net(num): return create_mobilenetv3_ssd_lite(num)
        config = mobilenetv1_ssd_config
        pretrained_ssd = 'pretrained/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
    else:
        logging.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Create data transforms for train/test/val
    train_transform = TrainAugmentation(
        config.image_size, config.image_mean, config.image_std)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)
    test_transform = TestTransform(
        config.image_size, config.image_mean, config.image_std)

    # Load datasets (could be multiple)
    logging.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        dataset = VOCDataset(dataset_path, transform=train_transform,
                             target_transform=target_transform)
        label_file = os.path.join(args.checkpoint_folder, "labels.txt")
        store_labels(label_file, dataset.class_names)
        num_classes = len(dataset.class_names)
        datasets.append(dataset)

    # Create training dataset
    logging.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)
    logging.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    # Create validation dataset
    logging.info("Prepare Validation datasets.")
    val_dataset = VOCDataset(dataset_path, transform=test_transform,
                             target_transform=target_transform, is_test=True)
    logging.info("Validation dataset size: {}".format(len(val_dataset)))
    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)

    # Create the network
    logging.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    # Prepare eval dataset (for mAP computation)
    if args.validation_mean_ap:
        eval_dataset = VOCDataset(dataset_path, is_test=True)
        eval = MeanAPEvaluator(eval_dataset, net, arch=args.net, eval_dir=os.path.join(
            args.checkpoint_folder, 'eval_results'))

    # Freeze certain layers (if requested)
    base_net_lr = args.lr
    extra_layers_lr = args.lr

    if args.freeze_base_net:
        logging.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(
            net.regression_headers.parameters(), net.classification_headers.parameters())
        logging.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    # Load a previous model checkpoint (if requested)
    timer.start("Load Model")

    if args.resume:
        logging.info(f"Resuming from the model {args.resume}")
        net.load(args.resume)
    elif pretrained_ssd:
        logging.info(f"Init from pretrained SSD {pretrained_ssd}")
        net.init_from_pretrained_ssd(pretrained_ssd)

        if not os.path.exists(pretrained_ssd):
            logging.fatal("The net is wrong.")
            parser.print_help(sys.stderr)
            sys.exit(1)

        net.init_from_pretrained_ssd(pretrained_ssd)

    logging.info(
        f'Took {timer.end("Load Model"):.2f} seconds to load the model.')

    # Move the model to GPU
    net.to(DEVICE)

    # Define loss function and optimizer
    criterion = MultiboxLoss(config.priors, iou_threshold=0.5, neg_pos_ratio=3,
                             center_variance=0.1, size_variance=0.2, device=DEVICE)
    optimizer = torch.optim.ASGD(
        params, lr=args.lr, weight_decay=args.weight_decay)

    logging.info(f"Learning rate: {args.lr}")

    # Set learning rate policy
    logging.info("Uses CosineAnnealingLR scheduler.")
    scheduler = CosineAnnealingLR(
        optimizer, args.num_epochs, last_epoch=last_epoch)

    # Train for the desired number of epochs
    logging.info(f"Start training from epoch {last_epoch + 1}.")

    for epoch in range(last_epoch + 1, args.num_epochs):
        train(train_loader, net, criterion, optimizer, device=DEVICE,
              debug_steps=args.debug_steps, epoch=epoch)
        scheduler.step()

        if epoch % 1 == 0 or epoch == args.num_epochs - 1:
            val_loss, val_regression_loss, val_classification_loss = test(
                val_loader, net, criterion, DEVICE)

            logging.info(
                f"Epoch: {epoch}, " +
                f"Validation Loss: {val_loss:.4f}, " +
                f"Validation Regression Loss {val_regression_loss:.4f}, " +
                f"Validation Classification Loss: {val_classification_loss:.4f}"
            )

            tensorboard.add_scalar('Loss/val', val_loss, epoch)
            tensorboard.add_scalar('Regression Loss/val',
                                   val_regression_loss, epoch)
            tensorboard.add_scalar(
                'Classification Loss/val', val_classification_loss, epoch)

            if args.validation_mean_ap:
                mean_ap, class_ap = eval.compute()
                eval.log_results(mean_ap, class_ap, f"Epoch: {epoch}, ")

                tensorboard.add_scalar(
                    'Mean Average Precision/val', mean_ap, epoch)

                for i in range(len(class_ap)):
                    tensorboard.add_scalar(
                        f"Class Average Precision/{eval_dataset.class_names[i+1]}", class_ap[i], epoch)

            model_path = os.path.join(
                args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            net.save(model_path)
            logging.info(f"Saved model {model_path}")

    logging.info("Task done, exiting program.")
    tensorboard.close()
