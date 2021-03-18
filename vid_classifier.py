import argparse
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

from util import deserialize_obj, AverageMeter
import models

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Classifier for Vid Dataset')

    parser.add_argument('--data', metavar='DIR', help='path to dataset')
    parser.add_argument('--ann', metavar='ANN_DIR', help='path to annotations')
    parser.add_argument('--dataset_pkl', metavar='PKL', help='path to a serialized dataset.')
    parser.add_argument('--labels_pkl', metavar='LABELS', help='path to serialized labels.')
    parser.add_argument('--load_step', metavar='STEP', type=int, default=1,
                        help='step by which lodead images from Data folder. Default: 1 (each image will be loaded.')
    parser.add_argument('--arch', '-a', type=str, metavar='ARCH',
                        choices=['alexnet', 'vgg16'], default='alexnet',
                        help='CNN architecture (default: alexnet)')
    parser.add_argument('--sobel', action='store_true', help='Sobel filtering')
    parser.add_argument('--lr', default=0.05, type=float,
                        help='learning rate (default: 0.05)')
    parser.add_argument('--wd', default=-5, type=float,
                        help='weight decay pow (default: -5)')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of total epochs to run (default: 200)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts) (default: 0)')
    parser.add_argument('--batch', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum (default: 0.9)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to checkpoint (default: None)')
    parser.add_argument('--checkpoints', type=int, default=25000,
                        help='how many iterations between two checkpoints (default: 25000)')
    parser.add_argument('--seed', type=int, default=31, help='random seed (default: 31)')
    parser.add_argument('--exp', type=str, default='', help='path to exp folder')
    parser.add_argument('--verbose', action='store_true', help='chatty')
    return parser.parse_args()


def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    fd = int(model.top_layer.weight.size()[1])
    model.top_layer = None
    # model.features = torch.nn.DataParallel(model.features)
    model.cuda()
    cudnn.benchmark = True

    print('CNN builded.')

    # create optimizer
    optimizer = torch.optim.SGD(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=10 ** args.wd,
    )
    print('Optimizer created.')

    # define loss function
    criterion = nn.CrossEntropyLoss().cuda()

    # Labels: they have been added in a second moment, with another processing of ImageSets files inside VID dataset.
    print('Start loading dataset...')
    train_dataset, labels = None, None
    if args.labels_pkl:
        labels = deserialize_obj(args.labels_pkl)

    # Dataset: Get dataset from serialized object
    train_dataset = deserialize_obj(args.dataset_pkl)

    # Dataset manipulation and labels assignment
    train_dataset.imgs = train_dataset.imgs[0::args.load_step]
    train_dataset.samples = train_dataset.samples[0::args.load_step]
    train_dataset.vid_labels = labels

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)

    # set last fully connected layer
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=False).cuda())
    model.classifier = nn.Sequential(*mlp)
    model.top_layer = nn.Linear(fd, 30)
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()
    model.top_layer.cuda()

    print('Training starts.')
    for epoch in range(args.start_epoch, args.epochs):
        loss = train(train_dataloader, model, criterion, optimizer, epoch)


def train(loader, model, crit, opt, epoch):
    """Training of the CNN.
        Args:
            loader (torch.utils.data.DataLoader): Data loader
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    # switch to train mode
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10 ** args.wd,
    )

    end = time.time()
    for i, sample in enumerate(loader):
        data_time.update(time.time() - end)

        input_var = torch.autograd.Variable(sample['image'].cuda())
        labels = torch.as_tensor(np.array(sample['label'], dtype='int_'))
        labels = labels.type(torch.LongTensor).cuda()
        print(labels.size(), input_var.size())
        output = model(input_var)
        loss = crit(output, labels)

        # record loss
        losses.update(loss.data, input_var.size(0))

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.verbose and (i % 200) == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss: {loss.val:.4f} ({loss.avg:.4f})'
                  .format(epoch, i, len(loader), loss=losses))

    return losses.avg


if __name__ == '__main__':
    args = parse_args()
    main(args)
