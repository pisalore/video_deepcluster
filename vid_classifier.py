import argparse
import os
import pickle
import time
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np

from util import deserialize_obj, AverageMeter, Logger
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Classifier for Vid Dataset')

    parser.add_argument('--dataset_pkl', metavar='PKL', help='path to a serialized dataset.')
    parser.add_argument('--labels_pkl', metavar='LABELS', help='path to serialized labels.')
    parser.add_argument('--model', metavar='PRETRAINED_MODEL', help='path to video deepcluster pretrained model.')
    parser.add_argument('--load_step', metavar='STEP', type=int, default=1,
                        help='step by which load images from Data folder. Default: 1 (each image will be loaded.')
    parser.add_argument('--k', type=int, default=300,
                        help='number of cluster obtained from for k-means in video deepcluster (default: 300)')
    parser.add_argument('--out_classes', type=int, default=30,
                        help='number of desired prediction output classes (default: 30, VidDataset)')
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


# percentage data of validation set wrt training set (374133 to 1122397)
VALIDATION_PERC = 15


def main(args):
    # fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # CNN
    if args.verbose:
        print('Architecture: {}'.format(args.arch))
    model = models.__dict__[args.arch](sobel=args.sobel)
    # top layer has a (4096, k) shape, where k is the number of cluster set in video deepcluster routine. It must be inserted in order
    # to correctly load model
    model.top_layer = nn.Linear(4096, args.k)
    model.features = torch.nn.DataParallel(model.features)
    checkpoint = torch.load(args.model)
    model.load_state_dict(checkpoint['state_dict'])
    fd = int(model.top_layer.weight.size()[1])
    model.cuda()
    cudnn.benchmark = True
    print('Correctly loaded pretrained model', args.model)
    # set last fully connected layer
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=False).cuda())
    model.classifier = nn.Sequential(*mlp)
    model.top_layer = nn.Linear(fd, args.out_classes)
    model.top_layer.weight.data.normal_(0, 0.01)
    model.top_layer.bias.data.zero_()
    model.top_layer.cuda()

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
    complete_dataset, val_dataset, labels = None, None, None
    if args.labels_pkl:
        labels = deserialize_obj(args.labels_pkl)

    # Dataset: Get dataset from serialized object
    complete_dataset = deserialize_obj(args.dataset_pkl)
    complete_dataset.vid_labels = labels

    # Dataset manipulation and labels assignment (both for train and val)
    train_dataset = deepcopy(complete_dataset)
    train_dataset.imgs = complete_dataset.imgs[0::args.load_step]
    train_dataset.samples = complete_dataset.samples[0::args.load_step]

    # val dataset. Since there are not labels for the original one, training set is used, considering images which
    # are not in training set.
    val_dataset = deepcopy(complete_dataset)
    remaining_imgs = list(set(complete_dataset.imgs) - set(train_dataset.imgs))
    remaining_samples = list(set(complete_dataset.samples) - set(train_dataset.samples))
    train_dataset.imgs = remaining_imgs
    train_dataset.samples = remaining_samples

    print("Training set dimension: {0}\n"
          "Validation set dimension: {1}\n".format(len(train_dataset.imgs), len(val_dataset.imgs)))

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch, shuffle=True)

    print('Training starts.')
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}
    since = time.time()
    val_acc_history = train(dataloaders, model, criterion, optimizer)
    with open(os.path.join(args.exp, 'val_acc_history.pkl'), 'wb') as f:
        pickle.dump(val_acc_history, f)
    print('Elapsed time: {} '.format(time.time() - since))


def train(data_loaders, model, crit, opt):
    """Training of the CNN.
        Args:
            @param data_loaders: (torch.utils.data.DataLoader) Dataloaders dict for train and val phases
            model (nn.Module): CNN
            crit (torch.nn): loss
            opt (torch.optim.SGD): optimizer for every parameters with True
                                   requires_grad in model except top layer
            epoch (int)
    """
    # logger
    epochs_log = Logger(os.path.join(args.exp, 'epochs'))
    val_acc_history = []

    best_acc = 0.0

    # create an optimizer for the last fc layer
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=args.lr,
        weight_decay=10 ** args.wd,
    )

    for epoch in range(args.start_epoch, args.epochs):
        losses = AverageMeter()
        print('\n')
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                # training mode
                model.train()
            else:
                # evaluate mode
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for i, sample in enumerate(data_loaders[phase]):
                input_var = torch.autograd.Variable(sample['image'].cuda())
                labels = torch.as_tensor(np.array(sample['label'], dtype='int_'))
                labels = labels.type(torch.LongTensor).cuda()

                with torch.set_grad_enabled(phase == 'train'):

                    output = model(input_var)
                    loss = crit(output, labels)
                    _, preds = torch.max(output, 1)

                    if phase == 'train':
                        # compute gradient and do SGD step
                        opt.zero_grad()
                        optimizer_tl.zero_grad()
                        loss.backward()
                        opt.step()
                        optimizer_tl.step()

                # record loss
                losses.update(loss.data, input_var.size(0))
                running_loss += loss.item() * input_var.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if args.verbose:
                    print('Epoch: [{0}][{1}/{2}]\n'
                          'Running loss:: {loss:.4f} \n'
                          'Running corrects: ({corrects:.4f}) \n'
                          .format(epoch+1, i+1, len(data_loaders['train']), loss=(loss.item() * input_var.size(0)), corrects=(torch.sum(preds == labels.data))))

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
            if phase == 'val':
                val_acc_history.append([epoch_loss, epoch_acc])

            epochs_log.log([phase, epoch+1, epoch_loss, epoch_acc])

        # save the model
        torch.save({'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict()},
                   os.path.join(args.exp, 'fine_tuning.pth.tar'))

    return val_acc_history


if __name__ == '__main__':
    args = parse_args()
    main(args)
