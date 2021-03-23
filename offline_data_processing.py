import argparse
import datetime
import os
import pickle
from copy import deepcopy

from VidDataLoaderLight import VidDatasetLight
import preprocessing
from util import deserialize_obj
from pathlib import Path
from PIL import Image

import torchvision.transforms as transforms


def parse_args():
    parser = argparse.ArgumentParser(
        description='Vid Dataset preprocessor for a lighter dataset and boost vid classifier.')
    parser.add_argument('dataset_pkl', metavar='PKL_DATASET', help='path to a serialized dataset (required).')
    parser.add_argument('labels_pkl', metavar='LABELS', help='path to serialized labels (required).')
    parser.add_argument('input_path', metavar='IN_PATH', help='path to existing dataset folder (required).')
    parser.add_argument('output_path', metavar='OUT_PATH', help='path to new dataset folder (required).')
    parser.add_argument('data', metavar='DATA_DIR', type=str, help='Dir where dataset images are saved (required).')
    parser.add_argument('--load_step', metavar='STEP', type=int, default=1,
                        help='step by which load images from Data folder. Default: 1 (each image will be loaded.')
    parser.add_argument('--img_ext', metavar='EXT', type=str, default='JPEG',
                        help='Cropped image extension (default: JPEG')
    return parser.parse_args()


# percentage data of validation set wrt training set (374133 to 1122397)
VALIDATION_PERC = 15


def main(args):
    # 1. create dir structure for new dataset
    inputpath = args.input_path  # 'C:/mnt/ILSVRC2017_VID/'
    outputpath = args.output_path  # 'C:/ILSVRC2017_VID2/'

    for dirpath, dirnames, filenames in os.walk(inputpath):
        structure = os.path.join(outputpath, dirpath[len(inputpath):])
        if not os.path.isdir(structure):
            os.mkdir(structure)
        else:
            print("Folder does already exits!")

    # 2. deserialize labels and dataset
    labels = deserialize_obj(args.labels_pkl)
    # Dataset: Get dataset from serialized object
    complete_dataset = deserialize_obj(args.dataset_pkl)
    complete_dataset.vid_labels = labels

    # Dataset manipulation and labels assignment (both for train and val)
    tra = [preprocessing.Rescale((224, 224)),
           preprocessing.ToTensor()]

    train_dataset = deepcopy(complete_dataset)
    train_dataset.imgs = complete_dataset.imgs[0::args.load_step]
    train_dataset.samples = complete_dataset.samples[0::args.load_step]

    # val dataset. Since there are not labels for the original one, training set is used, considering images which
    # are not in training set.
    val_dataset = deepcopy(complete_dataset)
    remaining_imgs = list(set(complete_dataset.imgs) - set(train_dataset.imgs))
    remaining_samples = list(set(complete_dataset.samples) - set(train_dataset.samples))
    val_step = len(remaining_imgs) // (len(train_dataset.imgs) * VALIDATION_PERC // 100)

    val_dataset.imgs = remaining_imgs[0::val_step]
    val_dataset.samples = remaining_samples[0::val_step]

    datasets = {'train_dataset': train_dataset, 'val_dataset': val_dataset}

    # 3. crop img and save for bot train and validation dataset
    for k in datasets.keys():
        print('Process dataset', k)
        for idx, img_dict in enumerate(datasets[k]):
            crop_coords = img_dict['crop_coord']
            height, width = crop_coords[3] - crop_coords[2], crop_coords[1] - crop_coords[0]

            crop_coords = img_dict['crop_coord']
            im = Image.open(img_dict['name'])
            im1 = im.crop((crop_coords[0], crop_coords[2], crop_coords[0] + width, crop_coords[3]))
            path = os.path.join(args.output_path, Path(img_dict['name'].split(args.input_path)[1]).parent)
            im1.save(os.path.join(path, Path(img_dict['name']).name), args.img_ext)
            if not idx % 100:
                print('Saved', idx, 'images.')

    # 4. pickle obtained dataset of cropped images
    light_dataset = VidDatasetLight(root_dir=args.data, transform=transforms.Compose(tra), labels=labels)
    serialize(light_dataset, tag='cropped_vid_dataset')


def serialize(dataset, tag=None):
    print('Serializing ' + tag + ' dataset...')
    now = datetime.datetime.now()
    date_str = f'{now.year}.{now.month}.{now.day}_{now.hour}_{now.minute}_{now.second}'
    filename = f'{tag}_{date_str}'
    with open(args.data+filename + '.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print('Task terminated.')


if __name__ == '__main__':
    args = parse_args()
    main(args)
