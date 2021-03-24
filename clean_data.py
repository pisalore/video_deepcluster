import argparse
import datetime
import pickle
import logging
from xml.dom import minidom
from PIL import Image


import torchvision.transforms as transforms

import preprocessing
from VidDataLoaderLight import VidDatasetLight


def parse_args():
    parser = argparse.ArgumentParser(
        description='Data cleaner for deepcluster. It deletes not annotated images in a vast dataset.')
    parser.add_argument('data', metavar='DATA_DIR', type=str, help='Dir where dataset images are saved.')
    parser.add_argument('ann', metavar='ANN_DIR', type=str, help='Dir where images annotations are saved.')
    parser.add_argument('labels_txt', metavar='LBL_TXT', type=str,
                        help='txt file where label where labels are provided along with objects names.')
    parser.add_argument('--data_type', metavar='DATA_TYPE', type=str, default='train',
                        help='Data type to be cleaned; it must match data dir name. For example: train, val. Default: train.')
    parser.add_argument('--output', metavar='LOG', type=str, default='', help='Dir where useful logs will be saved.')
    return parser.parse_args()


def serialize(dataset, tag=None):
    print('Serializing cleaned dataset...')
    now = datetime.datetime.now()
    date_str = f'{now.year}.{now.month}.{now.day}_{now.hour}_{now.minute}_{now.second}'
    filename = f'{tag}_{date_str}'
    with open(args.output + filename + '.pkl', 'wb') as f:
        pickle.dump(dataset, f)


def parse_annotation(ann):
    """
    Args:
        img (string): the image to which coordinates are related
        ann: an xml file with bb annotations: xmin, xmax, ymin, ymax
    Returns:
        a dictionary with bb coordinates and category for a given image.
    Exception:
        if coordinates are not present in xml file.
    """
    try:
        xml = minidom.parse(ann)
        crop_coord = {'xmin': int(xml.getElementsByTagName('xmin')[0].firstChild.data),
                      'xmax': int(xml.getElementsByTagName('xmax')[0].firstChild.data),
                      'ymin': int(xml.getElementsByTagName('ymin')[0].firstChild.data),
                      'ymax': int(xml.getElementsByTagName('ymax')[0].firstChild.data)
                      }
        label = str(xml.getElementsByTagName('name')[0].firstChild.data)
        return crop_coord, label

    except Exception:
        return None, None


def main(args):
    print('Load dataset...')
    logging.basicConfig(filename=args.output + 'clean_{}.log'.format(args.data_type), filemode='w', level=logging.INFO,
                        format='%(name)s - %(levelname)s - %(message)s')

    # Prepare dataset
    tra = [preprocessing.Rescale((224, 224)),
           preprocessing.ToTensor()]
    dataset = VidDatasetLight(root_dir=args.data, transform=transforms.Compose(tra))

    # Prepare labels and crop coords structures
    labels, dataset_labels, dataset_crop_coords = {}, {}, {}
    with open(args.labels_txt, 'r') as f:
        for line in f:
            lbl = line.rstrip().split(' ')
            labels[lbl[0]] = {'code': lbl[1], 'class': lbl[2]}

    print('Dataset and labels loaded.\n')
    print('Clean dataset...\n')
    not_annotated_imgs_idx = []

    # save images information (labels and bb coordinates) and discard erroneous samples
    for idx, img in enumerate(dataset.imgs):
        ann = (args.ann + img[0].split('/' + args.data_type + '/')[1]).split('.')[0] + '.xml'
        img_crop_coords, img_label_name = parse_annotation(ann)
        if None in (img_crop_coords, img_label_name):
            not_annotated_imgs_idx.append(idx)
            logging.info("Removed " + img[0] + "\n")
        else:
            img_width = img_crop_coords['xmax'] - img_crop_coords['xmin']
            im = Image.open(img[0])
            im.crop((img_crop_coords['xmin'], img_crop_coords['ymin'], img_crop_coords['xmin'] + img_width, img_crop_coords['ymax'])).save(img[0])
            dataset_labels[img[0]] = labels[img_label_name]
            dataset_crop_coords[img[0]] = img_crop_coords
        if not (idx + 1) % 1000:
            print("Analyzed ", idx + 1, "images")
    for i in sorted(not_annotated_imgs_idx, reverse=True):
        del dataset.imgs[i]
    logging.info("Removed " + str(len(not_annotated_imgs_idx)) + " images.\n")

    # Complete dataset definition with labels and crop coordinates
    dataset.vid_labels = dataset_labels
    dataset.crop_coords = dataset_crop_coords

    serialize(dataset, tag='vid_dataset_' + args.data_type)
    print('Task terminated. Find log and pkl file at ' + args.output)


if __name__ == '__main__':
    args = parse_args()
    main(args)
