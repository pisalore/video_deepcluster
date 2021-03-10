import argparse
import datetime
import pickle
import logging
from xml.dom import minidom

import torchvision.transforms as transforms

import preprocessing
from VidDataLoader import VidDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='Data cleaner for deepcluster. It deletes not annotated images in a vast dataset.')
    parser.add_argument('data', metavar='DATA_DIR', type=str, help='Dir where dataset images are saved.')
    parser.add_argument('ann', metavar='ANN_DIR', type=str, help='Dir where images annotations are saved.')
    parser.add_argument('--log', metavar='LOG', type=str, default='', help='Dir where useful logs will be saved.')
    return parser.parse_args()


def serialize(dataset, tag='vid_dataset'):
    print('Serializing cleaned dataset...')
    now = datetime.datetime.now()
    date_str = f'{now.year}.{now.month}.{now.day}_{now.hour}_{now.minute}_{now.second}'
    filename = f'{tag}_{date_str}'
    with open(filename + '.pkl', 'wb') as f:
        pickle.dump(dataset, f)


def parse_annotation(ann):
    """
    Args:
        img (string): the image to which coordinates are related
        ann: an xml file with bb annotations: xmin, xmax, ymin, ymax
    Returns:
        a list with bb coordinates for a given image.
    Exception:
        if coordinates are not present in xml file.
    """
    try:
        xml = minidom.parse(ann)
        int(xml.getElementsByTagName('xmin')[0].firstChild.data), \
        int(xml.getElementsByTagName('xmax')[0].firstChild.data), \
        int(xml.getElementsByTagName('ymin')[0].firstChild.data), \
        int(xml.getElementsByTagName('ymax')[0].firstChild.data)
        return True

    except Exception:
        return False


def main(args):
    print('Load dataset...')
    logging.basicConfig(filename=args.log + 'clean.log', filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

    tra = [preprocessing.Rescale((224, 224)),
           preprocessing.ToTensor()]
    dataset = VidDataset(xml_annotations_dir=args.ann, root_dir=args.data, transform=transforms.Compose(tra))
    print('Dataset loaded.\n')
    print('Clean dataset...\n')
    not_annotated_imgs_idx = []
    for idx, img in enumerate(dataset.imgs):
        ann = (args.ann + img[0].split('/train/')[1]).split('.')[0] + '.xml'
        if not parse_annotation(ann):
            not_annotated_imgs_idx.append(idx)
            logging.info("Removed " + img[0] + "\n")
        if not (idx+1) % 1000:
            print("Analyzed ", idx+1, "images")
    for i in sorted(not_annotated_imgs_idx, reverse=True):
        del dataset.imgs[i]
    logging.info("Removed " + str(len(not_annotated_imgs_idx)) + " images.\n")
    serialize(dataset)
    print('Task terminated. Find log at ' + args.log)


if __name__ == '__main__':
    args = parse_args()
    main(args)
