import argparse
import os
from xml.dom import minidom


def parse_args():
    parser = argparse.ArgumentParser(
        description='Data cleaner for deepcluster. It deletes not annotated images in a vast dataset.')
    parser.add_argument('data', metavar='DATA_DIR', type=str, help='Dir where dataset images are saved.')
    parser.add_argument('ann', metavar='ANN_DIR', type=str, help='Dir where images annotations are saved.')
    return parser.parse_args()


def list_files(root_path):
    """
    Args:
        root_path (string): root path of dir to be walked.

    Returns:
        path_list (list): all path found in walked dir.
    """
    path_list = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            path_list.append(os.path.join(path, name))
    return path_list


def parse_annotation(img, ann):
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

    except Exception as e:
        print("No annotations are provided for", img)
        return False


def main(args):
    data, annotations = list_files(args.data), list_files(args.ann)
    print("Number of images:", len(data), "\nNumber of annotations:", len(annotations))
    for idx, img in enumerate(data):
        ann = (args.ann + img.split('/train/')[1]).split('.')[0] + '.xml'
        if not parse_annotation(img, ann):
            os.remove(img)
            print("Removed" + img + "\n")
            print("Analyzed ", idx, "images")
    print('Task terminated')


if __name__ == '__main__':
    args = parse_args()
    main(args)
