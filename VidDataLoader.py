import torch
from skimage import io
import numpy as np
import os
from xml.dom import minidom
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder


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
        x_min, x_max, y_min, y_max = int(xml.getElementsByTagName('xmin')[0].firstChild.data), \
                                     int(xml.getElementsByTagName('xmax')[0].firstChild.data), \
                                     int(xml.getElementsByTagName('ymin')[0].firstChild.data), \
                                     int(xml.getElementsByTagName('ymax')[0].firstChild.data)
        return {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max}

    except Exception as e:
        print("Error:", e.__class__, "occurred for img " + img)
        print("Next entry.")
        pass


def crop(img_path, annotations_dir_path):
    """
    Args:
        img_path: the path of requested image.

    Returns:
        an ndarray representing the cropped bb image.
    Notes:
         the given image is read as a PIL image to be cropped; the it is converted as a normalized float32 ndarray
         for further manipulation in nn.
    """
    img = Image.fromarray(io.imread(img_path).astype('uint8'))
    ann_path = (annotations_dir_path+img_path.split('/train/')[1]).split('.')[0]+'.xml'
    img_crop_coord = parse_annotation(img_path, ann_path)
    height, width = img_crop_coord['y_max'] - img_crop_coord['y_min'], img_crop_coord['x_max'] - img_crop_coord['x_min']
    img_cropped = TF.crop(img, img_crop_coord['y_min'], img_crop_coord['x_min'], height, width)
    return np.array(img_cropped).astype('float32') / 255, img_crop_coord


class VidDataset(ImageFolder):
    """VID dataset."""

    def __init__(self, xml_annotations_dir: str, root_dir: str, transform=None):
        """
        Args:
            xml_annotations_dir (string): Path to the dir with images annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        Notes:
             xml_xml_annotations_dir and root_dir have the same structure since
              each images folder has its annotations folder counterpart.
        """
        super().__init__(root_dir, transform)
        self.annotations_dir = xml_annotations_dir

        # for idx, ann in enumerate(list_files(xml_annotations_dir)):
        #     if not idx % step:
        #         # Substitute 'Annotations' with 'Data' and 'xml' with '.JPEG -> img path
        #         ann_path = list(PurePath(ann).parts)
        #         # 3
        #         ann_path[4], ann_path[-1] = 'Data', ann_path[-1].split('.')[0] + '.JPEG'
        #         img = '/'.join(ann_path)
        #         img_coord = parse_annotation(img, ann)
        #         # add image to dataset iff both image and its coordinates exist
        #         if img_coord and os.path.exists(img):
        #             df_row = [img] + img_coord
        #             print('Processed ', idx // step + 1, ' images: ', df_row[0], ann, '\n')
        #             self.box_frame = self.box_frame.append(
        #                 pd.Series(df_row, index=['img', 'x_min', 'x_max', 'y_min', 'y_max']),
        #                 ignore_index=True)
        #         else:
        #             print('No loaded: ', img, '\n')
        # print("Dataset loading completed. Loaded", + len(self.box_frame.index), "images \n")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, crop_coords = crop(self.imgs[idx][0], self.annotations_dir)
        crop_coords = np.array(list(crop_coords.values())).astype('float32')
        sample = {'image': image, 'crop_coord': crop_coords}

        if self.transform:
            sample = self.transform(sample)

        return sample
