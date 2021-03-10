from skimage import io
import numpy as np
from xml.dom import minidom
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.datasets import ImageFolder


def parse_annotation(ann):
    """
    Args:
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
        print("Error:", e.__class__, "occurred for data " + ann)
        return None


def crop(img_path, annotations_dir_path):
    """
    Args:
        img_path: the path of requested image.
        annotations_dir_path: the path where annotations are stored.

    Returns:
        an ndarray representing the cropped bb image.
    Notes:
         the given image is read as a PIL image to be cropped; the it is converted as a normalized float32 ndarray
         for further manipulation in nn.
    """
    ann_path = (annotations_dir_path + img_path.split('/train/')[1]).split('.')[0]+'.xml'
    img_crop_coord = parse_annotation(ann_path)
    if img_crop_coord:
        img = Image.fromarray(io.imread(img_path).astype('uint8'))
        height, width = img_crop_coord['y_max'] - img_crop_coord['y_min'], img_crop_coord['x_max'] - img_crop_coord['x_min']
        img_cropped = TF.crop(img, img_crop_coord['y_min'], img_crop_coord['x_min'], height, width)
        return {'crop': np.array(img_cropped).astype('float32') / 255, 'coords': img_crop_coord}
    else:
        return None


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

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        cropped_image = crop(self.imgs[idx][0], self.annotations_dir)
        if cropped_image is not None:
            crop_coords = np.array(list(cropped_image['coords'].values())).astype('float32')
            sample = {'image': cropped_image['crop'], 'crop_coord': crop_coords, 'name': self.imgs[idx][0]}

            if self.transform:
                sample = self.transform(sample)

            return sample
        else:
            return None



