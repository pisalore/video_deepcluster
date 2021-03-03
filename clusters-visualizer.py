import argparse
import pickle
import os
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from skimage import io


def parse_args():
    parser = argparse.ArgumentParser(description='Data visualizer for deepcluster')
    parser.add_argument('clusters', metavar='CLUSTERS', type=str,
                        help='path to clusters pickle file created by deepcluster')
    parser.add_argument('dataset', metavar='DATASET', type=str, help='path to image dataset')
    parser.add_argument('--save_dir', metavar='SAVE_DIR', type=str, help='Dir where save clusters subplot images')
    parser.add_argument('--img_num', metavar='IMG_NUM', type=int, help='num of images for cluster to be visualized')
    return parser.parse_args()


def clusters_reader(clusters):
    with open(clusters, 'rb') as f:
        return pickle.load(f)


def save_cluster_imgs(dataset, clusters, save_dir):
    for idx, c in enumerate(clusters):
        with open(os.path.join(save_dir, str(idx) + "_cluster.txt"), "w") as output:
            output.write("Cluster " + str(idx) + '\n')
            for im in c:
                output.write(str(dataset[im][0] + '\n'))


def main(args):
    clusters = clusters_reader(args.clusters)
    dataset = datasets.ImageFolder(args.dataset)
    print("Number of images inside dataset:", len(dataset))
    save_cluster_imgs(dataset.imgs, clusters[0], args.save_dir)
    # im2 = io.imread(dataset.imgs[0][0])
    # plt.imshow(im2)
    # plt.show()


if __name__ == '__main__':
    args = parse_args()
    main(args)
