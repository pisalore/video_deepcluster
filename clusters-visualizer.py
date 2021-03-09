import argparse
import pickle
import os
import random
import torchvision.datasets as datasets
import matplotlib.pyplot as plt


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


def save_cluster_images(cluster,  cl_name):
    if len(cluster) == 9:
        img_dim = 3
        cols, rows = img_dim, img_dim
        fig = plt.figure(figsize=(8, 8))
        for i in range(cols * rows):
            img = plt.imread(cluster[i])
            fig.add_subplot(rows, cols, i + 1)
            plt.imshow(img)
        plt.savefig(cl_name)


def save_clusters(dataset, clusters, save_dir):
    for idx, c in enumerate(clusters):
        with open(os.path.join(save_dir, str(idx) + "_cluster.txt"), "w") as output:
            output.write("Cluster " + str(idx) + '\n')
            imgs_list = []
            for im in c:
                output.write(str(dataset[im][0] + '\n'))
                imgs_list.append(str(dataset[im][0]))
            if len(imgs_list) >= 9:
                save_cluster_images(random.sample(imgs_list, 9), os.path.join(save_dir, 'cluster_' + str(idx) + ".png"))


def main(args):
    clusters = clusters_reader(args.clusters)
    dataset = datasets.ImageFolder(args.dataset)
    print("Number of images inside dataset:", len(dataset))
    save_clusters(dataset.imgs, clusters[-1], args.save_dir)


if __name__ == '__main__':
    args = parse_args()
    main(args)
