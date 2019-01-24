"""
data folder
coco-animals/
  train/
    bear/
      COCO_train2014_000000005785.jpg
      COCO_train2014_000000015870.jpg
      [...]
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
  val/
    bear/
    bird/
    cat/
    dog/
    giraffe/
    horse/
    sheep/
    zebra/
"""
import argparse
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T
from torchvision.models.resnet import resnet18, resnet50

from torchvision.datasets import ImageFolder
import umap
import pylab as pl
import glob
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from matplotlib.colors import ListedColormap
import time
import pickle
import gzip


import pandas as pd
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram


import cv2


import umap
import numpy as np
from PIL import Image





# default_dir="/mnt/sdb2/repo/daewon/cocotinydataset/coco-animals"
default_dir="/mnt/sdb2/repo/daewon/deep_fashion_class_20_100_train_val"
# trained_dir = "/mnt/sdb2/repo/daewon/pytorch_pretrained_model/triplet_network"
trained_dir = "/mnt/sdb2/repo/daewon/pytorch_pretrained_model/triplet_network/19_01_23/"
name_pretrained_model = "_triplet_network.pt"


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default=os.path.join(default_dir, 'train'))
parser.add_argument('--val_dir', default=os.path.join(default_dir, 'val'))
parser.add_argument('--batch_size', default=200, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=20000, type=int)
parser.add_argument('--use_gpu', default='use_gpu', action='store_true')
parser.add_argument('--online_hard_triplet_loss', default=True)
parser.add_argument('--save_dir_trained_model', default=trained_dir)
parser.add_argument('--num_per_epoch' , default=400)
# parser.add_argument('--save_dir_trained_model', default=os.path.join(trained_dir, 'triplet_network.pt'))
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

"""
nonlinear_dimension_reduction method(t-sne, umap)

"""


def reduct_dim(feature_vec, components=3, method='umap'):
    if method == 'umap':
        plt.title("UMAP")
        embedding = umap.UMAP(n_components=components).fit_transform(feature_vec)
        # embedding = umap.UMAP(n_components=components).fit_transform(feature_vec, lable_lst_arr)
        return embedding
    elif method == 'tsne':
        plt.title("t-sne")
        model = TSNE(components, learning_rate=100, perplexity=50.0)
        embedding = model.fit_transform(feature_vec)
        return embedding
    elif method == 'pca':
        print("components :", components)
        embedding = PCA(n_components=components).fit_transform(feature_vec)
        return embedding



def ts_pandas(result_dict):
    """
    transform from file_label_list to pandas dataframe

    """
    df = pd.DataFrame(list(result_dict.keys()), columns=['file_name'])
    df['label'] = list(result_dict.values())
    return df

def elewise_str_concat(path, files):

    return [os.path.join(path, file_name) for file_name in files]

def encode_name_lable(root_dir):
    data_dir = root_dir
    lable_range = 0

    dict_file_lable = dict()
    for (path, dir, files) in sorted(os.walk(os.path.join(data_dir))):
        # print(path, dir, files)
        if dir:
            lable_range = len(dir)
            # print("num of lable : ", lable_range)
        else:
            lable_name = path.split("/")[2]
            # print(lable_name)
            # print(path, files)
            dict_file_lable[lable_name] = elewise_str_concat(path, files)

    # swap lable_name for file_name
    dict_file_lable_sw = dict()
    for (key, val) in sorted(dict_file_lable.items()):
        for name in val:
            dict_file_lable_sw[name] = key

    dict_file_lable_sw_temp = dict()
    for (x, y) in dict_file_lable_sw.items():
        lable_num = 0
        for lable_lst in sorted(dict_file_lable.keys()):
            if y == lable_lst:
                dict_file_lable_sw_temp[x] = lable_num
            else:
                lable_num += 1
    label_int_list = list()
    for (i, label_name) in enumerate(sorted(dict_file_lable.keys())):
        # print(i, label_name)
        for j in dict_file_lable_sw.values():
            if j == label_name:
                label_int_list.append(i)
    return dict_file_lable_sw_temp, dict_file_lable, label_int_list, dict_file_lable_sw


def main(args):
    dtype = torch.FloatTensor


    val_transform = T.Compose([
        T.Scale(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_dset = ImageFolder(args.val_dir, transform=val_transform)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    print(val_dset[0][0])


    print("val_dset")
    # print("-"*100)
    # for (i, data) in enumerate(val_loader):
    #     print(i, data)
    print("-"*100)


    print("-"*100)
    val_data = None
    for (i, data) in enumerate(val_loader):
        # print(i, data[0])
        val_data = data[0].numpy()
    print("-"*100)

    print(val_data.shape)

    gray_val_data =val_data.shape
    # print(val_data.shape[0])
    # for i in range(val_data.shape[0]):
    #     print(val_data[i])



    # First load the pretrained ResNet-18 model; this will download the model
    # weights from the web the first time you run it
    model = resnet50(pretrained=True)


    # Reinitialize the last layer of the model.
    # Each pretrained model has a slightly different structure, but from the Resnet class definition
    # we see that the final fully-connected layer is stored in model.fc:
    num_classes = 20
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Cast the model to the correct datatype, and create a loss function fro training the model
    model.type(dtype)

    check_point = torch.load(args.save_dir_trained_model + "19600" + name_pretrained_model)
    # check_point = torch.load(args.save_dir_trained_model + "1200" + name_pretrained_model)
    args.num_epoch1 = check_point['epoch']

    model.load_state_dict(check_point["model_state_dict"])

    loss = check_point["loss"]

    # print(args.num_epoch1)
    # print(loss)
    #
    # for (i, param) in enumerate(model.named_children()):
    #     print(i, param)
    #
    #
    #
    # print("val_dset[0][0].unsqueezeu(0).size() :", val_dset[0][0].unsqueeze(0).size())
    # # adding one dimension to a tensor in pytorch
    # data = val_dset[0][0].unsqueeze(0) # [3, 224, 224] -> [1, 3, 224, 224]
    # print(type(data))

    # embedding = model(data)
    # print("embedding.shape :", embedding.shape) # [1, 20]
    # feature = model.avgpool(data)
    # # print("emb.size(0)", emb.size()) # [1, 3, 218, 218]
    # emb = feature.view(feature.size(0), -1) # [1, 142572]
    # print("emb.shape :", emb.shape)





    # data_dir = "./deep_fashion_class_46" # class-wise folder
    data_dir = "./deep_fashion_class_20_100"  # class-wise folder
    result_dict, result_dict_2, label_int_list, result_dict_3 = encode_name_lable(data_dir)

    """
    result_dict : [file_name, label(integer)]
    result_dict_3 : [file_name, label(class_name)]
    result_dict_2 : swap (key ,value) -> (value, key) [label(integer), file_name]
    label_int_list : [label(integer)]

    """



    df_dict = ts_pandas(result_dict)
    # print(df_dict)
    print("Size of the dataframe :{}".format(df_dict.shape))
    print(df_dict['file_name'].values)
    # df_dict3 = ts_pandas(result_dict_3)
    # print("Size of the dataframe :{}".format(df_dict3.shape))
    # print(df_dict3['label'].values)
    # print(df_dict['label'].values)

    label_group_list = list(df_dict['label'].values)
    lable_lst_arr = np.array(label_group_list)

    num_label_dict = dict()
    for i, l in enumerate(result_dict_2.keys()):
        num_label_dict[i] = l

    file_list = sorted(glob.glob(os.path.join(data_dir, "*.jpg")))


    """
        img_list.pickle : deep_fashion_class_20_100
        img_list_2.pickle : deep_fashion_class_46
        img_list_pandas_46c.pickle : deep_fashion_class_46
    """

    pick_file = "./img_list.pickle"

    img_list = list()
    img_feature = list()
    if not os.path.exists(pick_file):
        # save
        for file in result_dict.keys():
            # print(file)
            data = Image.open(file)
            data = data.resize((224, 224))
            data =  np.asarray(data).transpose(2,0,1)
            # print(data.shape)
            data = data.astype(np.uint8)
            # adding one dimension to a tensor in pytorch
            data = np.expand_dims(data, axis=0)  # [3, 224, 224] -> [1, 3, 224, 224]
            # print(data.shape)
            data = torch.from_numpy(data)
            # print(type(data))
            data = data.type('torch.DoubleTensor')
            # embedding = model(data)
            # print("embedding.shape :", embedding.shape) # [1, 20]
            feature = model.avgpool(data)
            # print("emb.size(0)", emb.size()) # [1, 3, 218, 218]
            emb = feature.view(feature.size(0), -1)  # [1, 142572]
            emb = emb.numpy()
            # print("emb.shape ", emb.shape)
            img_list.append(emb)
        print("---------- save pickle file ----------")
        with gzip.open(pick_file, "wb") as f:
            pickle.dump(img_list, f)

    else:
        # load
        """"
        img_list.pickle : deep_fashion_class_20_100
        img_list_2.pickle : deep_fashion_class_46
        img_list_pandas_46c.pickle : deep_fashion_class_46
        """
        print("---------- load pickle file ----------")
        with gzip.open(pick_file, "rb") as f:
            img_list = pickle.load(f)


    print("len of img_list :", len(img_list))
    print("img_list[0].shape : ", img_list[0])
    np_img_list = np.array(img_list)  # (2000, 1, 128)
    print("np_img_list", np_img_list.shape)
    np_img_reduced = np.squeeze(np_img_list, axis=1)  # (2000, 128)

    print("np_img_reduced.shaped :", np_img_reduced.shape)

    lable_lst = list()
    for i in result_dict.values():
        lable_lst.append(i)

    fig = plt.figure(figsize=(10, 10), dpi=160)
    # fig , ax = plt.subplots()



    start = time.time()

    pca_opt = 0
    method = 'tsne'
    if pca_opt == 1:
        embedding = reduct_dim(np_img_reduced, components=100, method='pca')
        embedding = reduct_dim(embedding, components=2, method=method)
    else:
        embedding = reduct_dim(np_img_reduced, components=2, method=method)

    end = time.time()
    print("embedding shape :", embedding.shape)
    print("elapsed time for embedding: {} seconds".format(end - start))

    print("len(set(label_group_list)) :", len(set(label_group_list)))
    cmap = plt.cm.get_cmap('jet', len(set(label_group_list)))

    for i in range(len(label_group_list)):
        # plt.scatter(embedding[i, 0], embedding[i, 1], c=label_group_list[i], cmap=cmap, s=10)
        plt.scatter(embedding[i, 0], embedding[i, 1], c=label_group_list[i], cmap=cmap, vmin=min(lable_lst),
                    vmax=max(lable_lst), s=10)
        plt.annotate(str(label_group_list[i]), (embedding[i, 0], embedding[i, 1]), fontsize=5)

    # plot legend
    cbar = plt.colorbar(ticks=range(len(set(label_group_list))), label='a label')
    cbar.ax.get_yaxis().set_ticks([])
    # for j, lab in enumerate(range_label):
    for j, lab in enumerate(result_dict_2.keys()):
        cbar.ax.text(.5, (j + 0.5) / len(set(label_group_list)), str(j) + " : " + lab, ha='center', va='center')
    cbar.ax.get_yaxis().labelpad = 15
    # cbar.ax.set_ylabel('label', rotation=270)

    # plt.legend(loc='best')
    plt.show()

    fig.savefig("tsne-unsupervised_20_class.png")





if __name__ == '__main__':
    args = parser.parse_args()
    main(args)






























