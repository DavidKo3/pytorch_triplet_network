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
import hard_triplet_loss as hd_t_loss


# default_dir="/mnt/sdb2/repo/daewon/cocotinydataset/coco-animals"
default_dir="/mnt/sdb2/repo/daewon/deep_fashion_class_20_100_train_val"
# trained_dir = "/mnt/sdb2/repo/daewon/pytorch_pretrained_model/triplet_network"
trained_dir = "/mnt/sdb2/repo/daewon/pytorch_pretrained_model/triplet_network/19_01_17/"
name_pretrained_model = "_triplet_network.pt"


parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default=os.path.join(default_dir, 'train'))
parser.add_argument('--val_dir', default=os.path.join(default_dir, 'val'))
parser.add_argument('--batch_size', default=32, type=int)
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


def main(args):
    dtype = torch.FloatTensor


    val_transform = T.Compose([
        T.Scale(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_dset = ImageFolder(args.val_dir, transform=val_transform)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    print(val_dset[0][0])
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

    check_point = torch.load(args.save_dir_trained_model + "0" + name_pretrained_model)
    # check_point = torch.load(args.save_dir_trained_model + "1200" + name_pretrained_model)
    args.num_epoch1 = check_point['epoch']

    model.load_state_dict(check_point["model_state_dict"])

    loss = check_point["loss"]

    print(args.num_epoch1)
    print(loss)

    for (i, param) in enumerate(model.named_children()):
        print(i, param)



    print(val_dset[0][0].unsqueeze(0).size())
    # adding one dimension to a tensor in pytorch
    data = val_dset[0][0].unsqueeze(0) # [3, 224, 224] -> [1, 3, 224, 224]
    embedding = model(data)
    print(embedding)





def run_epoch(model, loss_fn, loader, optimizer, dtype, epoch):
    """
    Train the model for one epoch.
    """
    # Set the model to training mode
    model.train()
    mean_loss=0
    for x, y in loader:
        # The Dataloader produces Torch Tensors, so we need to cast them to the
        # correct datatype and wrap them in Variables.
        #
        # Note that the labels should be a torch.LongTensor on CPU and a
        # torch.cuda.LongTensor on GPU; to accomplish this we first cast to dtype
        # (either torch.FloatTensor or torch.cuda.FloatTensor) and then cast to
        # long; this ensures that y has the correct type in both cases.
        # x_var = Variable(x.type(dtype))
        x_var = Variable(x.cuda())
        y_var = Variable(y.type(dtype).long())

        # Run the model forward to compute scores and loss.
        scores = model(x_var)
        loss = loss_fn(scores, y_var)
        mean_loss += loss
        # Run the model backwrad and take a step using the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    size_batch_per_epoch = (len(loader)/args.batch_size)
    print("mean_loss : ", mean_loss.item()/size_batch_per_epoch)

    if epoch % args.num_per_epoch == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, args.save_dir_trained_model+str(epoch)+name_pretrained_model)




def check_accuracy(model, loader, dtype):
    """
    Check the accuracy of the model
    """
    # Set the model to eval mode
    model.eval()
    num_correct, num_samples = 0, 0

    for x, y in loader:
        # Cast the image data to the correct type and wrap it in a Variable. At
        # test-time when we do not need to compute gradients, marking the Variable
        # as volatile can reduce memory usage and slightly improve speed.
        x_var = Variable(x.cuda(), volatile=True)

        # Run the model forward, and compare the argmasx score with ground-truth category
        scores = model(x_var)
        _, preds = scores.data.cpu().max(1)
        num_correct += (preds == y).sum()
        num_samples += x.size(0)


    # Return the fraction of datapoints that were correctly classfied.
    acc = float(num_correct)/ num_samples
    return acc




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    # print(args.save_dir_trained_model)
    # model = resnet50(pretrained=True)
    # model.eval()
    # model.fc = nn.Linear(model.fc.in_features, 20)
    # optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)
    #
    #
    # check_point = torch.load(args.save_dir_trained_model+"1200"+name_pretrained_model)
    # args.num_epoch1 = check_point['epoch']
    #
    # model.load_state_dict(check_point["model_state_dict"])
    # optimizer.load_state_dict(check_point["optimizer_state_dict"])
    # loss = check_point["loss"]
    #
    # print(args.num_epoch1)
    # print(loss)
    #
    #
    # for (i, param) in enumerate(model.fc.parameters()):
    #     print(i, param)
    # main(args)









































