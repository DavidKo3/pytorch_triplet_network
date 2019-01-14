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
from torchvision.models.resnet import resnet18
from torchvision.datasets import ImageFolder
import hard_triplet_loss as hd_t_loss


default_dir="/mnt/sdb2/repo/daewon/cocotinydataset/coco-animals"

parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', default=os.path.join(default_dir, 'train'))
parser.add_argument('--val_dir', default=os.path.join(default_dir, 'val'))
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--num_epochs1', default=10, type=int)
parser.add_argument('--num_epochs2', default=500, type=int)
parser.add_argument('--use_gpu', default='use_gpu', action='store_true')
parser.add_argument('--online_hard_triplet_loss', default=True)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main(args):
    dtype = torch.FloatTensor
    if args.use_gpu:
        print("~~~use_gpu~~~")
        dtype = torch.cuda.FloatTensor


    train_transform = T.Compose([
        T.Scale(256),
        T.RandomSizedCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])



    # you can read more about the ImageFolder class here:
    # https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    train_dset = ImageFolder(args.train_dir, transform=train_transform)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    val_transform = T.Compose([
        T.Scale(224),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    val_dset = ImageFolder(args.val_dir, transform=val_transform)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)


    # First load the pretrained ResNet-18 model; this will download the model
    # weights from the web the first time you run it
    model = resnet18(pretrained=True)


    # Reinitialize the last layer of the model.
    # Each pretrained model has a slightly different structure, but from the Resnet class definition
    # we see that the final fully-connected layer is stored in model.fc:
    num_classes = len(train_dset.classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes).cuda()

    # Cast the model to the correct datatype, and create a loss function fro training the model
    model.type(dtype)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = hd_t_loss.HardTripletLoss()
    # First we want to train only the reinitialized last layer for a few epochs.
    # During this phase we do not need to compute gradients with respect to the
    # other weights of the model, so we set the requires_grad flag to False for
    # all model parameters, then set the requires_grad=True for the parameters in the
    # last layer only.
    for param in model.parameters():
        param.requires_grad =  False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Construct an Optimizer object for updating the last layer only.
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-5)

    # Train the entire model for a few more epochs, checking accuracy on the
    # train and validation sets after each epoch.
    for epoch in range(args.num_epochs2):
        print('Starting epoch %d / %d' % (epoch +1, args.num_epochs2))
        run_epoch(model, loss_fn, train_loader, optimizer, dtype)


        if not args.online_hard_triplet_loss :
            train_acc = check_accuracy(model, train_loader, dtype)
            val_acc = check_accuracy(model, val_loader, dtype)
            print("train_Acc", train_acc)
            print("val_acc", val_acc)
            print("\n")





def run_epoch(model, loss_fn, loader, optimizer, dtype):
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









































