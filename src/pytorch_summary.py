import torch
from torchvision import models
from torchsummary import summary
from torchvision.models.resnet import resnet18, resnet50


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

resnet50 = models.resnet50().to(device)

summary(resnet50, (3, 224, 224))