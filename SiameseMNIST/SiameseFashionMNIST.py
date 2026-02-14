import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import random

from SiameseMNIST import loss_function, MyModel

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
print(device)

class SiameseFashionMNIST(Dataset):
    def __init__(self,mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        img1,label1 = self.mnist_dataset[index]

        should_get_same_class = random.randint(0,1)

        if should_get_same_class:
            while True:
                idx = random.randint(0,len(self.mnist_dataset)-1)
                img2,label2 = self.mnist_dataset[idx]

                if label1 == label2:
                    break

            target = torch.tensor([1],dtype = torch.float32)
        else:
            while True:
                idx = random.randint(0,len(self.mnist_dataset)-1)
                img2,label2 = self.mnist_dataset[idx]

                if label1 != label2:
                    break
            target = torch.tensor([0],dtype = torch.float32)

        return img1,img2,target
    def __len__(self):
        return len(self.mnist_dataset)


def get_data_loader(batch_size = 64):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = torchvision.datasets.FashionMNIST(root="/data",download=True,transform = transform,train=True)
    test_dataset = torchvision.datasets.FashionMNIST(root="/data",download=True,train = False,transform = transform)


    train_dataset = SiameseFashionMNIST(train_dataset)
    test_dataset = SiameseFashionMNIST(test_dataset)

    train_dataloader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

    return train_dataloader,test_dataloader


def visualize_dataset(dataset,num_samples = 5):

    img1,img2,target = next(iter(dataset))

    fig,axes = plt.subplots(2,num_samples,figsize = (10,5))

    for i in range(num_samples):

        axes[0,i].imshow(img1[i].squeeze(), cmap = "gray")
        axes[0,i].axis("off")

        axes[1,i].imshow(img2[i].squeeze(), cmap = "gray")
        axes[1,i].axis("off")

        if target[i].item() == 1:
            axes[0,i].set_title("Same")
        else:
            axes[0,i].set_title("Different")

    plt.show()

class SiameseNetwork(nn.modules):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.resnet50()
        self.backbone.fc = nn.Identity()

        self.embedding = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.Linear(512,128)
        )

    def forward_once(self,x):
        x = self.backbone(x)
        x = self.embedding(x)
        return x
    def forward(self,output1,output2):
        output1 = self.forward_once(output1)
        output2 = self.forward_once(output2)

        return output1,output2


class ContrastiveLoss(nn.modules):
    def __init__(self):
        pass


if "__main__" == __name__:
    train_dataloader,test_dataloader = get_data_loader()
    visualize_dataset(train_dataloader,10)
    MyModel = SiameseNetwork().to(device)
    optim = optim.Adam(params=MyModel.parameters(),lr=0.001)
    loss_function = ContrastiveLoss()