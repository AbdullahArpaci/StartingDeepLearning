import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F



class SiameseMNIST(Dataset):
    def __init__(self,mnist_dataset):
        self.mnist_dataset = mnist_dataset

    def __getitem__(self, index):
        print("getitem çağrıldı:", index)
        img1,label1 = self.mnist_dataset[index]

        should_get_same_class = random.randint(0,1)

        if should_get_same_class:
            while True:
                idx = random.randint(0,len(self.mnist_dataset)-1)
                img2,label2 = self.mnist_dataset[idx]

                if label1 == label2:
                    break
            target = torch.tensor([1],dtype=torch.float32)
        else:
            while True:
                idx = random.randint(0,len(self.mnist_dataset)-1)
                img2,label2 = self.mnist_dataset[idx]

                if label1 != label2:
                    break
            target =torch.tensor([0],dtype = torch.float32)


        return img1 ,img2,target

    def __len__(self):
        return len(self.mnist_dataset)


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Basit bir CNN (Feature Extractor)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5), nn.ReLU(), nn.MaxPool2d(2, 2)
        )

        # Fully Connected (Embedding Üretici)
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, 128)  # Çıktı: 128 uzunluğunda bir vektör
        )

    def forward_once(self, x):
        # Bir resim için işlem
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1, input2):
        # İki resmi de aynı fonksiyondan geçir
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

def get_data_loader(batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root="/data",train = True,transform=transform,download=True)
    test_dataset = torchvision.datasets.MNIST(root="/data",train = False,transform=transform,download = False)

    train_dataset = SiameseMNIST(train_dataset)
    test_dataset = SiameseMNIST(test_dataset)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    return train_dataloader,test_dataloader

train_dataloader,test_dataloader = get_data_loader()


def visualize_siamese_samples(loader, n=5):
    # Bir batch veri çekiyoruz: img1, img2, target
    img1, img2, target = next(iter(loader))

    fig, axes = plt.subplots(2, n, figsize=(10, 5))

    for i in range(n):
        # Üst satır: İlk resimler
        axes[0, i].imshow(img1[i].squeeze(), cmap="gray")
        axes[0, i].axis("off")

        # Alt satır: İkinci resimler
        axes[1, i].imshow(img2[i].squeeze(), cmap="gray")
        axes[1, i].axis("off")

        # Başlık: Aynı mı Farklı mı?
        durum = "Aynı" if target[i].item() == 1 else "Farklı"
        axes[0, i].set_title(durum)

    plt.show()

visualize_siamese_samples(train_dataloader,10)