import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import random
import torch.nn.functional as F
import torch.optim as optim

device = "cpu"

if torch.cuda.is_available():
    device = "cuda"
print(device)



class SiameseMNIST(Dataset):
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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        label = label.view(-1)
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean(label * torch.pow(euclidean_distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


def get_data_loader(batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = torchvision.datasets.MNIST(root="/data",train = True,transform=transform,download=True)
    test_dataset = torchvision.datasets.MNIST(root="/data",train = False,transform=transform,download = False)

    train_dataset = SiameseMNIST(train_dataset)
    test_dataset = SiameseMNIST(test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
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



MyModel = SiameseNetwork().to(device)
loss_function = ContrastiveLoss()
epoch = 5
lr = 0.001
optimizer = optim.Adam(MyModel.parameters(),lr=lr)


def train_model(model,epochs,optimizer,loss_function,train_dataset):
    model.train()
    train_loss = []
    for epoch in range(epochs):
        total_loss = 0
        for img1,img2,target in train_dataset:
            img1,img2,target = img1.to(device),img2.to(device),target.to(device)
            optimizer.zero_grad()
            output1,output2 = model(img1,img2)


            loss = loss_function(output1,output2,target)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()

        avg_loss = total_loss / len(train_dataloader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs}, loss = {avg_loss:.3f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), train_loss, marker="o", linestyle="-", label="Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()


def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        img1, img2, target = next(iter(test_loader))
        img1, img2 = img1.to(device), img2.to(device)

        out1, out2 = model(img1, img2)
        dist = F.pairwise_distance(out1, out2)

        for i in range(10):
            gercek_durum = "AYNI" if target[i].item() == 1 else "FARKLI"
            print(f"Örnek {i + 1}: Gerçek: {gercek_durum} | Modelin Hesapladığı Mesafe: {dist[i].item():.4f}")




if __name__ == '__main__':
    # Model, loss ve optimizer tanımlamaları burada kalsın
    MyModel = SiameseNetwork().to(device)
    loss_function = ContrastiveLoss()
    optimizer = torch.optim.Adam(MyModel.parameters(), lr=lr)

    # Eğitimi başlat
    train_model(MyModel, 5, optimizer, loss_function, train_dataloader)
    evaluate_model(MyModel,test_dataloader)