import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy


device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print("Device:",device)


def get_data_loader(batch_size = 64):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5))]
    )
    train_dataset = torchvision.datasets.CIFAR10(root = "./data",train = True,download=True,transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root = "./data",train = False,download=True,transform=transform)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    return train_dataloader,test_dataloader


def imshow(img, ax):
    img = img / 2 + 0.5                  # denormalize
    np_img = img.numpy()
    np_img = np.transpose(np_img, (1, 2, 0))  # C,H,W → H,W,C
    ax.imshow(np_img)
    ax.axis("off")

def get_samples_images(data):
    data_iter = iter(data)
    images, labels = next(data_iter)
    return images, labels

def visualize(data, n):
    images, labels = get_samples_images(data)
    fig, axes = plt.subplots(1, n, figsize=(12, 6))

    for i in range(n):
        imshow(images[i], axes[i])
        axes[i].set_title(f"{labels[i].item()}")

    plt.tight_layout()
    plt.show()

class MyCnnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=256*4*4,out_features=128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )
    def forward(self,x):
        x = self.sequential(x)
        return x

def model_train(model, epochs, optimizer, loss_function, train_dataloader):
    model.train()
    train_loss = []
    train_accuracy = []

    for epoch in range(epochs):
        total_loss = 0
        total_correct = 0
        total_samples = 0

        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # forward
            outputs = model(images)

            # loss
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # prediction
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        avg_loss = total_loss / len(train_dataloader)
        avg_accuracy = total_correct / total_samples

        train_loss.append(avg_loss)
        train_accuracy.append(avg_accuracy)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Accuracy: {avg_accuracy:.4f}")

    fig,(ax1,ax2) = plt.subplots(1,2,figsize = (10,5))

    ax1.plot(range(1,epochs+1),train_accuracy,marker = "o",linestyle = "-")
    ax1.set_title("Accuracy Score")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")

    ax2.plot(range(1, epochs + 1), train_loss,marker = "o",linestyle = "-")
    ax2.set_title("Loss Score")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    plt.tight_layout()
    plt.show()


def test_model(model,loss_function,test_dataloader):
    model.eval()
    total_accuracy = 0
    total_loss = 0
    total = 0
    with torch.no_grad():
        for images,labels in test_dataloader:
            images,labels = images.to(device),labels.to(device)

            outputs = model(images)
            loss = loss_function(outputs,labels)

            total_loss += loss.item()

            _,predict = torch.max(outputs,1)
            total+= labels.size(0)
            total_accuracy += (predict == labels).sum().item()
    print(f"Test Accuracy = {total_accuracy / total:.3f}%")
    print(f"Test Loss = {total_loss / len(test_dataloader):.3f}%")



if "__main__" == __name__  :
    train_dataloader, test_dataloader = get_data_loader(64)

    visualize(train_dataloader, 10)

    model = MyCnnModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    model_train(model, 10, optimizer, loss_function, train_dataloader)
    test_model(model, loss_function, test_dataloader)