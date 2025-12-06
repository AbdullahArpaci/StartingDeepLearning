import torch #pytorch library,tensor operations
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt #Visulation

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(device)

def get_data_loader(batch_size = 64):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,),(0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data",download=True,train = True,transform=transform)
    test_dataset = torchvision.datasets.MNIST(root = "./data",train = False,download=True,transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size= batch_size,shuffle=False)

    return train_loader,test_loader



def visualize_samples(load_data,n):
    images,labels = next(iter(load_data))
    print(images[0].shape)
    fig,axes = plt.subplots(1,n,figsize = (10,5))

    for i in range(n):
        axes[i].imshow(images[i].squeeze(),cmap = "gray")
        axes[i].set_title(f"Label {labels[i].item()}")
        axes[i].axis("off")
    plt.show()


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sequential = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28,128),
        nn.ReLU(),
        nn.Linear(128,64),
        nn.ReLU(),
        nn.Linear(64,10))
    def forward(self,x):
        x = self.Sequential(x)
        return x


def train_model(model,epochs,loss_function,optim,train_dataloader):
    model.train()
    train_loss = []
    train_accuracy = []
    for epoch in range(epochs):
        total_loss = 0
        for images,labels in train_dataloader:
            images,labels = images.to(device),labels.to(device)
            optim.zero_grad()
            output = model(images)

            loss = loss_function(output,labels)
            loss.backward()
            optim.step()

            total_loss = total_loss + loss.item()
        avg_loss = total_loss / len(train_dataloader)
        train_loss.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, loss = {avg_loss:.3f}")

    plt.figure()
    plt.plot(range(1,epochs+1),train_loss,marker = "o" , linestyle = "-",label = "Train Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.show()



def test_model(model,test_dataloader):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for images,labels in test_dataloader:
            images,labels = images.to(device),labels.to(device)

            outputs = model(images)
            _,predict = torch.max(outputs,1)
            total+= labels.size(0)
            correct += (predict == labels).sum().item()

    print(f"Test Accuracy = {correct/total:.3f}%")


if "__main__" == __name__:
    train_dataloader, test_dataloader = get_data_loader()
    visualize_samples(train_dataloader, 10)
    model = MyModel().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, 5, loss_function, optimizer, train_dataloader)
    test_model(model, test_dataloader)
