import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True' # pyplot error 해결코드 -> 라이브러리 충돌문제!!

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from matplotlib import pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



def train(model, device, train_loader, optimizer, epochs):
    model.train()
    i=0
    for epoch in range(epochs):
        for data, target in train_loader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if i%1000 == 0:
                print('Train Step : {}\tLoss: {:.3f}'.format(i, loss.item()))
            i += 1

def eval(model, device, test_loader):
    model.eval()
    correct = 0
    for data, target in test_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        prediction = output.data.max(1)[1]
        correct += prediction.eq(target.data).sum()

    print('Test set: Accuraccy : {:.2f}%'.format(100 * correct / len(test_loader.dataset)))



if __name__=="__main__":
    # -----------------------------------------------
    isCuda = torch.cuda.is_available()
    device = torch.device('cuda' if isCuda else 'cpu')

    print(f'Current cuda device is {device}')
    # -----------------------------------------------
    # Hyperparameter
    batch_size = 50
    epochs = 15
    lr = 0.0001

    # -----------------------------------------------
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    print("Number of Training Data : ", len(train_data))
    print("Number of Test Data : ", len(test_data))

    print("train_data sample : \n", train_data[0])
    image, label = train_data[0]
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title('label : %s' %label)
    plt.savefig('train_data_sampleFig.png')
    plt.show()

    # -----------------------------------------------------

    train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(dataset = test_data, batch_size = batch_size, shuffle = True)

    model = CNN().to(device)

    # optimizer = optim.Adamax(model.parameters(), lr = lr) Accuracy = 98.23%
    optimizer = optim.Adam(model.parameters(), lr = lr) # Accuracy = 99.02%
    criterion = nn.CrossEntropyLoss()
    print("my model : \n", model)

    train(model, device, train_loader, optimizer, epochs)
    eval(model, device, test_loader)
    torch.save(model, './mnistModel.pth')