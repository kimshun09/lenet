import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5, 1, 0)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool(x)
        x = f.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 16*5*5)
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_cifar10(batch=128):
    train_loader = DataLoader(
        datasets.CIFAR10('./data',
                         train=True,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5],
                             )
                         ])),
        batch_size=batch,
        shuffle=True
    )
    test_loader = DataLoader(
        datasets.CIFAR10('./data',
                         train=False,
                         download=True,
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(
                                 [0.5, 0.5, 0.5],
                                 [0.5, 0.5, 0.5],
                             )
                         ])),
        batch_size=batch,
        shuffle=True
    )
    return {'train':train_loader, 'test':test_loader}

def main():
    epoch = 500
    history = {'train_loss':[], 'test_loss':[], 'train_acc':[], 'test_acc':[]}
    loader = load_cifar10()

    net = CNN()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=net.parameters(), lr=0.001, momentum=0.9)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    net.to(device)
    print(device)

    writer = SummaryWriter(log_dir='./logs')

    for e in range(epoch):
        net.train()
        loss = None
        for i, (images, labels) in enumerate(loader['train']):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = net(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            if i%10 == 0:
                print(f'Training log: {e+1:03} epoch ({(i+1)*128:05} / 50000 train. data). Loss: {loss.item()}')

        history['train_loss'].append(loss.item())
        net.eval()

        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader['train']):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
        acc = float(correct / 50000)
        history['train_acc'].append(acc)
        print(f'Accuracy on train. data: {acc}')

        loss_test = None
        correct = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader['test']):
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
                loss_test = criterion(outputs, labels)
        acc_test = float(correct / 10000)
        history['test_acc'].append(acc_test)
        history['test_loss'].append(loss_test.item())
        print(f'Accuracy on test data: {acc_test}')
        print(f'Loss on test: {loss_test.item()}')

        writer.add_scalars('Loss', {'train':loss.item(), 'test':loss_test.item()}, e)
        writer.add_scalars('Accuracy', {'train':acc, 'test':acc_test}, e)

    print(history)
    writer.close()


if __name__ == '__main__':
    main()


