import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from snntorch import utils
from snntorch import functional as SF

from networks import AlexNet, AlexNetSNN, ResNet



### DOWNLOAD DATA AND CREATE DATALOADERS ###
# dataloader arguments
batch_size = 128
DATA_PATH = './data/'

dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(DATA_PATH, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(DATA_PATH, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


### TRAINING LOOP ###

def train_alexnet():
    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # loss_fn = SF.ce_rate_loss()
    loss_fn = nn.CrossEntropyLoss()
    num_epoch = 1
    train_losses = []

    model.train()
    for epoch in range(num_epoch):
        for batch_idx, batch in enumerate(train_loader):
            data, target = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            
            out = model.forward(data)
            # out_spk = model.forward_pass(model, data, num_steps=2)

            loss = loss_fn(out, target)
            
            # Gradient calculation + weight update
            loss.backward()
            optimizer.step()


            train_losses.append(loss.item())

            # print("Yes")
            if batch_idx % 100 == 0:
                print('\rEpoch: {} {:.0f}%\t     Loss: {:.6f}'.format(epoch, 100. * batch_idx / len(train_loader), loss.item()), end='')


# train_alexnet()



# def forward_pass(net, data, num_steps=50):
#     # mem_rec = []
#     spk_rec = []
#     utils.reset(net)  # resets hidden states for all LIF neurons in net

#     for step in range(num_steps):
#         # utils.reset(net)  # resets hidden states for all LIF neurons in net
#         spk_out = net.forward(data)
#         spk_rec.append(spk_out)
#         # mem_rec.append(mem_out)

#     # return torch.stack(spk_rec), torch.stack(mem_rec)
#     return torch.stack(spk_rec)


def forward_pass(net, num_steps, data):
  mem_rec = []
  spk_rec = []
  utils.reset(net)  # resets hidden states for all LIF neurons in net

  for step in range(num_steps):
      spk_out, mem_out = net(data)
      spk_rec.append(spk_out)
      mem_rec.append(mem_out)

  return torch.stack(spk_rec), torch.stack(mem_rec)


def train_alexnet_snn():
    model = AlexNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = SF.ce_rate_loss()
    # loss_fn = nn.CrossEntropyLoss()
    num_epoch = 1
    train_losses = []

    for epoch in range(num_epoch):
        for batch_idx, batch in enumerate(train_loader):
            data, target = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()

            model.train()            
            # out = model.forward(data)
            out_spk,_ = forward_pass(model, num_steps=2, data=data)

            loss = loss_fn(out_spk, target)
            
            # Gradient calculation + weight update
            loss.backward()
            optimizer.step()


            train_losses.append(loss.item())

            # print("Yes")
            if batch_idx % 100 == 0:
                print('\rEpoch: {} {:.0f}%\t     Loss: {:.6f}'.format(epoch, 100. * batch_idx / len(train_loader), loss.item()), end='')

train_alexnet_snn()