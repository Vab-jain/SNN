import torch
import torch.nn as nn

import snntorch as snn
from snntorch import surrogate
from snntorch import utils



class AlexNetSNN(nn.Module):   
    '''
    AlexNetSNN: 
    '''
    def __init__(self, num=10):
        super(AlexNetSNN, self).__init__()

        self.spike_grad = surrogate.fast_sigmoid(slope=25)
        self.beta = 0.5
        self.num_steps = 50

        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            nn.ReLU(inplace=True), 
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(32*12*12,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.Linear(1024,num),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
         
        )
    
    def forward(self, x):
        x_feature = self.feature(x)
        x_feature_2 = x_feature.view(-1,32*12*12)
        x_out = self.classifier(x_feature_2)
        return x_out
    
    @staticmethod
    def forward_pass(self, data, num_steps=50):
        # mem_rec = []
        spk_rec = []
        utils.reset(self)  # resets hidden states for all LIF neurons in net

        for step in range(num_steps):
            spk_out = self.forward(data)
            spk_rec.append(spk_out)
            # mem_rec.append(mem_out)

        # return torch.stack(spk_rec), torch.stack(mem_rec)
        return torch.stack(spk_rec)


class AlexNet(nn.Module):   
    '''
    AlexNet: 
    '''
    def __init__(self, num=10):
        super(AlexNet, self).__init__()

        self.spike_grad = surrogate.fast_sigmoid(slope=25)
        self.beta = 0.5
        self.num_steps = 50

        self.feature = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1),
            # snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
            nn.ReLU(inplace=True), 
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),   
            nn.MaxPool2d( kernel_size=2, stride=2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d( kernel_size=2, stride=1),
            snn.Leaky(beta=self.beta, spike_grad=self.spike_grad, init_hidden=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32*12*12,2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,num),
        )
    
    def forward(self, x):
        x_feature = self.feature(x)
        x_feature_2 = x_feature.view(-1,32*12*12)
        return x_feature_2
        # x_out = self.classifier(x_feature_2)
        # return x_out
        
    




class ResNet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)