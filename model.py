import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, drop_p =0.15):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 5,stride=3) #o/p size 74
        self.pool = nn.MaxPool2d(2,2) #o/p size 37
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2) #o/p size 18
        self.pool2 = nn.MaxPool2d(2,2) # o/p size = 9
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1) # o/p size = 7
        self.conv4 = nn.Conv2d(128,256, 3, stride=1) #o/psize = 5
        self.pool3 = nn.MaxPool2d(2,2) # o/p size 2
        self.dropout = nn.Dropout(p = drop_p)
        #self.dropout2 = nn.Dropout(p = 0.2)
        #self.dropout3 = nn.Dropout(p = 0.25)
        #self.dropout4 = nn.Dropout(p = 0.3)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512, 136)
        #self.fc3 = nn.Linear(400,136)
        #self.fc4 = nn.Linear(300,136)
        #self.fc5 = nn.Linear(150,136)

    def forward(self, x):

        x = self.pool(F.elu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool2(F.elu(self.conv2(x)))
        x = self.dropout(x)
        x = F.elu(self.conv3(x))
        x = self.dropout(x)
        x = self.pool3(F.elu(self.conv4(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.elu(self.fc1(x))
        x = self.dropout(x)
        x = F.elu(self.fc2(x))
        x = self.dropout(x)
        #x = F.elu(self.fc3(x))
        #x = self.dropout(x)
        #x = F.tanh(self.fc4(x))
        #x = self.dropout(x)
        #x = F.relu(self.fc5(x))
        #x = self.dropout(x)
        #x = F.log_softmax(x, dim =1)
        return x
