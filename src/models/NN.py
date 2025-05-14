import torch
import torch.nn as nn

class NN(nn.Module):
    # input: 4096
    # output: 2
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x

    # implement a function that tensor of complex numbers (e.g. (2048,1)) to tensor of real and complex parts (2048,2)
    def complex_to_real(self, x):
        return torch.cat((torch.real(x), torch.imag(x)), dim=1)

    # implement a function that tensor of real numbers (e.g. (2048,2)) to tensor of complex numbers
    def real_to_complex(self, x):
        return x[:, 0] + x[:, 1] * 1j

    # TODO: implement drop out layers
    # TODO: Normalize input data to -[1,1]