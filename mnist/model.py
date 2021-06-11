import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x= x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(input=x, dim=1)

class Perceptron(nn.Module):
    def __init__(self, input_dim=784, output_dim=10):
        super(Perceptron, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(self.input_dim, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc1(x)
        return x

class MutiPreceptron(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MutiPreceptron, self).__init__()
        in_dims = [input_dim, ] + hidden_dims
        out_dims = hidden_dims + [output_dim, ]
        self.layer_seq = nn.Sequential(nn.Flatten())
        for idx in range(len(in_dims)):
            layer_name = "fc{}".format(idx)
            relu_name = "relu{}".format(idx)
            self.layer_seq.add_module(layer_name, nn.Linear(in_dims[idx], out_dims[idx]))
            if idx < len(in_dims) - 1:
                self.layer_seq.add_module(relu_name, nn.ReLU())

    def forward(self, x):
        x = self.layer_seq(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super(Autoencoder, self).__init__()
        in_dims = [input_dim, ] + hidden_dims[:-1]
        out_dims = hidden_dims
        self.encoder = nn.Sequential(nn.Flatten())
        for idx in range(len(in_dims)):
            layer_name = "fc{}".format(idx)
            relu_name = "relu{}".format(idx)
            self.encoder.add_module(layer_name, nn.Linear(in_dims[idx], out_dims[idx]))
            if idx < len(in_dims) - 1:
                self.encoder.add_module(relu_name, nn.ReLU())

        self.decoder = nn.Sequential()
        for idx in reversed(range(len(in_dims))):
            layer_name = "fc{}".format(idx)
            relu_name = "relu{}".format(idx)
            self.decoder.add_module(layer_name, nn.Linear(out_dims[idx], in_dims[idx]))
            if idx > 0:
                self.decoder.add_module(relu_name, nn.ReLU())
        self.decoder.add_module("tanh", nn.Tanh())
        return

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class AutoencoderLossFunc(nn.Module):
    def __init__(self):
        super(AutoencoderLossFunc, self).__init__()
        return

    def forward(self, predict, target):
        predict = predict.view(-1, 784)
        target = target.view(-1, 784)
        loss = torch.mean(torch.square(torch.subtract(predict, target)))
        return loss
