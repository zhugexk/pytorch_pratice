import torch
from torch import nn
from data import AudioDataset, Dataloader
import numpy as np
import json


# in: (f, t) out : (f, t)
class IntegralLayer(nn.Module):
    def __init__(self, t):
        super(IntegralLayer, self).__init__()
        self.t = t
        self.integral_matrix = self.__get_integral_matrix()

    def __get_integral_matrix(self):
        matrix = np.ones(self.t)
        matrix = np.tril(matrix)
        # matrix[:, -1:] = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.from_numpy(matrix).float().to(device)

    def forward(self, x):
        x = torch.matmul(x, self.integral_matrix)
        return x


# in: (f, t) out : (f, t) but last t is 0
class DifferenceLayer(nn.Module):
    def __init__(self, t):
        super(DifferenceLayer, self).__init__()
        self.t = t
        self.diff_matrix = self.__get_diff_matrix()

    def __get_diff_matrix(self):
        a = np.diag([-1] * self.t)
        b = np.pad(np.diag([1] * (self.t - 1)), ((1, 0), (0, 1)))
        matrix = (a + b)
        matrix[:, -1:] = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        matrix = torch.from_numpy(matrix).float().to(device)
        return matrix

    def forward(self, x):
        x = torch.matmul(x, self.diff_matrix)
        return x


class ConvolutionalPIDLayer(nn.Module):
    def __init__(self, f, t, in_channels, out_channels, kernel_size):
        super(ConvolutionalPIDLayer, self).__init__()
        self.t = t
        self.f = f
        self.integral_layer = IntegralLayer(t)
        self.diff_layer = DifferenceLayer(t)
        self.conv_present = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_integral = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.conv_diff = nn.Conv2d(in_channels, out_channels, kernel_size)

    def forward(self, x):
        integral = self.integral_layer(x)
        diff = self.diff_layer(x)
        x = self.conv_present(x) + self.conv_integral(integral) + self.conv_diff(diff)
        return x

class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),

            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),

            nn.Flatten(),
        )

    def forward(self, x):
        x = self.block(x)
        return x

class ConvolutionalPIDNetwork(nn.Module):
    def __init__(self, f, t, class_num=8):
        super(ConvolutionalPIDNetwork, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.BatchNorm2d(1),
            ConvolutionalPIDLayer(f, t, 1, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),

            ConvBlock(),
        )
        self.recong = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, class_num),
        )
        self.name = "conv_pid_model"

    def forward(self, x):
        x = self.feature_extract(x)
        logits = self.recong(x)
        return logits


class ConvolutionalNetwork(nn.Module):
    def __init__(self, class_num):
        super(ConvolutionalNetwork, self).__init__()
        self.feature_extract = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(1, 32, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(0.5),

            ConvBlock(),
        )
        self.recong = nn.Sequential(
            nn.Linear(4096, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, class_num),
        )
        self.name = "conv_model"

    def forward(self, x):
        x = self.feature_extract(x)
        logits = self.recong(x)
        return logits

class Linear2D(nn.Module):
    def __init__(self, input_f, output_f):
        super(Linear2D, self).__init__()
        self.w = nn.Parameter(torch.randn(output_f, input_f))
        self.b = nn.Parameter(torch.randn(output_f, 1))

    def forward(self, x):
        x = torch.matmul(self.w, x) + self.b
        return x

class LinearPIDLayer(nn.Module):
    def __init__(self, f, t, output_f):
        super(LinearPIDLayer, self).__init__()
        self.integral_layer = IntegralLayer(t)
        self.diff_layer = DifferenceLayer(t)
        self.linear_present = Linear2D(f, output_f)
        self.linear_integral = Linear2D(f, output_f)
        self.linear_diff = Linear2D(f, output_f)

    def forward(self, x):
        integral = self.integral_layer(x)
        diff = self.diff_layer(x)
        x = self.linear_present(x) + self.linear_integral(integral) + self.linear_diff(diff)
        return x

class LinearNetwork(nn.Module):
    def __init__(self, f=128, t=126, class_num=8):
        super(LinearNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm2d(1),
            Linear2D(f, 16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * t, 256),
            nn.ReLU(),
            nn.Linear(256, class_num),
        )
        self.name = "linear_network"

    def forward(self, x):
        x = self.network(x)
        return x

class LinearPIDNetwork(nn.Module):
    def __init__(self, f=128, t=126, class_num=8):
        super(LinearPIDNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.BatchNorm2d(1),
            LinearPIDLayer(f, t, 16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * t, 256),
            nn.ReLU(),
            nn.Linear(256, class_num),
        )
        self.name = "linear_pid_network"

    def forward(self, x):
        x = self.network(x)
        return x


if __name__ == '__main__':
    pass


