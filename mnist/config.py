import os
import json
import torch
import data
from model import *
from torch import optim

class Config(object):

    def __init__(self, network, optimizer, loss_fn, log_dir, train_loader=data.train_loader,
                 test_loader=data.test_loader, is_supervised=True):
        super(Config, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = network.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_losses = []
        self.test_losses = []
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.model_file = os.path.join(log_dir, 'model.pth')
        self.optimizer_file = os.path.join(log_dir, 'optimizer.pth')
        self.log_file = os.path.join(log_dir, 'log.json')
        self.n_epoch = 10
        self.log_interval = 10
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.is_supervised = is_supervised

    def load(self):
        network_state_dict = torch.load(self.model_file)
        self.network.load_state_dict(network_state_dict)
        optimizer_state_dict = torch.load(self.optimizer_file)
        self.optimizer.load_state_dict(optimizer_state_dict)
        with open(self.log_file, 'r') as f:
            log = json.load(f)
            self.train_losses = log["train_losses"]
            self.test_losses = log["test_losses"]

    def save(self):
        torch.save(self.network.state_dict(), self.model_file)
        torch.save(self.optimizer.state_dict(), self.optimizer_file)
        with open(self.log_file, 'w') as f:
            json.dump({"train_losses": self.train_losses, "test_losses": self.test_losses}, f)

    def clear(self):
        os.remove(self.model_file)
        os.remove(self.optimizer_file)
        os.remove(self.log_file)

class ConfigManager:

    @staticmethod
    def AutoencoderConfig(hidden_dims, input_dim=784):
        network = Autoencoder(input_dim=input_dim, hidden_dims=hidden_dims)
        log_dir = './log/Autoencoder'
        config = Config(network=network,
                        optimizer=optim.Adam(network.parameters(), weight_decay=1e-4),
                        loss_fn=AutoencoderLossFunc(),
                        log_dir=log_dir,
                        is_supervised=False)
        return config

    @staticmethod
    def PerceptronConfig(input_dim=784, output_dim=10, train_loader=data.train_loader, test_loader=data.test_loader):
        network = Perceptron(input_dim, output_dim)
        log_dir = './log/Perceptron'
        config = Config(network=network,
                        optimizer=optim.Adam(network.parameters()),
                        loss_fn=nn.CrossEntropyLoss(),
                        log_dir=log_dir,
                        train_loader=train_loader,
                        test_loader=test_loader)
        return config

    @staticmethod
    def NetConfig():
        network = Net()
        log_dir = './log/Net'
        config = Config(network=network,
                        optimizer=optim.Adam(network.parameters()),
                        loss_fn=F.nll_loss,
                        log_dir=log_dir)
        return config

    @staticmethod
    def MutiPerceptionConfig(hidden_dims, input_dim=784, output_dim=10, train_loader=data.train_loader,
                             test_loader=data.test_loader):
        network = MutiPreceptron(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        log_dir = './log/MultiPerception'
        config = Config(network=network,
                        optimizer=optim.Adam(network.parameters(), weight_decay=1e-4),
                        loss_fn=nn.CrossEntropyLoss(),
                        log_dir=log_dir,
                        train_loader=train_loader,
                        test_loader=test_loader)
        return config
