import torch
from config import Config, ConfigManager
from train import train, s_train
import matplotlib.pyplot as plt
import numpy as np
import data
import os
import json

def test_autoencoder():
    config = ConfigManager.AutoencoderConfig(hidden_dims=[128, 64, 32, 16])
    # train(config)
    config.load()
    from data import test_loader
    examples = enumerate(test_loader)
    idx, (data, target) = next(examples)
    data = data.to(config.device)
    with torch.no_grad():
        predict = config.network(data).view(-1, 1, 28, 28)
    import draw
    draw.draw(predict, data, target)

    code = torch.randn((1, 16)).cuda()
    with torch.no_grad():
        decode = config.network.decoder(code)
        decode = decode.cpu().view(28, 28)
        plt.imshow(decode, cmap='gray')
        plt.show()

def test_net():
    config = ConfigManager.NetConfig()
    train(config)

def test_perceptron():
    config = ConfigManager.PerceptronConfig()
    train(config)

def test_mutiperceptron():
    config = ConfigManager.MutiPerceptionConfig(hidden_dims=[50])
    train(config)

def get_encode(encoder, data_loader, encode_dim):
    encoder_data, encoder_target = [], []
    for idx, (input_x, target) in enumerate(data_loader):
        input_x = input_x.to("cuda")
        with torch.no_grad():
            encode = encoder(input_x)
            encode, target = list(encode.to("cpu").numpy()), list(target.to("cpu").numpy())
            encoder_data.append(encode)
            encoder_target.append(target)
    encoder_data = np.array(encoder_data).reshape((-1, encode_dim))
    encoder_target = np.array(encoder_target).reshape(-1)
    return encoder_data, encoder_target

def get_encode_data_loader(encode_dim):
    encode_file = "./data/MNIST/encode.json"
    try:
        with open(encode_file, "r") as f:
            encode_data = json.load(f)
            train_encoder_input, train_encoder_target = encode_data["train_input"], encode_data["train_target"]
            test_encoder_input, test_encoder_target = encode_data["test_input"], encode_data["test_target"]
    except Exception:
        config = ConfigManager.AutoencoderConfig(hidden_dims=[128, 64, 32, encode_dim])
        # train(config)
        config.load()
        encoder = config.network.encoder
        train_encoder_input, train_encoder_target = get_encode(encoder, config.train_loader, encode_dim)
        test_encoder_input, test_encoder_target = get_encode(encoder, config.test_loader, encode_dim)
        with open(encode_file, "w") as f:
            encode_data = {"train_input": train_encoder_input.astype(np.float32).tolist(),
                           "train_target": train_encoder_target.astype(np.int64).tolist(),
                           "test_input": test_encoder_input.astype(np.float32).tolist(),
                           "test_target": test_encoder_target.astype(np.int64).tolist()}
            json.dump(encode_data, f)

    train_dataset = data.MyDataset(np.array(train_encoder_input).astype(np.float32),
                                   np.array(train_encoder_target).astype(np.int64))
    test_dataset = data.MyDataset(np.array(test_encoder_input).astype(np.float32),
                                  np.array(test_encoder_target).astype(np.int64))

    train_loader = data.get_data_loader(train_dataset)
    test_loader = data.get_data_loader(test_dataset)

    torch.cuda.empty_cache()
    return train_loader, test_loader

def test_autoencoder_perceptron():
    encode_dim = 16
    train_loader, test_loader = get_encode_data_loader(encode_dim)
    config = ConfigManager.PerceptronConfig(input_dim=encode_dim, output_dim=10, train_loader=train_loader,
                                            test_loader=test_loader)
    # train(config)
    s_train(config)

def test_autoencoder_mutiperception():
    encode_dim = 16
    train_loader, test_loader = get_encode_data_loader(encode_dim)
    config = ConfigManager.MutiPerceptionConfig(input_dim=encode_dim, hidden_dims=[20, 20, 20], output_dim=10,
                                                train_loader=train_loader, test_loader=train_loader)
    config.n_epoch = 20
    config.load()
    # train(config)
    s_train(config)

    import draw
    draw.draw(config.train_losses)

    network = config.network

    with torch.no_grad():
        n_correct = 0
        train_loss = 0.
        for idx, (input_x, target) in enumerate(train_loader):
            input_x, target = input_x.to(config.device), target.to(config.device)
            output = network(input_x)
            pred = output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(target.data.view_as(pred)).sum()
            train_loss += config.loss_fn(output, target).item() * train_loader.batch_size
        train_loss /= len(train_loader.dataset)
        print('\ntrain set: Avg. loss: {:.4f}, Accuracy:{}/{} ({:0f}%)\n'.format(
            train_loss, n_correct, len(train_loader.dataset),
            100. * n_correct / len(train_loader.dataset)))

    with torch.no_grad():
        n_correct = 0
        test_loss = 0.
        for idx, (input_x, target) in enumerate(test_loader):
            input_x, target = input_x.to(config.device), target.to(config.device)
            output = network(input_x)
            pred = output.data.max(1, keepdim=True)[1]
            n_correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss += config.loss_fn(output, target).item() * test_loader.batch_size
        test_loss /= len(test_loader.dataset)
        print('\ntest set: Avg. loss: {:.4f}, Accuracy:{}/{} ({:0f}%)\n'.format(
            test_loss, n_correct, len(test_loader.dataset),
            100. * n_correct / len(test_loader.dataset)))


if __name__ == "__main__":
    # test_net()
    # test_perceptron()
    # test_autoencoder()
    # test_mutiperceptron()
    # test_autoencoder_perceptron()
    test_autoencoder_mutiperception()