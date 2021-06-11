import torch
from data import Dataset, Dataloader
import json
from model import *

def train_step(dataloader, model, loss_fn, optimizer):
    loss_list = []
    for batch in range(dataloader.steps_per_epoch):
        train_x, label = dataloader.get_data()
        pred_y = model(train_x)
        loss = loss_fn(pred_y, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * dataloader.batch_size
            print(f"loss: {loss:>7f} [{current:>5d}/{dataloader.size:>5d}]")
            loss_list.append(loss)

    return loss_list


def test(dataloader, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for batch in range(dataloader.steps_per_epoch):
            test_x, label = dataloader.get_data()
            pred_y = model(test_x)
            test_loss += loss_fn(pred_y, label).item()
            correct += (pred_y.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= dataloader.steps_per_epoch
    correct /= dataloader.size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def train(dataset, model, epochs=1):
    train_idx_list = np.random.choice(dataset.size, int(dataset.size * 0.8))
    train_data = [dataset.data[idx] for idx in train_idx_list]
    test_data = [dataset.data[idx] for idx in range(dataset.size) if idx not in train_idx_list]
    train_dataloader = Dataloader(train_data, batch_size=64)
    test_dataloader = Dataloader(test_data, batch_size=64)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_loss_list = []
    test_loss_list = []
    test_correct_list = []

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss_list += train_step(train_dataloader, model, loss_fn, optimizer)
        test_loss, correct = test(test_dataloader, model, loss_fn)
        test_loss_list.append(test_loss)
        test_correct_list.append(correct)
    print("Done!")

    torch.save(model.state_dict(), "cnn_model.pth")
    print("Saved PyTorch Model State to cnn_model.pth")

    with open("data/" + model.name + ".json", 'w') as f:
        json.dump({"train_loss": train_loss_list, "test_loss": test_loss_list, "test_correct": test_correct_list}, f)


if __name__ == '__main__':
    dataset = AudioDataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    # model = ConvolutionalNetwork(class_num=len(dataset.label2idx)).to(device)
    # model = ConvolutionalPIDNetwork(128, 126, 8).to(device)
    # model = LinearNetwork(128, 126, 8).to(device)
    # train(dataset, model, 100)
    model = LinearPIDNetwork(128, 126, 8).to(device)
    train(dataset, model, 100)