import matplotlib.pyplot as plt
import torch

def plot_data(data, target):
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(target[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()

def draw(predict, data, target):
    for i in range(6):
        plt.subplot(2, 6, i+1)
        plt.tight_layout()
        plt.imshow(torch.Tensor.cpu(data[i][0]), cmap='gray', interpolation='none')
        plt.title(target[i].item())
        plt.xticks([])
        plt.yticks([])

        plt.subplot(2, 6, i+7)
        plt.tight_layout()
        plt.imshow(torch.Tensor.cpu(predict[i][0]), cmap='gray', interpolation='none')
        plt.title(target[i].item())
        plt.xticks([])
        plt.yticks([])
    plt.show()

def draw(y):
    plt.plot(y)
    plt.show()

if __name__ == '__main__':
    import data
    examples = enumerate(data.test_loader)
    batch_idx, (example_data, examples_targets) = next(examples)
    plot_data(example_data, examples_targets)