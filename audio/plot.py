import json
import matplotlib.pyplot as plt

def get_data(file_name, data_name):
    data = []
    with open(file_name, 'r') as f:
        data = json.load(f)
    if data_name is None:
        return data
    else:
        return data[data_name]


CONV_MODEL = "data/conv_model.json"
CONV_PID_MODEL = "data/conv_pid_model.json"
LINEAR_NETWORK = "data/linear_network.json"
LINEAR_PID_NETWORK = "data/linear_pid_network.json"

linear_train_loss = get_data(LINEAR_NETWORK, "train_loss")
linear_pid_train_loss = get_data(LINEAR_PID_NETWORK, "train_loss")
conv_train_loss = get_data(CONV_MODEL, "train_loss")
conv_pid_train_loss = get_data(CONV_PID_MODEL, "train_loss")
# plt.plot(linear_train_loss, label="linear_train_loss")
# plt.plot(linear_pid_train_loss, label="linear_pid_train_loss")
# plt.plot(conv_train_loss, label="conv_train_loss")
# plt.plot(conv_pid_train_loss, label="conv_pid_train_loss")
linear_test_accuracy = get_data(LINEAR_NETWORK, "test_correct")
linear_pid_test_accuracy = get_data(LINEAR_PID_NETWORK, "test_correct")
conv_test_accuracy = get_data(CONV_MODEL, "test_correct")
conv_pid_test_accuracy = get_data(CONV_PID_MODEL, "test_correct")
# plt.plot(linear_test_accuracy, label="linear_test_accuracy")
# plt.plot(linear_pid_test_accuracy, label="linear_pid_test_accuracy")
plt.plot(conv_test_accuracy, label="conv_test_accuracy")
plt.plot(conv_pid_test_accuracy, label="conv_pid_test_accuracy")
plt.xlabel("batch/10")
plt.ylabel("accuracy")
plt.legend()
plt.show()