import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

input_features = np.array([[random.uniform(0, 5000)], [random.uniform(0, 5000)]])
price_previous = input_features[0, 0]
price_current = input_features[1, 0]
log_return = np.log(price_current / price_previous)
calculate_class = 1 if log_return > 0 else 0


class Neuron:
    def __init__(self):
        self.bias = 0.0
        self.weights = np.random.randn(1, 2) * np.sqrt(1 / 2)  # Xavier Initialization


class OutputNeuron:
    def __init__(self):
        self.bias = 0.0
        self.weights = np.random.randn(1, 3) * np.sqrt(1 / 3)  # Xavier Initialization


# Hidden layer 1
n1 = Neuron()
n2 = Neuron()
n3 = Neuron()

# Output layer
o1 = OutputNeuron()

weights_history = []
log_returns_history = []
weighted_sum_history = []


def forward_propagation():
    hidden_scalar_1 = np.dot(n1.weights, input_features) + n1.bias
    hidden_scalar_2 = np.dot(n2.weights, input_features) + n2.bias
    hidden_scalar_3 = np.dot(n3.weights, input_features) + n3.bias

    activation_1 = np.tanh(hidden_scalar_1)
    activation_2 = np.tanh(hidden_scalar_2)
    activation_3 = np.tanh(hidden_scalar_3)

    activation_vec = np.array([[activation_1[0, 0]], [activation_2[0, 0]], [activation_3[0, 0]]])
    z_o = np.dot(o1.weights, activation_vec) + o1.bias
    output1 = np.tanh(z_o)
    final_probability = (output1 + 1) / 2

    weighted_sum_history.append(z_o[0, 0])
    return final_probability, activation_vec, (activation_1, activation_2, activation_3), z_o, output1


# Cross-Entropy
def loss_function(true_label, predicted_probability):
    epsilon = 1e-15  # Avoid log(0) errors
    predicted_probability = np.clip(predicted_probability, epsilon, 1 - epsilon)
    return -(true_label * np.log(predicted_probability) + (1 - true_label) * np.log(1 - predicted_probability))


def backpropagation(true_label, learning_rate):
    p, activation_vec, (a1, a2, a3), z_o, output1 = forward_propagation()
    loss = loss_function(true_label, p)

    dl_dp = - (true_label / p - (1 - true_label) / (1 - p))
    dp_dz = 0.5 * (1 - output1 ** 2)

    delta_output = dl_dp * dp_dz
    grad_o_weights = delta_output * activation_vec.T
    grad_o_bias = delta_output

    activations = np.array([a1, a2, a3])
    delta_hidden = delta_output * o1.weights[0, :].reshape(3, 1) * (1 - activations.reshape(3, 1) ** 2)

    grad_n1 = delta_hidden[0, 0] * input_features.T
    grad_n2 = delta_hidden[1, 0] * input_features.T
    grad_n3 = delta_hidden[2, 0] * input_features.T

    grad_n1_bias = delta_hidden[0, 0]
    grad_n2_bias = delta_hidden[1, 0]
    grad_n3_bias = delta_hidden[2, 0]

    o1.weights -= learning_rate * grad_o_weights
    o1.bias -= learning_rate * grad_o_bias

    n1.weights -= learning_rate * grad_n1
    n1.bias -= learning_rate * grad_n1_bias

    n2.weights -= learning_rate * grad_n2
    n2.bias -= learning_rate * grad_n2_bias

    n3.weights -= learning_rate * grad_n3
    n3.bias -= learning_rate * grad_n3_bias

    weights_history.append(o1.weights.flatten().copy())

    return loss


epochs = 1000
validation_cycles = 0

while epochs > 0:
    price_previous = random.uniform(0, 5000)
    price_current = random.uniform(0, 5000)
    log_return = np.log((price_current + 1e-10) / (price_previous + 1e-10))
    log_returns_history.append(log_return)

    input_features = np.array([[price_previous], [price_current]]) / 5000.0
    calculate_class = 1 if log_return > 0 else 0
    backpropagation(calculate_class, 0.01)
    epochs -= 1

smoothed_log_returns = gaussian_filter1d(log_returns_history, sigma=50)
smoothed_weights = np.array(weights_history)
smoothed_weighted_sum = gaussian_filter1d(weighted_sum_history, sigma=50)

plt.figure(figsize=(12, 6))
plt.plot(smoothed_weights, linestyle="dashed", label=["Weight 1", "Weight 2", "Weight 3"])
plt.plot(smoothed_log_returns, label="Smoothed Log Return", color="black", linewidth=2)
plt.plot(smoothed_weighted_sum, label="Smoothed Weighted Sum", color="red", linewidth=2, linestyle="dotted")
plt.xlabel("Training Iterations")
plt.ylabel("Value")
plt.title("Smoothed Model Weights vs. Log Returns Over Training")
plt.legend()
plt.grid()
plt.show()
