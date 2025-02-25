import numpy as np
import random

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

def forward_propagation():
     hidden_scalar_1 =  np.dot(n1.weights, input_features) + n1.bias
     hidden_scalar_2 = np.dot(n2.weights, input_features) + n2.bias
     hidden_scalar_3 = np.dot(n3.weights, input_features) + n3.bias

     activation_1 = np.tanh(hidden_scalar_1)
     activation_2 = np.tanh(hidden_scalar_2)
     activation_3 = np.tanh(hidden_scalar_3)

     activation_vec = np.array([[activation_1[0, 0]], [activation_2[0, 0]], [activation_3[0, 0]]])
     z_o = np.dot(o1.weights, activation_vec) + o1.bias
     output1 = np.tanh(z_o)
     final_probability = (output1 + 1) / 2
     return final_probability, activation_vec, (activation_1, activation_2, activation_3), z_o, output1

# Cross-Entropy
def loss_function(true_label, predicted_probability):
    epsilon = 1e-15  # Avoid log(0) errors
    predicted_probability = np.clip(predicted_probability, epsilon, 1 - epsilon)
    return -(true_label * np.log(predicted_probability) + (1 - true_label) * np.log(1 - predicted_probability))


def backpropagation(true_label, learning_rate):
    # Forward pass
    p, activation_vec, (a1, a2, a3), z_o, output1 = forward_propagation()
    # Loss
    loss = loss_function(true_label, p)
    print("Loss value: ", loss)
    # Backward Pass
    # Derivative of the of loss with predicted value p
    dl_dp = - (true_label / p - (1 - true_label) / (1 - p))
    # Derivative probability score with respect to the inputs of the output layer
    dp_dz = 0.5 * (1 - output1 ** 2)

    delta_output = dl_dp * dp_dz
    # This error term tells us how much the output neuron's net input should change to reduce the loss. It is the key signal used to update the output neuron's weights and bias.
    grad_o_weights = delta_output * activation_vec.T  # Shape (1, 3)
    grad_o_bias = delta_output
    # These gradients indicate how each parameter (weight and bias) of the output neuron contributes to the error.
    activations = np.array([a1, a2, a3])
    delta_hidden = delta_output * o1.weights[0, :].reshape(3, 1) * (1 - activations.reshape(3, 1) ** 2)

    grad_n1 = delta_hidden[0, 0] * input_features.T  # For neuron 1, shape (1, 2)
    grad_n2 = delta_hidden[1, 0] * input_features.T  # For neuron 2
    grad_n3 = delta_hidden[2, 0] * input_features.T  # For neuron 3

    grad_n1_bias = delta_hidden[0, 0]
    grad_n2_bias = delta_hidden[1, 0]
    grad_n3_bias = delta_hidden[2, 0]

    print(f"Loss: {loss}")
    print(f"True Label: {true_label}, Predicted Probability: {p}")
    print(f"Gradients - Output Weights: {grad_o_weights}, Bias: {grad_o_bias}")
    print(f"Gradients - Hidden Layer Δh1: {delta_hidden[0, 0]}, Δh2: {delta_hidden[1, 0]}, Δh3: {delta_hidden[2, 0]}")
    print(f"Weights Before Update - o1: {o1.weights}, n1: {n1.weights}, n2: {n2.weights}, n3: {n3.weights}")

    # Update the output neuron:
    o1.weights = o1.weights - learning_rate * grad_o_weights
    o1.bias = o1.bias - learning_rate * grad_o_bias

    # Update each hidden neuron:
    n1.weights = n1.weights - learning_rate * grad_n1
    n1.bias = n1.bias - learning_rate * grad_n1_bias

    n2.weights = n2.weights - learning_rate * grad_n2
    n2.bias = n2.bias - learning_rate * grad_n2_bias

    n3.weights = n3.weights - learning_rate * grad_n3
    n3.bias = n3.bias - learning_rate * grad_n3_bias
    print(f"Weights After Update - o1: {o1.weights}, n1: {n1.weights}, n2: {n2.weights}, n3: {n3.weights}")
    return loss

epochs = 6000
validation_cycles = 1000


while epochs > 0:
    input_features = np.array([[random.uniform(0, 5000)], [random.uniform(0, 5000)]])
    price_previous = input_features[0, 0]
    price_current = input_features[1, 0]
    log_return = np.log((price_current + 1e-10) / (price_previous + 1e-10))
    input_features = input_features / 5000.0
    calculate_class = 1 if log_return > 0 else 0
    backpropagation(calculate_class, 0.01)
    epochs -= 1

validation_count = {"accurate": 0, "miss": 0}  # Fixed typo in "miss"

def compute_accuracy(true_classification, model_classification):
    if true_classification == model_classification:
        validation_count["accurate"] += 1
    else:
        validation_count["miss"] += 1  # Ensure it refers to "miss"

# Validation
while validation_cycles > 0:
    input_features = np.array([[random.uniform(0, 5000)], [random.uniform(0, 5000)]])
    input_features = input_features / 5000.0
    price_previous = input_features[0, 0]
    price_current = input_features[1, 0]
    log_return = np.log((price_current + 1e-6) / (price_previous + 1e-6))
    calculate_class = "positive" if log_return > 0 else "negative"

    # Unpack only the final probability from forward_propagation
    validation_probability, _, _, _, _ = forward_propagation()
    model_prediction = "positive" if validation_probability[0, 0] > 0.5 else "negative"

    compute_accuracy(calculate_class, model_prediction)
    print(f"Log return is: {(log_return * 100):.0f}%, Model prediction is {model_prediction}")
    print("Calculated class:", calculate_class,
          "Model classification:", model_prediction)  # Fixed print statement

    validation_cycles -= 1

n = validation_count["accurate"] + validation_count["miss"]
mean_accuracy = validation_count["accurate"] / n if n > 0 else 0  # Avoid division by zero

print("Mean accuracy:", mean_accuracy)
