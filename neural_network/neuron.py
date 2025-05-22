import random

class Neuron:
    def __init__(self, n_inputs):
        self.weights = [0.5 for _ in range(n_inputs)]
        self.bias = 0.0
        self.last_inputs = []
        self.last_output = 0.0
        self.grad_weights = [0.0] * n_inputs
        self.grad_bias = 0.0

    def forward(self, inputs):
        self.last_inputs = inputs
        self.last_output = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        return self.last_output

    def compute_gradients(self, error_gradient):
        self.grad_weights = [-error_gradient * x for x in self.last_inputs]
        self.grad_bias = -error_gradient

    def update_params(self, learning_rate):
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, self.grad_weights)]
        self.bias -= learning_rate * self.grad_bias
