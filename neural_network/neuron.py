import random
import math

class Neuron:
    def __init__(self, n_inputs, activation='sigmoid'):
        self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
        self.bias = 0.0
        self.activation = activation
        self.last_inputs = []
        self.last_output = 0.0
        self.last_activation_input = 0.0
        self.grad_weights = [0.0] * n_inputs
        self.grad_bias = 0.0

    def sigmoid(self, x):
        """Fonction d'activation sigmoïde : f(x) = 1 / (1 + e^(-x))"""
        try:
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 0.0 if x < 0 else 1.0

    def sigmoid_derivative(self, x):
        """Dérivée de la sigmoïde : f'(x) = f(x) * (1 - f(x))"""
        sig = self.sigmoid(x)
        return sig * (1 - sig)

    def apply_activation(self, x):
        """Applique la fonction d'activation choisie"""
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'linear':
            return x
        else:
            return x

    def forward(self, inputs):
        self.last_inputs = inputs
        self.last_activation_input = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.last_output = self.apply_activation(self.last_activation_input)
        return self.last_output

    def compute_gradients(self, error_gradient):
        self.grad_weights = [-error_gradient * x for x in self.last_inputs]
        self.grad_bias = -error_gradient

    def update_params(self, learning_rate):
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, self.grad_weights)]
        self.bias -= learning_rate * self.grad_bias
