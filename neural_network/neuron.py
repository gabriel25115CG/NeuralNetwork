import random
import math

class Neuron:
    def __init__(self, n_inputs, activation='sigmoid'):
        # Initialisation Xavier/Glorot pour de meilleures performances
        if activation == 'sigmoid':
            # Xavier initialization pour sigmoid
            limit = (6.0 / (n_inputs + 1)) ** 0.5
            self.weights = [random.uniform(-limit, limit) for _ in range(n_inputs)]
        else:
            # Initialisation normale pour linear
            self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_inputs)]
        
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
    
    def linear_derivative(self, x):
        """Dérivée de la fonction linéaire : f'(x) = 1"""
        return 1.0
    
    def activation_derivative(self, x):
        """Retourne la dérivée de la fonction d'activation"""
        if self.activation == 'sigmoid':
            return self.sigmoid_derivative(x)
        elif self.activation == 'linear':
            return self.linear_derivative(x)
        else:
            return 1.0  # Par défaut

    def apply_activation(self, x):
        """Applique la fonction d'activation choisie"""
        if self.activation == 'sigmoid':
            return self.sigmoid(x)
        elif self.activation == 'linear':
            return x
        else:
            return x

    def forward(self, inputs):
        """Forward pass : calcule la sortie du neurone"""
        self.last_inputs = inputs[:]  # Copie des entrées
        self.last_activation_input = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.last_output = self.apply_activation(self.last_activation_input)
        return self.last_output

    def backward(self, delta):
        """
        Calcule les gradients pour ce neurone
        delta : gradient de l'erreur par rapport à la sortie de ce neurone
        """
        # Gradient de l'erreur par rapport à l'entrée de l'activation
        activation_grad = self.activation_derivative(self.last_activation_input)
        local_gradient = delta * activation_grad
        
        # Gradients par rapport aux poids et biais
        self.grad_weights = [local_gradient * inp for inp in self.last_inputs]
        self.grad_bias = local_gradient
        
        # Retourner les gradients pour les entrées (pour la couche précédente)
        input_gradients = [local_gradient * w for w in self.weights]
        return input_gradients

    def update_params(self, learning_rate):
        self.weights = [w - learning_rate * gw for w, gw in zip(self.weights, self.grad_weights)]
        self.bias -= learning_rate * self.grad_bias
