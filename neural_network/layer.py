from neural_network.neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, n_neurons, n_inputs_per_neuron, activation='sigmoid'):
        self.neurons = [Neuron(n_inputs_per_neuron, activation) for _ in range(n_neurons)]
        self.last_inputs = []
        self.last_outputs = []

    def forward(self, inputs):
        """Forward pass pour toute la couche"""
        self.last_inputs = inputs[:]
        self.last_outputs = [neuron.forward(inputs) for neuron in self.neurons]
        return self.last_outputs

    def backward(self, deltas):
        """
        Backward pass pour toute la couche
        deltas : gradients de l'erreur par rapport aux sorties de cette couche
        Retourne : gradients par rapport aux entrées de cette couche
        """
        # S'assurer qu'on a le bon nombre de deltas
        if len(deltas) != len(self.neurons):
            raise ValueError(f"Nombre de deltas ({len(deltas)}) != nombre de neurones ({len(self.neurons)})")
        
        # Calculer les gradients pour chaque neurone
        input_gradients_per_neuron = []
        for neuron, delta in zip(self.neurons, deltas):
            input_grads = neuron.backward(delta)
            input_gradients_per_neuron.append(input_grads)
        
        # Sommer les gradients de tous les neurones pour chaque entrée
        if input_gradients_per_neuron and len(input_gradients_per_neuron[0]) > 0:
            n_inputs = len(input_gradients_per_neuron[0])
            input_gradients = [
                sum(input_gradients_per_neuron[j][i] for j in range(len(self.neurons)))
                for i in range(n_inputs)
            ]
        else:
            input_gradients = []
            
        return input_gradients

    def update_params(self, learning_rate):
        """Mise à jour des paramètres de tous les neurones de la couche"""
        for neuron in self.neurons:
            neuron.update_params(learning_rate)
