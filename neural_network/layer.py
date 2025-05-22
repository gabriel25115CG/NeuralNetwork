from neural_network.neuron import Neuron

class Layer:
    def __init__(self, n_neurons, n_inputs_per_neuron):
        self.neurons = [Neuron(n_inputs_per_neuron) for _ in range(n_neurons)]

    def forward(self, inputs):
        return [neuron.forward(inputs) for neuron in self.neurons]

    def compute_gradients(self, error_gradients):
        for neuron, grad in zip(self.neurons, error_gradients):
            neuron.compute_gradients(grad)

    def update_params(self, learning_rate):
        for neuron in self.neurons:
            neuron.update_params(learning_rate)
