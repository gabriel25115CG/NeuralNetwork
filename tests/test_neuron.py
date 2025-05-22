import unittest
from neural_network.neuron import Neuron

class TestNeuron(unittest.TestCase):
    def test_forward_output(self):
        neuron = Neuron(1)
        neuron.weights = [0.5]
        neuron.bias = 1.0
        output = neuron.forward([2])
        self.assertAlmostEqual(output, 2 * 0.5 + 1.0)

    def test_gradients_and_update(self):
        neuron = Neuron(1)
        neuron.weights = [0.5]
        neuron.bias = 0
        neuron.forward([2])
        neuron.compute_gradients(error_gradient=-1)  # dE/d_output = -1
        neuron.update_params(learning_rate=0.1)
        self.assertAlmostEqual(neuron.weights[0], 0.5 - 0.1 * (-1 * 2))  # 0.5 + 0.2 = 0.7
        self.assertAlmostEqual(neuron.bias, 0 - 0.1 * (-1))  # 0 + 0.1 = 0.1

if __name__ == '__main__':
    unittest.main()
