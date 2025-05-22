from neural_network.layer import Layer

class NeuralNetwork:
    def __init__(self, layers_config):
        """
        layers_config: liste des tailles des couches, ex: [1, 1] = 1 neurone en entrée, 1 en sortie
        """
        self.layers = []
        for i in range(len(layers_config) - 1):
            layer = Layer(layers_config[i + 1], layers_config[i])
            self.layers.append(layer)

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def train(self, data, targets, learning_rate=0.01, epochs=200):
        n_samples = len(data)
        losses = []

        for epoch in range(epochs):
            total_loss = 0
            grad_w_sum = 0
            grad_b_sum = 0

            for x, y in zip(data, targets):
                output = self.predict([x])[0]
                error = y - output
                total_loss += error ** 2 / 2

                grad_w_sum += -error * x
                grad_b_sum += -error

            grad_w_avg = grad_w_sum / n_samples
            grad_b_avg = grad_b_sum / n_samples

            neuron = self.layers[-1].neurons[0]
            neuron.weights[0] -= learning_rate * grad_w_avg
            neuron.bias -= learning_rate * grad_b_avg

            avg_loss = total_loss / n_samples
            losses.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}, Poids: {neuron.weights[0]:.6f}, Biais: {neuron.bias:.6f}")

            if avg_loss > 1e10:
                print("Loss trop grande, arrêt de l'entraînement.")
                break

        return losses
