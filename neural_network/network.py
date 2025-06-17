from neural_network.layer import Layer
import random

class NeuralNetwork:
    def __init__(self, layers_config, activations=None):
        """
        layers_config: list of layer sizes, e.g. [2, 5, 1] means 2 inputs, 5 hidden neurons, 1 output
        activations: list of activation functions for each layer (except input)
        """
        self.layers_config = layers_config
        self.layers = []
        
        # Activations par défaut : sigmoid pour les couches cachées, linear pour la sortie
        if activations is None:
            activations = ['sigmoid'] * (len(layers_config) - 2) + ['linear']
        
        for i in range(len(layers_config) - 1):
            activation = activations[i] if i < len(activations) else 'sigmoid'
            layer = Layer(layers_config[i + 1], layers_config[i], activation)
            self.layers.append(layer)

    def predict(self, inputs):
        """
        inputs: list of floats
        returns: list of outputs
        """
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs)
        return current_inputs

    def train(self, X_data, y_data, learning_rate=0.01, epochs=200, verbose=True, progress_callback=None):
        """
        X_data: list of input lists [[x1, x2], [x3, x4], ...]
        y_data: list of target values [y1, y2, ...]
        progress_callback: fonction appelée à chaque époque avec (epoch, total_epochs, loss)
        """
        n_samples = len(X_data)
        losses = []

        for epoch in range(epochs):
            total_loss = 0.0

            for X, y in zip(X_data, y_data):
                # Forward pass
                output = self.predict(X)[0]  # Assuming single output
                
                # Calculate loss
                error = y - output
                total_loss += error ** 2 / 2# FIXME: Implémentation simplifiée - Backpropagation à implémenter
                # Pour l'instant, on met à jour seulement la couche de sortie de manière approximative
                
                # Simple gradient descent pour la couche de sortie
                output_layer = self.layers[-1]  # Dernière couche (sortie)
                output_neuron = output_layer.neurons[0]  # Premier neurone de sortie
                
                # Obtenir les entrées de la couche de sortie (sorties de la couche précédente)
                if len(self.layers) == 1:
                    # Pas de couches cachées, les entrées sont X directement
                    layer_inputs = X
                else:
                    # Il y a des couches cachées, calculer les sorties de la couche précédente
                    layer_inputs = X
                    for layer in self.layers[:-1]:  # Toutes les couches sauf la dernière
                        layer_inputs = layer.forward(layer_inputs)
                
                # Mise à jour des poids de la couche de sortie
                for i, input_val in enumerate(layer_inputs):
                    output_neuron.weights[i] -= learning_rate * (-error * input_val)
                output_neuron.bias -= learning_rate * (-error)
                
                # TODO: Implémenter la vraie backpropagation pour toutes les couches
                # Pour l'instant, on fait une mise à jour approximative des couches cachées
                if len(self.layers) > 1:
                    # Mise à jour très simplifiée des couches cachées (pas optimal)
                    for layer_idx in range(len(self.layers) - 1):
                        layer = self.layers[layer_idx]
                        layer_inputs_current = X if layer_idx == 0 else layer.forward(X)
                        
                        for neuron in layer.neurons:
                            for i, input_val in enumerate(neuron.weights):
                                if i < len(X):
                                    neuron.weights[i] -= learning_rate * 0.001 * error * X[i]  # Très petit ajustement
                            neuron.bias -= learning_rate * 0.001 * error

            avg_loss = total_loss / n_samples
            losses.append(avg_loss)

            # Appeler le callback de progression si fourni
            if progress_callback:
                progress_callback(epoch + 1, epochs, avg_loss)

            if verbose and ((epoch + 1) % 20 == 0 or epoch == 0):
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

            # Early stopping si la loss devient trop grande
            if avg_loss > 1e10:
                if verbose:
                    print("Loss trop grande, arrêt de l'entraînement.")
                break

        return losses

    def get_info(self):
        """Retourne des informations sur l'architecture du réseau"""
        return {
            'layers_config': self.layers_config,
            'total_params': sum(len(layer.neurons) * (len(layer.neurons[0].weights) + 1) for layer in self.layers),
            'n_layers': len(self.layers)
        }
