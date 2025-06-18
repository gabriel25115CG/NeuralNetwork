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
        Forward pass complet à travers le réseau
        inputs: list of floats
        returns: list of outputs
        """
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs)
        return current_inputs
    
    def backward(self, target, output):
        """
        Backward pass complet (vraie backpropagation)
        target: valeur cible (y) - peut être un scalaire ou une liste
        output: sortie prédite du réseau - liste des sorties
        """
        # Convertir target en liste si c'est un scalaire
        if not isinstance(target, list):
            target = [target]
        
        # S'assurer que target et output ont la même taille
        if len(target) != len(output):
            if len(target) == 1:
                target = target * len(output)
            else:
                raise ValueError("Incompatibilité entre target et output")
        
        # Étape 1: Calculer l'erreur de la couche de sortie
        # Pour MSE: ∂E/∂output = -(target - output)
        output_deltas = [-(t - o) for t, o in zip(target, output)]
        
        # Étape 2: Propager l'erreur en arrière à travers toutes les couches
        current_deltas = output_deltas
        
        # Parcourir les couches en sens inverse
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            
            # Calculer les gradients pour cette couche et obtenir les deltas pour la couche précédente
            if i > 0:  # Pas la première couche
                input_deltas = layer.backward(current_deltas)
                current_deltas = input_deltas
            else:  # Première couche - pas besoin de propager plus loin
                layer.backward(current_deltas)

    def train(self, X_data, y_data, learning_rate=0.01, epochs=200, verbose=True, progress_callback=None):
        """
        Entraînement avec vraie backpropagation
        X_data: list of input lists [[x1, x2], [x3, x4], ...]
        y_data: list of target values [y1, y2, ...]
        progress_callback: fonction appelée à chaque époque avec (epoch, total_epochs, loss)
        """
        n_samples = len(X_data)
        losses = []

        for epoch in range(epochs):
            total_loss = 0.0

            for X, y in zip(X_data, y_data):
                # ÉTAPE 1: Forward pass
                output = self.predict(X)
                predicted = output[0] if len(output) == 1 else output  # Support multi-output
                
                # ÉTAPE 2: Calcul de l'erreur (MSE)
                if isinstance(predicted, list):
                    error = sum((y - pred) ** 2 for pred in output) / len(output)
                else:
                    error = (y - predicted) ** 2
                total_loss += error / 2  # Facteur 1/2 comme dans la théorie
                
                # ÉTAPE 3: Backward pass (vraie backpropagation)
                self.backward(y, output)
                
                # ÉTAPE 4: Mise à jour des paramètres
                for layer in self.layers:
                    layer.update_params(learning_rate)

            # Calculer la loss moyenne sur tous les échantillons (comme dans la théorie)
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
