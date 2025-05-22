from neural_network.network import NeuralNetwork
from data.dataset import data, targets, data_min, data_max, targets_min, targets_max
from utils.visualization import plot_loss_curve, plot_predictions

import numpy as np

def denormalize(value, min_val, max_val):
    return value * (max_val - min_val) + min_val

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def interactive_prediction(model):
    while True:
        try:
            superficie_input = input("Entrez une superficie (en m²) pour prédire le prix (ou 'q' pour quitter) : ")
            if superficie_input.lower() == 'q':
                print("Fin de la session de prédiction.")
                break
            superficie = float(superficie_input)
            superficie_norm = normalize(superficie, data_min, data_max)
            prediction_norm = model.predict([superficie_norm])[0]
            prediction_real = denormalize(prediction_norm, targets_min, targets_max)
            print(f"Superficie: {superficie} m² => Prix prédit: {prediction_real:.2f} milliers $")
        except ValueError:
            print("Entrée invalide. Veuillez entrer un nombre réel ou 'q' pour quitter.")

if __name__ == "__main__":
    # Création et entraînement du réseau
    nn = NeuralNetwork([1, 1])
    losses = nn.train(data, targets, learning_rate=0.01, epochs=200)

    # Visualisations après entraînement
    plot_loss_curve(losses)
    plot_predictions(nn, data, targets, data_min, data_max, targets_min, targets_max)

    # Prédictions sur valeurs fixes
    test_inputs = np.array([120, 180, 220], dtype=float)
    test_inputs_norm = (test_inputs - data_min) / (data_max - data_min)

    for x_norm, x_raw in zip(test_inputs_norm, test_inputs):
        pred_norm = nn.predict([x_norm])[0]
        pred_real = denormalize(pred_norm, targets_min, targets_max)
        print(f"Superficie: {x_raw} m² => Prix prédit: {pred_real:.2f} milliers $")

    # Prédiction utilisateur
    interactive_prediction(nn)
