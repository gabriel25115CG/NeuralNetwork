import matplotlib.pyplot as plt
import numpy as np

def plot_loss_curve(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Courbe de la perte pendant l'entraînement")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_predictions(model, data, targets, data_min, data_max, targets_min, targets_max):
    x = np.linspace(data_min, data_max, 100)
    x_norm = (x - data_min) / (data_max - data_min)
    y_pred = [model.predict([xi])[0] for xi in x_norm]
    y_pred_denorm = [yi * (targets_max - targets_min) + targets_min for yi in y_pred]

    plt.figure(figsize=(8, 5))
    plt.scatter(data, targets, label="Données réelles")
    plt.plot(x, y_pred_denorm, color='red', label="Prédictions du modèle")
    plt.xlabel("Superficie (m²)")
    plt.ylabel("Prix (en milliers $)")
    plt.title("Prédictions du modèle vs Données")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
