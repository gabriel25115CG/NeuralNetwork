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

def plot_predictions_vs_actual(model, data, targets, data_min, data_max, targets_min, targets_max):
    """
    Affiche un graphique prédictions vs valeurs réelles pour un réseau de neurones.
    Visualisation adaptée aux méthodes non-linéaires avec zones de confiance.
    """
    # Normaliser les données pour les prédictions
    data_norm = np.array([(np.array(row) - data_min) / (data_max - data_min) for row in data])
    
    # Générer les prédictions
    predictions = []
    for row in data_norm:
        pred = model.predict(row.tolist())
        predictions.append(pred[0] if isinstance(pred, list) else pred)
      # Dénormaliser les prédictions et les cibles
    predictions_denorm = np.array(predictions) * (targets_max - targets_min) + targets_min
    targets_denorm = np.array(targets) * (targets_max - targets_min) + targets_min
    
    # Créer le graphique avec plusieurs sous-graphiques
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Ajouter un jitter (dispersion) sur l'axe des abscisses pour éviter la superposition
    jitter_strength = (np.max(targets_denorm) - np.min(targets_denorm)) * 0.005  # 0.5% de la plage
    targets_jittered = targets_denorm + np.random.normal(0, jitter_strength, len(targets_denorm))
    
    # Graphique 1: Nuage de points avec zones de performance
    ax1.scatter(targets_jittered, predictions_denorm, alpha=0.7, s=60, 
               c=np.abs(targets_denorm - predictions_denorm), cmap='RdYlGn_r', 
               edgecolors='black', linewidth=0.5)
    
    # Ajouter des zones de performance (bandes d'erreur)
    min_val = min(min(targets_denorm), min(predictions_denorm))
    max_val = max(max(targets_denorm), max(predictions_denorm))
    
    # Zone d'erreur acceptable (±10%)
    x_range = np.linspace(min_val, max_val, 100)
    ax1.fill_between(x_range, x_range * 0.9, x_range * 1.1, alpha=0.2, color='green', 
                     label='Zone excellente (±10%)')
    ax1.fill_between(x_range, x_range * 0.8, x_range * 0.9, alpha=0.15, color='orange')
    ax1.fill_between(x_range, x_range * 1.1, x_range * 1.2, alpha=0.15, color='orange', 
                     label='Zone acceptable (±20%)')
    
    # Calculer et afficher les métriques
    mse = np.mean((targets_denorm - predictions_denorm) ** 2)
    mae = np.mean(np.abs(targets_denorm - predictions_denorm))
    ss_res = np.sum((targets_denorm - predictions_denorm) ** 2)
    ss_tot = np.sum((targets_denorm - np.mean(targets_denorm)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    ax1.set_xlabel("Valeurs réelles (Prix en milliers €)", fontsize=12)
    ax1.set_ylabel("Prédictions neuronales (Prix en milliers €)", fontsize=12)
    ax1.set_title(f"Réseau de Neurones: Performance de Prédiction\nR² = {r_squared:.3f} | MAE = {mae:.0f}k€", fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
      # Graphique 2: Distribution des erreurs
    errors = predictions_denorm - targets_denorm
    ax2.hist(errors, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erreur nulle')
    ax2.axvline(x=np.mean(errors), color='orange', linestyle='-', linewidth=2, 
                label=f'Erreur moyenne: {np.mean(errors):.1f}k€')
    
    ax2.set_xlabel("Erreur de prédiction (k€)", fontsize=12)
    ax2.set_ylabel("Fréquence", fontsize=12)
    ax2.set_title("Distribution des Erreurs du Réseau Neuronal", fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return r_squared

def plot_predictions(model, data, targets, data_min, data_max, targets_min, targets_max):
    """
    Affiche un graphique de prédictions pour un réseau de neurones (nuage de points uniquement).
    Pour les réseaux de neurones, on ne trace pas de courbe continue car les relations peuvent être non-linéaires complexes.
    """
    # Normaliser les données pour les prédictions
    data_norm = np.array([(np.array(row) - data_min) / (data_max - data_min) for row in data])
    
    # Générer les prédictions pour chaque point de données
    predictions = []
    for row in data_norm:
        pred = model.predict(row.tolist())
        predictions.append(pred[0] if isinstance(pred, list) else pred)
    
    # Dénormaliser les prédictions
    predictions_denorm = np.array(predictions) * (targets_max - targets_min) + targets_min
    targets_denorm = np.array(targets) * (targets_max - targets_min) + targets_min

    # Créer le graphique avec nuage de points seulement
    data_flat = np.array(data).flatten()
    
    plt.figure(figsize=(8, 5))
    plt.scatter(data_flat, targets_denorm, label="Valeurs réelles", alpha=0.7, color='blue')
    plt.scatter(data_flat, predictions_denorm, label="Prédictions neuronales", alpha=0.7, color='red', marker='x')
    plt.xlabel("Superficie (m²)")
    plt.ylabel("Prix (en milliers €)")
    plt.title("Prédictions du Réseau de Neurones vs Données Réelles")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
