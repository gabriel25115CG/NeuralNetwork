"""
🧪 SCRIPT DE TEST POUR COMPARER LES ARCHITECTURES
Exécutez ce script pour voir la différence entre architectures simples et complexes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from neural_network.network import NeuralNetwork
import matplotlib.pyplot as plt

def test_architecture(layers_config, name, X_train, y_train, epochs=100):
    print(f"\n🧠 Test de l'architecture {name}: {layers_config}")
    
    # Créer et entraîner le réseau
    network = NeuralNetwork(layers_config)
    losses = network.train(X_train, y_train, learning_rate=0.01, epochs=epochs, verbose=False)
    
    # Calculer les statistiques
    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
    
    print(f"   📊 Loss initiale: {initial_loss:.6f}")
    print(f"   🎯 Loss finale: {final_loss:.6f}")
    print(f"   📉 Amélioration: {improvement:.2f}%")
    
    return losses, improvement

def main():
    print("🚀 COMPARAISON D'ARCHITECTURES POUR COURBES DE LOSS")
    print("=" * 60)
    
    # Charger le nouveau dataset complexe
    try:
        df = pd.read_csv('data/dataset_complexe.csv')
        print(f"✅ Dataset complexe chargé: {df.shape[0]} lignes, {df.shape[1]} colonnes")
    except:
        print("❌ Utilisez le dataset simple")
        df = pd.read_csv('data/immobilier_simple.csv')
        df = df.rename(columns={'valeur_fonciere': 'prix', 'surface_reelle_bati': 'surface', 'nombre_pieces_principales': 'pieces'})
    
    # Préparer les données
    feature_cols = [col for col in df.columns if col != 'prix']
    X = df[feature_cols].values.tolist()
    y = df['prix'].values.tolist()
    
    # Normalisation simple
    X_norm = []
    for row in X:
        X_norm.append([(val - np.mean([r[i] for r in X])) / (np.std([r[i] for r in X]) + 1e-8) for i, val in enumerate(row)])
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_norm = [(val - y_mean) / y_std for val in y]
    
    n_features = len(feature_cols)
    print(f"📊 Utilisation de {n_features} variables: {feature_cols}")
    
    # Test de différentes architectures
    architectures = [
        ([n_features, 1], "Linéaire simple"),
        ([n_features, 5, 1], "1 couche cachée"),
        ([n_features, 10, 5, 1], "2 couches cachées"),
        ([n_features, 15, 10, 5, 1], "3 couches cachées"),
        ([n_features, 20, 15, 8, 3, 1], "Réseau profond")
    ]
    
    results = []
    
    for arch, name in architectures:
        losses, improvement = test_architecture(arch, name, X_norm, y_norm, epochs=150)
        results.append((name, losses, improvement))
    
    # Créer un graphique de comparaison
    plt.figure(figsize=(12, 8))
    
    for i, (name, losses, improvement) in enumerate(results):
        plt.subplot(2, 3, i+1)
        plt.plot(losses, linewidth=2)
        plt.title(f'{name}\n({improvement:.1f}% amélioration)', fontsize=10)
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_architectures.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n🏆 RÉSULTATS:")
    print("-" * 40)
    for name, _, improvement in results:
        status = "🟢" if improvement > 50 else "🟡" if improvement > 20 else "🔴"
        print(f"{status} {name}: {improvement:.1f}% d'amélioration")
    
    print("\n💡 CONSEIL:")
    best = max(results, key=lambda x: x[2])
    print(f"   Meilleure architecture: {best[0]} ({best[2]:.1f}% d'amélioration)")

if __name__ == "__main__":
    main()
