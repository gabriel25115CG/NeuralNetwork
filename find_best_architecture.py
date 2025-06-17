"""
🔍 SCRIPT POUR TROUVER AUTOMATIQUEMENT LA MEILLEURE ARCHITECTURE
Analyse votre dataset et recommande l'architecture optimale
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from neural_network.network import NeuralNetwork
import matplotlib.pyplot as plt

def analyze_dataset_complexity(df, target_col):
    """Analyser la complexité du dataset"""
    n_samples, n_features = df.shape
    n_features -= 1  # Exclure la target
    
    # Calculer les corrélations seulement avec les colonnes numériques
    correlations = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col != target_col:
            try:
                corr = abs(df[col].corr(df[target_col]))
                if not np.isnan(corr):
                    correlations.append(corr)
            except:
                continue  # Ignorer les colonnes problématiques
    
    avg_correlation = np.mean(correlations) if correlations else 0
    max_correlation = max(correlations) if correlations else 0
    
    # Classifier la complexité
    if avg_correlation > 0.8:
        complexity = "SIMPLE"
    elif avg_correlation > 0.5:
        complexity = "MOYEN"
    else:
        complexity = "COMPLEXE"
    
    return {
        'n_samples': n_samples,
        'n_features': len(numeric_cols) - 1,  # Nombre de features numériques utilisables
        'avg_correlation': avg_correlation,
        'max_correlation': max_correlation,
        'complexity': complexity
    }

def recommend_architectures(analysis):
    """Recommander des architectures basées sur l'analyse"""
    n_samples = analysis['n_samples']
    n_features = analysis['n_features'] 
    complexity = analysis['complexity']
    
    recommendations = []
    
    # Règle 1: Dataset très petit
    if n_samples < 50:
        recommendations.append({
            'name': 'Linéaire',
            'architecture': [n_features, 1],
            'reason': 'Dataset très petit - éviter overfitting'
        })
        recommendations.append({
            'name': 'Simple',
            'architecture': [n_features, max(2, n_features//2), 1],
            'reason': 'Une couche cachée minimale'
        })
    
    # Règle 2: Dataset petit à moyen
    elif n_samples < 200:
        recommendations.append({
            'name': 'Linéaire',
            'architecture': [n_features, 1],
            'reason': 'Baseline simple'
        })
        recommendations.append({
            'name': 'Optimal',
            'architecture': [n_features, n_features, 1],
            'reason': 'Équilibre complexité/performance'
        })
        recommendations.append({
            'name': 'Avancé',
            'architecture': [n_features, n_features*2, n_features//2, 1],
            'reason': 'Pour relations non-linéaires'
        })
    
    # Règle 3: Dataset plus grand
    elif n_samples < 1000:
        recommendations.append({
            'name': 'Simple',
            'architecture': [n_features, n_features, 1],
            'reason': 'Architecture conservative'
        })
        recommendations.append({
            'name': 'Standard',
            'architecture': [n_features, n_features*2, n_features, 1],
            'reason': 'Architecture équilibrée'
        })
        recommendations.append({
            'name': 'Complexe',
            'architecture': [n_features, n_features*3, n_features*2, n_features, 1],
            'reason': 'Pour données complexes'
        })
    
    # Règle 4: GROS DATASET - Peut supporter des architectures profondes
    else:
        recommendations.append({
            'name': 'Standard',
            'architecture': [n_features, n_features*2, n_features, 1],
            'reason': 'Architecture équilibrée pour gros dataset'
        })
        recommendations.append({
            'name': 'Profond',
            'architecture': [n_features, n_features*4, n_features*3, n_features*2, n_features, 1],
            'reason': 'Réseau profond - assez de données pour éviter overfitting'
        })
        recommendations.append({
            'name': 'Très Profond',
            'architecture': [n_features, n_features*6, n_features*4, n_features*3, n_features*2, n_features, 1],
            'reason': 'Réseau très profond pour patterns complexes'
        })
        recommendations.append({
            'name': 'Wide & Deep',
            'architecture': [n_features, n_features*8, n_features*6, n_features*4, n_features*2, 1],
            'reason': 'Architecture large et profonde pour données riches'
        })
    
    # Ajuster selon la complexité détectée
    if complexity == "SIMPLE" and n_samples < 1000:
        # Favoriser les architectures simples sauf pour gros datasets
        recommendations = [r for r in recommendations if len(r['architecture']) <= 4]
    
    return recommendations

def test_architectures_with_validation(X, y, architectures, test_split=0.3):
    """Tester les architectures avec validation"""
    # Split données
    n_test = int(len(X) * test_split)
    indices = list(range(len(X)))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_idx = indices[n_test:]
    test_idx = indices[:n_test]
    
    X_train = [X[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]
    
    # Normaliser
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = normalize_all_data(X_train, X_test, y_train, y_test)
    
    results = []
    
    for arch_info in architectures:
        arch = arch_info['architecture']
        name = arch_info['name']
        
        try:
            # Entraîner sur train set
            network = NeuralNetwork(arch)
            train_losses = network.train(X_train_norm, y_train_norm, epochs=100, verbose=False)
            
            # Évaluer sur test set
            test_predictions = [network.predict(x)[0] for x in X_test_norm]
            test_mse = np.mean([(y_true - y_pred)**2 for y_true, y_pred in zip(y_test_norm, test_predictions)])
            
            # Calculer overfitting
            train_predictions = [network.predict(x)[0] for x in X_train_norm]
            train_mse = np.mean([(y_true - y_pred)**2 for y_true, y_pred in zip(y_train_norm, train_predictions)])
            
            overfitting = (test_mse - train_mse) / train_mse * 100 if train_mse > 0 else 0
            
            # Amélioration de la loss
            improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100 if train_losses[0] > 0 else 0
            
            results.append({
                'name': name,
                'architecture': arch,
                'train_mse': train_mse,
                'test_mse': test_mse,
                'overfitting': overfitting,
                'improvement': improvement,
                'reason': arch_info['reason'],
                'losses': train_losses
            })
            
        except Exception as e:
            print(f"❌ Erreur avec {name}: {e}")
    
    return results

def normalize_all_data(X_train, X_test, y_train, y_test):
    """Normaliser toutes les données"""
    # Normaliser X
    X_train_norm = []
    X_test_norm = []
    
    for feature_idx in range(len(X_train[0])):
        feature_values = [row[feature_idx] for row in X_train]
        feature_mean = np.mean(feature_values)
        feature_std = np.std(feature_values) + 1e-8
        
        # Appliquer aux données train et test
        for i, row in enumerate(X_train):
            if feature_idx == 0:
                X_train_norm.append([])
            X_train_norm[i].append((row[feature_idx] - feature_mean) / feature_std)
        
        for i, row in enumerate(X_test):
            if feature_idx == 0:
                X_test_norm.append([])
            X_test_norm[i].append((row[feature_idx] - feature_mean) / feature_std)
    
    # Normaliser y
    y_mean = np.mean(y_train)
    y_std = np.std(y_train) + 1e-8
    
    y_train_norm = [(y - y_mean) / y_std for y in y_train]
    y_test_norm = [(y - y_mean) / y_std for y in y_test]
    
    return X_train_norm, X_test_norm, y_train_norm, y_test_norm

def main():
    print("🔍 ANALYSEUR AUTOMATIQUE D'ARCHITECTURE OPTIMALE")
    print("=" * 60)
    
    # Choix du dataset
    print("\n📁 DATASETS DISPONIBLES:")
    print("   1. Dataset complexe (40 lignes, relations linéaires)")
    print("   2. Dataset non-linéaire (50 lignes, relations complexes)")
    print("   3. Dataset simple original (15 lignes)")
    print("   4. Gros dataset immobilier Doubs (nettoyé)")
    
    choice = input("\n🔢 Choisissez un dataset (1-4) ou appuyez sur Entrée pour auto: ").strip()
    
    # Charger le dataset selon le choix
    if choice == "2":
        try:
            df = pd.read_csv('data/immobilier_nonlinear.csv')
            target_col = 'prix'
            print(f"✅ Dataset non-linéaire chargé: {df.shape}")
        except:
            print("❌ Dataset non-linéaire non trouvé, utilisation du complexe")
            df = pd.read_csv('data/dataset_complexe.csv')
            target_col = 'prix'
            print(f"✅ Dataset complexe chargé: {df.shape}")
    elif choice == "3":
        df = pd.read_csv('data/immobilier_simple.csv')
        df = df.rename(columns={'valeur_fonciere': 'prix', 'surface_reelle_bati': 'surface', 'nombre_pieces_principales': 'pieces'})
        target_col = 'prix'
        print(f"✅ Dataset simple chargé: {df.shape}")
    elif choice == "4":
        try:
            print("📁 Chargement du gros dataset immobilier Doubs...")
            # Charger avec gestion des types mixtes
            df = pd.read_csv('data/Tarif vente doubs.csv', low_memory=False)
            print(f"✅ Dataset Doubs chargé: {df.shape}")
            
            # Nettoyer et préparer les données
            print("🧹 Nettoyage des données...")
            
            # Convertir valeur_fonciere en numérique
            df['valeur_fonciere'] = pd.to_numeric(df['valeur_fonciere'], errors='coerce')
            
            # Filtrer les types de biens intéressants
            df_clean = df[df['type_local'].isin(['Maison', 'Appartement'])].copy()
            
            # Convertir les colonnes importantes en numérique
            numeric_cols = ['surface_reelle_bati', 'nombre_pieces_principales', 'surface_terrain', 'longitude', 'latitude']
            for col in numeric_cols:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Supprimer les lignes avec des données manquantes importantes
            df_clean = df_clean.dropna(subset=['valeur_fonciere', 'surface_reelle_bati', 'nombre_pieces_principales'])
            
            # Filtrer les valeurs aberrantes
            df_clean = df_clean[
                (df_clean['valeur_fonciere'] >= 10000) & 
                (df_clean['valeur_fonciere'] <= 2000000) &  # Entre 10k et 2M€
                (df_clean['surface_reelle_bati'] > 0) & 
                (df_clean['surface_reelle_bati'] <= 500) &  # Max 500m²
                (df_clean['nombre_pieces_principales'] >= 1) & 
                (df_clean['nombre_pieces_principales'] <= 15)  # Max 15 pièces
            ]
            
            # Prendre un échantillon si trop volumineux (pour les tests)
            if len(df_clean) > 5000:
                df_clean = df_clean.sample(n=5000, random_state=42)
                print(f"📊 Échantillon de 5000 lignes prélevé")
            
            # Renommer et créer les colonnes finales
            df = df_clean.rename(columns={
                'surface_reelle_bati': 'surface',
                'nombre_pieces_principales': 'pieces',
                'valeur_fonciere': 'prix'
            })
            
            # Ajouter des variables dérivées
            df['prix_par_m2'] = df['prix'] / (df['surface'] + 1)  # +1 pour éviter division par 0
            df['surface_terrain_log'] = np.log1p(df['surface_terrain'].fillna(0))
            
            # Encoder le type de bien
            df['est_maison'] = (df['type_local'] == 'Maison').astype(int)
            
            target_col = 'prix'
            print(f"✅ Dataset Doubs nettoyé: {df.shape[0]} lignes utilisables")
            print(f"   Prix moyen: {df['prix'].mean():,.0f}€")
            print(f"   Surface moyenne: {df['surface'].mean():.1f}m²")
            
        except Exception as e:
            print(f"❌ Erreur avec le dataset Doubs: {e}")
            print("Utilisation du dataset complexe par défaut")
            df = pd.read_csv('data/dataset_complexe.csv')
            target_col = 'prix'
            print(f"✅ Dataset complexe chargé: {df.shape}")
    else:
        # Auto ou choix 1
        try:
            df = pd.read_csv('data/dataset_complexe.csv')
            target_col = 'prix'
            print(f"✅ Dataset complexe chargé: {df.shape}")
        except:
            df = pd.read_csv('data/immobilier_simple.csv')
            df = df.rename(columns={'valeur_fonciere': 'prix', 'surface_reelle_bati': 'surface', 'nombre_pieces_principales': 'pieces'})
            target_col = 'prix'
            print(f"✅ Dataset simple chargé: {df.shape}")
    
    # Analyser la complexité
    analysis = analyze_dataset_complexity(df, target_col)
    
    print(f"\n📊 ANALYSE DU DATASET:")
    print(f"   📈 Échantillons: {analysis['n_samples']}")
    print(f"   📊 Variables: {analysis['n_features']}")
    print(f"   🔗 Corrélation moyenne: {analysis['avg_correlation']:.3f}")
    print(f"   🎯 Complexité: {analysis['complexity']}")
    
    # Recommander des architectures
    recommendations = recommend_architectures(analysis)
    
    print(f"\n🎯 ARCHITECTURES RECOMMANDÉES:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec['name']}: {rec['architecture']}")
        print(f"      └─ {rec['reason']}")
      # Préparer les données
    feature_cols = [col for col in df.columns if col != target_col]
    # Ne garder que les colonnes numériques pour l'entraînement
    numeric_feature_cols = [col for col in feature_cols if df[col].dtype in ['int64', 'float64']]
    
    print(f"📊 Utilisation de {len(numeric_feature_cols)} variables numériques: {numeric_feature_cols}")
    
    X = df[numeric_feature_cols].values.tolist()
    y = df[target_col].values.tolist()
      # Tester les architectures
    print(f"\n🧪 TEST DES ARCHITECTURES AVEC VALIDATION:")
    try:
        results = test_architectures_with_validation(X, y, recommendations)
    except Exception as e:
        print(f"❌ Erreur lors du test des architectures: {e}")
        print("💡 Le dataset est peut-être trop complexe ou les architectures trop grandes.")
        print("📋 Résumé des recommandations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec['name']}: {rec['architecture']}")
        return
    
    # Afficher les résultats
    print(f"\n🏆 RÉSULTATS (avec validation train/test):")
    print("-" * 70)
    
    best_arch = None
    best_score = -1
    
    for result in results:
        # Score composite: amélioration - pénalité overfitting
        overfitting_penalty = max(0, result['overfitting']) * 0.5
        composite_score = result['improvement'] - overfitting_penalty
        
        status = "🟢" if composite_score > 70 else "🟡" if composite_score > 40 else "🔴"
        overfitting_status = "⚠️" if result['overfitting'] > 20 else "✅"
        
        print(f"{status} {result['name']}: {result['architecture']}")
        print(f"      📉 Amélioration: {result['improvement']:.1f}%")
        print(f"      📊 MSE Test: {result['test_mse']:.4f}")
        print(f"      {overfitting_status} Overfitting: {result['overfitting']:+.1f}%")
        print(f"      ⭐ Score composite: {composite_score:.1f}")
        print()
        
        if composite_score > best_score:
            best_score = composite_score
            best_arch = result
    
    print(f"🏆 RECOMMANDATION FINALE:")
    print(f"   Architecture optimale: {best_arch['name']} {best_arch['architecture']}")
    print(f"   Score: {best_score:.1f} points")
    print(f"   Raison: {best_arch['reason']}")
    
    # Graphique comparatif
    if len(results) > 1:
        plt.figure(figsize=(15, 5))
        
        # Graphique 1: Courbes de loss
        plt.subplot(1, 3, 1)
        for result in results:
            plt.plot(result['losses'], label=f"{result['name']}")
        plt.xlabel('Époque')
        plt.ylabel('Loss')
        plt.title('Courbes de Loss d\'Entraînement')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Graphique 2: Comparaison MSE
        plt.subplot(1, 3, 2)
        names = [r['name'] for r in results]
        train_mses = [r['train_mse'] for r in results]
        test_mses = [r['test_mse'] for r in results]
        
        x = np.arange(len(names))
        plt.bar(x - 0.2, train_mses, 0.4, label='Train MSE', alpha=0.7)
        plt.bar(x + 0.2, test_mses, 0.4, label='Test MSE', alpha=0.7)
        plt.xlabel('Architecture')
        plt.ylabel('MSE')
        plt.title('Train vs Test MSE')
        plt.xticks(x, names, rotation=45)
        plt.legend()
        
        # Graphique 3: Score composite
        plt.subplot(1, 3, 3)
        scores = [r['improvement'] - max(0, r['overfitting'])*0.5 for r in results]
        colors = ['green' if s > 70 else 'orange' if s > 40 else 'red' for s in scores]
        plt.bar(names, scores, color=colors, alpha=0.7)
        plt.xlabel('Architecture')
        plt.ylabel('Score Composite')
        plt.title('Performance Globale')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('architecture_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    main()
