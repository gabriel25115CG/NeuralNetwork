#!/usr/bin/env python3
"""
Script de test pour vérifier les prédictions du réseau de neurones
"""
import pandas as pd
import sys
import os

# Ajouter le chemin vers neural_network
sys.path.append(os.path.dirname(__file__))
from neural_network.model_persistence import ModelPersistence

def test_predictions():
    """Tester les prédictions avec les modèles sauvegardés"""
    
    print("🔍 Test des prédictions des modèles sauvegardés")
    print("=" * 60)
      # Charger le gestionnaire de persistence
    persistence = ModelPersistence(models_directory="../saved_models")
    models = persistence.list_saved_models()
    
    if not models:
        print("❌ Aucun modèle trouvé")
        return
    
    for model in models:
        print(f"\n📊 Modèle: {model['model_info']['name']}")
        print(f"   Variable cible: {model['model_info'].get('target_column', 'Non définie')}")
        print(f"   Variables d'entrée: {model['model_info'].get('feature_columns', [])}")
        
        # Charger le modèle complet
        loaded_model = persistence.load_model(model['filename'])
        if loaded_model.get('success'):
            model_info = loaded_model['model_info']
            neural_network = loaded_model['neural_network']
            normalization_params = loaded_model.get('normalization_params', {})
            
            # Tester avec deux exemples
            print(f"   🏠 Test 1: Maison 80m², 3 pièces")
            test_prediction(model_info, neural_network, normalization_params, 
                          surface=80, pieces=3, name="Petite maison")
            
            print(f"   🏠 Test 2: Maison 200m², 6 pièces") 
            test_prediction(model_info, neural_network, normalization_params,
                          surface=200, pieces=6, name="Grande maison")
        else:
            print(f"   ❌ Erreur de chargement: {loaded_model.get('error')}")

def test_prediction(model_info, neural_network, normalization_params, surface, pieces, name):
    """Tester une prédiction avec des valeurs données"""
    
    feature_columns = model_info.get('feature_columns', [])
    target_column = model_info.get('target_column', '')
    
    # Créer les valeurs d'entrée dans l'ordre des feature_columns
    input_values = []
    
    # Valeurs par défaut basées sur les données d'exemple
    defaults = {
        'valeur_fonciere': 150000,  # Ne devrait pas être utilisé si c'est la cible
        'surface_reelle_bati': surface,
        'nombre_pieces_principales': pieces,
        'surface_terrain': 800,  # Valeur moyenne
        'longitude': 6.3,
        'latitude': 47.0,
        # Pour les modèles plus complexes
        'numero_disposition': 1,
        'code_postal': 25000,
        'code_commune': 25000,
        'code_departement': 25,
        'nombre_lots': 1
    }
    
    for feature_name in feature_columns:
        if feature_name in defaults:
            input_values.append(defaults[feature_name])
        else:
            input_values.append(0)  # Valeur par défaut
    
    # Normaliser
    normalized_inputs = input_values.copy()
    if normalization_params:
        feature_mins = normalization_params.get('feature_mins')
        feature_maxs = normalization_params.get('feature_maxs')
        
        if feature_mins and feature_maxs:
            for i, value in enumerate(input_values):
                if i < len(feature_mins) and i < len(feature_maxs):
                    if feature_maxs[i] != feature_mins[i]:
                        normalized_inputs[i] = (value - feature_mins[i]) / (feature_maxs[i] - feature_mins[i])
                    else:
                        normalized_inputs[i] = 0.0
    
    # Prédiction
    prediction = neural_network.predict(normalized_inputs)
    predicted_value = prediction[0] if isinstance(prediction, list) else prediction
    
    # Dénormaliser
    if normalization_params:
        target_min = normalization_params.get('target_min')
        target_max = normalization_params.get('target_max')
        
        if target_min is not None and target_max is not None:
            predicted_value = predicted_value * (target_max - target_min) + target_min
    
    print(f"     {name}: {predicted_value:.0f} (cible: {target_column})")

if __name__ == "__main__":
    test_predictions()
