#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour préparer le dataset Doubs selon les recommandations optimales
"""

import pandas as pd
import numpy as np
import os

def prepare_doubs_dataset():
    """
    Prépare le dataset Doubs avec les meilleures variables pour l'application
    
    RECOMMANDATIONS FINALES :
    - Type de régression : Réseau de Neurones (MLP) - Architecture [5, 12, 8, 4, 1]
    - Variable cible : valeur_fonciere
    - Variables explicatives : surface_reelle_bati, nombre_pieces_principales, 
                              surface_terrain, longitude, latitude
    """
    
    print("🏠 PRÉPARATION DU DATASET DOUBS - RECOMMANDATIONS OPTIMALES")
    print("=" * 60)
      # 1. Charger le dataset
    print("📁 Chargement du dataset...")
    df = pd.read_csv('data/Tarif vente doubs.csv', low_memory=False)
    print(f"   • Dataset original : {len(df):,} lignes, {len(df.columns)} colonnes")
    
    # 2. Sélectionner les variables recommandées
    target_col = 'valeur_fonciere'
    feature_cols = [
        'surface_reelle_bati',      # Surface du bâti (m²)
        'nombre_pieces_principales', # Nombre de pièces
        'surface_terrain',          # Surface du terrain (m²)
        'longitude',                # Position géographique X
        'latitude'                  # Position géographique Y
    ]
    
    selected_cols = [target_col] + feature_cols
    
    print("\n🎯 VARIABLES SÉLECTIONNÉES :")
    print(f"   • Variable cible (Y) : {target_col}")
    print(f"   • Variables explicatives (X) :")
    for col in feature_cols:
        print(f"     - {col}")
    
    # 3. Filtrer et nettoyer les données
    print("\n🧹 NETTOYAGE DES DONNÉES :")
    
    # Garder seulement les colonnes sélectionnées
    df_clean = df[selected_cols].copy()
    print(f"   • Après sélection : {len(df_clean)} lignes")
    
    # Convertir toutes les colonnes en numérique (gérer les types mixtes)
    print("   • Conversion en types numériques...")
    for col in selected_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Supprimer les lignes avec des valeurs manquantes dans la cible
    df_clean = df_clean.dropna(subset=[target_col])
    print(f"   • Après suppression NaN cible : {len(df_clean)} lignes")
    
    # Filtrer les prix aberrants (< 1000€ ou > 2M€)
    df_clean = df_clean[
        (df_clean[target_col] >= 1000) & 
        (df_clean[target_col] <= 2000000)
    ]
    print(f"   • Après filtrage prix aberrants : {len(df_clean)} lignes")
    
    # Supprimer les lignes avec trop de valeurs manquantes
    # Garder seulement les lignes avec au moins 3 des 5 variables explicatives
    df_clean = df_clean.dropna(thresh=4)  # au moins 4 valeurs non-nulles (cible + 3 features)
    print(f"   • Après suppression lignes incomplètes : {len(df_clean)} lignes")
    
    # 4. Créer des variables dérivées utiles
    print("\n🔧 CRÉATION DE VARIABLES DÉRIVÉES :")
    
    # Prix au m² (si surface disponible)
    df_clean['prix_par_m2'] = np.where(
        df_clean['surface_reelle_bati'] > 0,
        df_clean[target_col] / df_clean['surface_reelle_bati'],
        np.nan
    )
    
    # Logarithme de la surface terrain (pour normaliser les grandes surfaces)
    df_clean['surface_terrain_log'] = np.where(
        df_clean['surface_terrain'] > 0,
        np.log1p(df_clean['surface_terrain']),
        np.nan
    )
    
    # Type de bien estimé (maison si terrain > 0, sinon appartement)
    df_clean['est_maison'] = (df_clean['surface_terrain'] > 0).astype(int)
    
    print(f"   • Variables ajoutées : prix_par_m2, surface_terrain_log, est_maison")
    
    # 5. Statistiques finales
    print(f"\n📊 STATISTIQUES FINALES :")
    print(f"   • Dataset final : {len(df_clean):,} lignes")
    print(f"   • Réduction : {((len(df) - len(df_clean)) / len(df) * 100):.1f}%")
    print(f"   • Prix moyen : {df_clean[target_col].mean():,.0f}€")
    print(f"   • Prix médian : {df_clean[target_col].median():,.0f}€")
    
    # 6. Créer différentes versions pour l'application
    
    # Version 1 : Variables de base (5 features)
    basic_features = feature_cols
    df_basic = df_clean[[target_col] + basic_features].dropna()
    
    # Version 2 : Variables étendues (8 features avec dérivées)
    extended_features = feature_cols + ['prix_par_m2', 'surface_terrain_log', 'est_maison']
    df_extended = df_clean[[target_col] + extended_features].dropna()
    
    # Version 3 : Échantillon pour tests rapides (1000 lignes)
    df_sample = df_basic.sample(n=min(1000, len(df_basic)), random_state=42)
    
    # 7. Sauvegarder les datasets
    print(f"\n💾 SAUVEGARDE DES DATASETS :")
    
    # Dataset de base
    basic_path = 'data/doubs_basic_dataset.csv'
    df_basic.to_csv(basic_path, index=False)
    print(f"   • Dataset de base : {basic_path} ({len(df_basic):,} lignes)")
    
    # Dataset étendu
    extended_path = 'data/doubs_extended_dataset.csv'
    df_extended.to_csv(extended_path, index=False)
    print(f"   • Dataset étendu : {extended_path} ({len(df_extended):,} lignes)")
    
    # Échantillon de test
    sample_path = 'data/doubs_sample_dataset.csv'
    df_sample.to_csv(sample_path, index=False)
    print(f"   • Échantillon test : {sample_path} ({len(df_sample):,} lignes)")
    
    # 8. Recommandations d'architecture
    print(f"\n🧠 RECOMMANDATIONS D'ARCHITECTURE :")
    print(f"   • Type de régression : Réseau de Neurones (MLP)")
    print(f"   • Architecture de base : [5, 12, 8, 4, 1]")
    print(f"   • Architecture étendue : [8, 16, 12, 6, 1]")
    print(f"   • Learning rate : 0.01")
    print(f"   • Epochs : 500-1000")
    print(f"   • Test size : 20%")
    
    print(f"\n✅ PRÉPARATION TERMINÉE !")
    print(f"   Utilisez '{basic_path}' dans votre application.")
    
    return df_basic, df_extended, df_sample

if __name__ == "__main__":
    try:
        df_basic, df_extended, df_sample = prepare_doubs_dataset()
        
        print(f"\n📋 UTILISATION DANS L'APPLICATION :")
        print(f"1. Charger le fichier : data/doubs_basic_dataset.csv")
        print(f"2. Variable cible (Y) : valeur_fonciere")
        print(f"3. Variables explicatives (X) : toutes les autres")
        print(f"4. Architecture réseau : [5, 12, 8, 4, 1]")
        print(f"5. Type : Régression neuronale")
        
    except Exception as e:
        print(f"❌ Erreur : {e}")
