#!/usr/bin/env python3
"""
Module d'encodage des variables textuelles pour le réseau de neurones
"""

import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any

class TextualDataEncoder:
    """Classe pour encoder/décoder les variables textuelles"""
    
    def __init__(self):
        self.encoders = {}
        self.feature_types = {}
        
    def fit(self, df: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
        """
        Créer les encodages pour les variables textuelles
        
        Args:
            df: DataFrame avec les données
            text_columns: Liste des colonnes textuelles à encoder
            
        Returns:
            Dictionnaire avec les informations d'encodage
        """
        encoding_info = {
            'text_columns': text_columns,
            'encoders': {},
            'feature_mapping': {},
            'new_columns': []
        }
        
        for col in text_columns:
            if col not in df.columns:
                continue
                
            # Nettoyer les données
            clean_data = df[col].dropna().astype(str).str.strip()
            unique_values = sorted(clean_data.unique())
            
            # Décider du type d'encodage
            n_unique = len(unique_values)
            
            if n_unique <= 1:
                # Pas assez de variabilité
                continue
                
            elif n_unique == 2:
                # Encodage binaire (0/1)
                encoder = self._create_binary_encoder(unique_values)
                encoding_type = 'binary'
                new_cols = [f"{col}_encoded"]
                
            elif n_unique <= 10:
                # One-hot encoding pour peu de catégories
                encoder = self._create_onehot_encoder(unique_values)
                encoding_type = 'onehot'
                new_cols = [f"{col}_{self._clean_category_name(val)}" for val in unique_values]
                
            else:
                # Label encoding pour beaucoup de catégories
                encoder = self._create_label_encoder(unique_values)
                encoding_type = 'label'
                new_cols = [f"{col}_encoded"]
            
            # Stocker l'encodeur
            encoding_info['encoders'][col] = {
                'type': encoding_type,
                'encoder': encoder,
                'unique_values': unique_values
            }
            
            encoding_info['feature_mapping'][col] = new_cols
            encoding_info['new_columns'].extend(new_cols)
            
        return encoding_info
    
    def transform(self, df: pd.DataFrame, encoding_info: Dict[str, Any]) -> pd.DataFrame:
        """
        Appliquer l'encodage aux données
        
        Args:
            df: DataFrame à encoder
            encoding_info: Informations d'encodage de fit()
            
        Returns:
            DataFrame avec colonnes encodées ajoutées
        """
        df_encoded = df.copy()
        
        for col, encoder_info in encoding_info['encoders'].items():
            if col not in df.columns:
                continue
                
            # Nettoyer les données
            clean_data = df[col].fillna('Unknown').astype(str).str.strip()
            
            encoder = encoder_info['encoder']
            encoding_type = encoder_info['type']
            unique_values = encoder_info['unique_values']
            
            if encoding_type == 'binary':
                # Encodage binaire
                new_col = f"{col}_encoded"
                df_encoded[new_col] = clean_data.map(encoder).fillna(0).astype(float)
                
            elif encoding_type == 'onehot':
                # One-hot encoding
                for val in unique_values:
                    new_col = f"{col}_{self._clean_category_name(val)}"
                    df_encoded[new_col] = (clean_data == val).astype(float)
                    
            elif encoding_type == 'label':
                # Label encoding
                new_col = f"{col}_encoded"
                df_encoded[new_col] = clean_data.map(encoder).fillna(0).astype(float)
        
        return df_encoded
    
    def get_categorical_options(self, encoding_info: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Obtenir les options disponibles pour chaque variable catégorielle
        
        Returns:
            Dictionnaire {nom_variable: [liste_des_options]}
        """
        options = {}
        
        for col, encoder_info in encoding_info['encoders'].items():
            options[col] = encoder_info['unique_values']
            
        return options
    
    def encode_user_input(self, user_values: Dict[str, str], encoding_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Encoder les valeurs saisies par l'utilisateur
        
        Args:
            user_values: {nom_variable: valeur_choisie}
            encoding_info: Informations d'encodage
            
        Returns:
            Dictionnaire avec les valeurs encodées
        """
        encoded_values = {}
        
        for col, value in user_values.items():
            if col not in encoding_info['encoders']:
                continue
                
            encoder_info = encoding_info['encoders'][col]
            encoder = encoder_info['encoder']
            encoding_type = encoder_info['type']
            unique_values = encoder_info['unique_values']
            
            if encoding_type == 'binary':
                new_col = f"{col}_encoded"
                encoded_values[new_col] = encoder.get(value, 0)
                
            elif encoding_type == 'onehot':
                for val in unique_values:
                    new_col = f"{col}_{self._clean_category_name(val)}"
                    encoded_values[new_col] = 1.0 if val == value else 0.0
                    
            elif encoding_type == 'label':
                new_col = f"{col}_encoded"
                encoded_values[new_col] = encoder.get(value, 0)
        
        return encoded_values
    
    def _create_binary_encoder(self, unique_values: List[str]) -> Dict[str, int]:
        """Créer un encodeur binaire"""
        return {unique_values[0]: 0, unique_values[1]: 1}
    
    def _create_onehot_encoder(self, unique_values: List[str]) -> Dict[str, Dict[str, int]]:
        """Créer un encodeur one-hot"""
        return {val: i for i, val in enumerate(unique_values)}
    
    def _create_label_encoder(self, unique_values: List[str]) -> Dict[str, int]:
        """Créer un encodeur par labels"""
        return {val: i for i, val in enumerate(unique_values)}
    
    def _clean_category_name(self, category: str) -> str:
        """Nettoyer un nom de catégorie pour en faire un nom de colonne valide"""
        import re
        # Remplacer les caractères spéciaux par des underscores
        clean = re.sub(r'[^a-zA-Z0-9]', '_', str(category))
        # Supprimer les underscores multiples
        clean = re.sub(r'_+', '_', clean)
        # Supprimer les underscores en début/fin
        clean = clean.strip('_')
        # Limiter la longueur
        return clean[:20]

def get_recommended_text_columns(df: pd.DataFrame) -> List[str]:
    """
    Recommander les meilleures colonnes textuelles à encoder pour la prédiction immobilière
    
    Args:
        df: DataFrame avec les données
        
    Returns:
        Liste des colonnes recommandées
    """
    # Colonnes potentiellement utiles pour l'immobilier
    priority_columns = [
        'nature_mutation',      # Type de transaction
        'nom_commune',          # Localisation
        'type_local',          # Type de bien
        'nature_culture',      # Type de terrain
        'nature_culture_speciale'  # Spécificité terrain
    ]
    
    recommended = []
    
    for col in priority_columns:
        if col in df.columns:
            # Vérifier qu'il y a assez de variabilité
            unique_count = df[col].nunique()
            total_count = len(df[col].dropna())
            
            if unique_count > 1 and total_count > 100:  # Critères minimum
                recommended.append(col)
    
    return recommended

def detect_text_columns(df: pd.DataFrame, exclude_columns: List[str] = None) -> List[str]:
    """
    Détecter automatiquement les colonnes textuelles dans un DataFrame
    
    Args:
        df: DataFrame à analyser
        exclude_columns: Liste des colonnes à exclure de la détection
        
    Returns:
        Liste des noms de colonnes identifiées comme textuelles
    """
    if exclude_columns is None:
        exclude_columns = []
    
    text_columns = []
    
    for col in df.columns:
        # Exclure les colonnes spécifiées
        if col in exclude_columns:
            continue
            
        # Vérifier si c'est principalement textuel
        try:
            # Essayer de convertir en numérique
            numeric_series = pd.to_numeric(df[col], errors='coerce')
            non_null_count = numeric_series.notna().sum()
            total_count = len(df[col].dropna())
            
            if total_count > 0:
                numeric_ratio = non_null_count / total_count
                # Si moins de 80% sont numériques, considérer comme textuel
                if numeric_ratio < 0.8:
                    # Vérifier qu'il y a assez de variabilité
                    unique_count = df[col].nunique()
                    if unique_count > 1:  # Au moins 2 valeurs différentes
                        text_columns.append(col)
        except:
            # En cas d'erreur, considérer comme textuel si pas numérique
            text_columns.append(col)
    
    return text_columns

# Test de la classe
if __name__ == "__main__":
    print("Test du module d'encodage textuel...")
    
    # Charger les données de test
    df = pd.read_csv('data/Tarif vente doubs.csv', low_memory=False)
    
    # Obtenir les recommandations
    recommended_cols = get_recommended_text_columns(df)
    print(f"Colonnes recommandées: {recommended_cols}")
    
    # Tester l'encodage
    encoder = TextualDataEncoder()
    
    # Prendre un échantillon pour le test
    sample_df = df.head(1000)
    
    # Créer l'encodage
    encoding_info = encoder.fit(sample_df, recommended_cols)
    
    print("\nInformations d'encodage créées:")
    for col, info in encoding_info['encoders'].items():
        print(f"  {col}: {info['type']} encoding ({len(info['unique_values'])} valeurs)")
    
    # Transformer les données
    df_encoded = encoder.transform(sample_df, encoding_info)
    
    print(f"\nNouvelles colonnes créées: {len(encoding_info['new_columns'])}")
    print(f"Colonnes: {encoding_info['new_columns'][:5]}...")  # Afficher les 5 premières
    
    print("\n✅ Test réussi !")
