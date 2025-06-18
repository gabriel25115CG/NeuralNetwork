#!/usr/bin/env python3
"""
Script pour ajouter une colonne d'ID unique pour les marques de voitures
"""

import pandas as pd
import os
from typing import Dict, Tuple

def add_brand_id_column(csv_file: str) -> Tuple[str, Dict[str, int]]:
    """
    Ajouter une colonne brand_id pour les marques de voitures
    
    Args:
        csv_file: Chemin vers le fichier CSV
    
    Returns:
        Tuple (chemin_fichier_sortie, dictionnaire_mapping)
    """
    
    # Charger le CSV
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"✓ Fichier chargé: {len(df)} lignes, {len(df.columns)} colonnes")
    except Exception as e:
        raise Exception(f"Erreur lors du chargement du fichier: {e}")
    
    # Vérifier que la colonne brand existe
    if 'brand' not in df.columns:
        available_cols = list(df.columns)
        raise Exception(f"Colonne 'brand' non trouvée. Colonnes disponibles: {available_cols}")
    
    # Créer le mapping des marques vers des IDs
    unique_brands = df['brand'].dropna().unique()
    unique_brands = sorted([str(brand) for brand in unique_brands])  # Trier pour cohérence
    
    # Créer le dictionnaire de mapping (commencer à 1)
    brand_to_id = {brand: i + 1 for i, brand in enumerate(unique_brands)}
    
    print(f"\n📊 MAPPING CRÉÉ:")
    print(f"Colonne source: 'brand'")
    print(f"Nouvelle colonne: 'brand_id'")
    print(f"Nombre de marques uniques: {len(unique_brands)}")
    print(f"IDs utilisés: 1 à {len(unique_brands)}")
    
    print(f"\n📋 CORRESPONDANCES:")
    for brand, id_num in list(brand_to_id.items())[:15]:  # Afficher les 15 premiers
        print(f"  {brand} → {id_num}")
    if len(brand_to_id) > 15:
        print(f"  ... et {len(brand_to_id) - 15} autres")
    
    # Appliquer le mapping pour créer la nouvelle colonne
    df['brand_id'] = df['brand'].astype(str).map(brand_to_id)
    
    # Gérer les valeurs manquantes (NaN dans la colonne brand)
    df.loc[df['brand'].isna(), 'brand_id'] = 0
    
    # Définir le fichier de sortie
    base_name = os.path.splitext(csv_file)[0]
    output_file = f"{base_name}_with_brand_id.csv"
    
    # Sauvegarder le fichier modifié
    try:
        df.to_csv(output_file, index=False)
        print(f"\n✓ Fichier sauvegardé: {output_file}")
    except Exception as e:
        raise Exception(f"Erreur lors de la sauvegarde: {e}")
    
    # Afficher un aperçu du résultat
    print(f"\n👀 APERÇU DU RÉSULTAT:")
    preview_cols = ['brand', 'brand_id', 'model', 'price']
    available_preview = [col for col in preview_cols if col in df.columns]
    
    print(df[available_preview].head(15).to_string(index=False))
    
    # Sauvegarder le mapping dans un fichier séparé
    mapping_df = pd.DataFrame([
        {"brand": k, "brand_id": v} 
        for k, v in brand_to_id.items()
    ])
    
    mapping_file = output_file.replace('.csv', '_mapping.csv')
    mapping_df.to_csv(mapping_file, index=False)
    print(f"\n✓ Mapping sauvegardé: {mapping_file}")
    
    return output_file, brand_to_id

def main():
    print("🚗 ENCODAGE DES MARQUES DE VOITURES")
    print("=" * 40)
    
    # Fichier d'entrée
    input_file = "data/trainCar.csv"
    
    try:
        # Vérifier que le fichier existe
        if not os.path.exists(input_file):
            print(f"❌ Erreur: Fichier '{input_file}' non trouvé")
            return
        
        print(f"🔄 Traitement de '{input_file}'...")
        
        # Traiter le fichier
        output_file, mapping = add_brand_id_column(input_file)
        
        print(f"\n🎉 TERMINÉ ! Nouveau fichier créé avec {len(mapping)} marques encodées.")
        print(f"📁 Fichier de sortie: {output_file}")
        print(f"📋 Fichier de mapping: {output_file.replace('.csv', '_mapping.csv')}")
        
    except Exception as e:
        print(f"❌ Erreur: {e}")

if __name__ == "__main__":
    main()
