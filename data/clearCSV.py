import pandas as pd

# Charger le fichier CSV
file_path = "fichier_nettoye.csv"  # Remplacez par le chemin de votre fichier
df = pd.read_csv(file_path)

# Supprimer les lignes avec des champs vides dans une colonne spécifique
colonne_cible = "surface_reelle_bati"  # Remplacez par le nom de la colonne
df = df.dropna(subset=[colonne_cible])

# Sauvegarder le fichier nettoyé
df.to_csv("fichier_nettoye.csv", index=False)