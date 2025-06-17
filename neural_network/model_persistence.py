import json
import pickle
import os
from datetime import datetime
import numpy as np
from .network import NeuralNetwork

class ModelPersistence:
    """Classe pour gérer la sauvegarde et le chargement des modèles"""
    
    def __init__(self, models_directory="saved_models"):
        """
        Initialiser le gestionnaire de persistence
          Args:
            models_directory (str): Répertoire où sauvegarder les modèles
        """
        self.models_directory = models_directory
        self.ensure_directory_exists()
        
    def ensure_directory_exists(self):
        """Créer le répertoire de sauvegarde s'il n'existe pas"""
        if not os.path.exists(self.models_directory):
            os.makedirs(self.models_directory)
            
    def save_model(self, model_info, neural_network, normalization_params=None, training_results=None):
        """
        Sauvegarder un modèle complet avec ses métadonnées et résultats d'entraînement
        
        Args:
            model_info (dict): Informations sur le modèle
            neural_network (NeuralNetwork): Le réseau de neurones entraîné
            normalization_params (dict): Paramètres de normalisation
            training_results (dict): Résultats d'entraînement pour les graphiques
                - loss_history: historique des losses
                - y_test_real: vraies valeurs de test (dénormalisées)
                - y_test_pred: prédictions de test (dénormalisées)  
                - metrics: métriques calculées (MSE, MAE, R2)
                - training_data_stats: statistiques des données
            
        Returns:
            str: Chemin du fichier sauvegardé
        """
        # Générer un nom de fichier unique
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = model_info.get('name', 'model').replace(' ', '_')
        filename = f"{model_name}_{timestamp}"
        
        # Chemins des fichiers
        metadata_path = os.path.join(self.models_directory, f"{filename}_metadata.json")
        network_path = os.path.join(self.models_directory, f"{filename}_network.pkl")
          # Préparer les métadonnées
        # Utiliser l'architecture mise à jour depuis model_info si disponible
        if 'network_architecture' in model_info and model_info['network_architecture']:
            layers_config_to_save = model_info['network_architecture']
        else:
            layers_config_to_save = neural_network.layers_config
            
        metadata = {
            "model_info": {
                "name": model_info.get('name'),
                "target_column": model_info.get('target_column'),
                "feature_columns": model_info.get('feature_columns'),
                "data_shape": model_info.get('data_shape'),
                "data_file": model_info.get('data_file'),
                "accuracy": model_info.get('accuracy'),
                "created_date": model_info.get('created_date', datetime.now().strftime("%d/%m/%Y à %H:%M")),
                "model_type": "Neural Network"
            },
            "network_architecture": {
                "layers_config": layers_config_to_save,
                "layers_sizes": [len(layer.neurons) for layer in neural_network.layers],
                "total_layers": len(neural_network.layers),
                "activation_function": "sigmoid"
            },
            "normalization_params": normalization_params or {},
            "training_results": training_results or {},
            "file_info": {
                "metadata_file": os.path.basename(metadata_path),
                "network_file": os.path.basename(network_path),
                "saved_at": datetime.now().isoformat()
            }
        }
        
        try:
            # Sauvegarder les métadonnées en JSON
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            # Sauvegarder le réseau de neurones en pickle
            with open(network_path, 'wb') as f:
                pickle.dump(neural_network, f)
            
            return {
                "success": True,
                "metadata_path": metadata_path,
                "network_path": network_path,
                "filename": filename
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_model(self, filename):
        """
        Charger un modèle sauvegardé
        
        Args:
            filename (str): Nom du fichier (sans extension)
            
        Returns:
            dict: Contient le model_info, neural_network et normalization_params
        """
        metadata_path = os.path.join(self.models_directory, f"{filename}_metadata.json")
        network_path = os.path.join(self.models_directory, f"{filename}_network.pkl")
        
        try:
            # Charger les métadonnées
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
              # Charger le réseau de neurones
            with open(network_path, 'rb') as f:
                neural_network = pickle.load(f)
            
            return {
                "success": True,
                "model_info": metadata["model_info"],
                "neural_network": neural_network,
                "normalization_params": metadata.get("normalization_params", {}),
                "network_architecture": metadata.get("network_architecture", {}),
                "training_results": metadata.get("training_results", {}),
                "file_info": metadata.get("file_info", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_saved_models(self):
        """
        Lister tous les modèles sauvegardés
        
        Returns:
            list: Liste des modèles avec leurs métadonnées
        """
        models = []
        
        if not os.path.exists(self.models_directory):
            return models
        
        try:
            # Chercher tous les fichiers de métadonnées
            for filename in os.listdir(self.models_directory):
                if filename.endswith("_metadata.json"):
                    metadata_path = os.path.join(self.models_directory, filename)
                    
                    try:
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        
                        # Extraire le nom de base du fichier
                        base_filename = filename.replace("_metadata.json", "")
                        
                        # Vérifier que le fichier réseau existe aussi
                        network_path = os.path.join(self.models_directory, f"{base_filename}_network.pkl")
                        if os.path.exists(network_path):
                            models.append({
                                "filename": base_filename,
                                "model_info": metadata["model_info"],
                                "network_architecture": metadata.get("network_architecture", {}),
                                "file_info": metadata.get("file_info", {}),
                                "file_size": self._get_file_sizes(base_filename)
                            })
                    
                    except Exception as e:
                        print(f"Erreur lors du chargement de {filename}: {e}")
                        continue
            
            # Trier par date de création (plus récent en premier)
            models.sort(key=lambda x: x["file_info"].get("saved_at", ""), reverse=True)
            
        except Exception as e:
            print(f"Erreur lors de la liste des modèles: {e}")
        
        return models
    
    def delete_model(self, filename):
        """
        Supprimer un modèle sauvegardé
        
        Args:
            filename (str): Nom de base du fichier
            
        Returns:
            dict: Résultat de la suppression
        """
        metadata_path = os.path.join(self.models_directory, f"{filename}_metadata.json")
        network_path = os.path.join(self.models_directory, f"{filename}_network.pkl")
        
        try:
            files_deleted = 0
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
                files_deleted += 1
                
            if os.path.exists(network_path):
                os.remove(network_path)
                files_deleted += 1
            
            return {
                "success": True,
                "files_deleted": files_deleted
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_file_sizes(self, filename):
        """Obtenir la taille des fichiers d'un modèle"""
        metadata_path = os.path.join(self.models_directory, f"{filename}_metadata.json")
        network_path = os.path.join(self.models_directory, f"{filename}_network.pkl")
        
        sizes = {}
        
        try:
            if os.path.exists(metadata_path):
                sizes["metadata_size"] = os.path.getsize(metadata_path)
            
            if os.path.exists(network_path):
                sizes["network_size"] = os.path.getsize(network_path)
                
            sizes["total_size"] = sum(sizes.values())
            
        except Exception:
            sizes = {"total_size": 0}
        
        return sizes
    
    def export_model_summary(self, filename):
        """
        Exporter un résumé du modèle en format texte
        
        Args:
            filename (str): Nom de base du fichier
            
        Returns:
            str: Résumé formaté du modèle
        """
        model_data = self.load_model(filename)
        
        if not model_data["success"]:
            return f"Erreur: {model_data['error']}"
        
        model_info = model_data["model_info"]
        arch = model_data["network_architecture"]
        
        summary = f"""═══════════════════════════════════════════════════════
RÉSUMÉ DU MODÈLE: {model_info['name']}
═══════════════════════════════════════════════════════

📊 INFORMATIONS GÉNÉRALES:
• Nom du modèle: {model_info['name']}
• Type: {model_info['model_type']}
• Précision: {model_info['accuracy']}
• Date de création: {model_info['created_date']}

🎯 CONFIGURATION DES DONNÉES:
• Variable cible: {model_info['target_column']}
• Variables explicatives: {len(model_info['feature_columns'])} variables
• Fichier de données: {model_info['data_file']}
• Taille des données: {model_info['data_shape'][0]} lignes × {model_info['data_shape'][1]} colonnes

🧠 ARCHITECTURE DU RÉSEAU:
• Couches: {' → '.join(map(str, arch.get('layers', [])))}
• Fonction d'activation: {arch.get('activation_function', 'Non spécifiée')}

📁 FICHIERS:
• Fichier: {filename}
• Sauvegardé le: {model_data['file_info'].get('saved_at', 'Date inconnue')}

═══════════════════════════════════════════════════════
"""
        return summary

# Utilitaires pour la persistence globale
def format_file_size(size_bytes):
    """Formater la taille d'un fichier en unités lisibles"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"
