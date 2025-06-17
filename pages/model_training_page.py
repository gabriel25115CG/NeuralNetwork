import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import sys
import os

# Ajouter le chemin vers neural_network
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_network.network import NeuralNetwork
from neural_network.utils import mean_squared_error, mean_absolute_error, r2_score
from neural_network.model_persistence import ModelPersistence

def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    """Fonction simple pour diviser les données en train/test"""
    import random
    random.seed(random_state)
    
    n_samples = len(X)
    n_test = int(n_samples * test_size)
    
    # Créer les indices et les mélanger
    indices = list(range(n_samples))
    random.shuffle(indices)
    
    # Diviser les indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    # Créer les datasets
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    
    return X_train, X_test, y_train, y_test

def normalize_data(data):
    """Normalisation simple min-max"""
    if len(data) == 0:
        return data, 0, 1
    
    data_min = min(data)
    data_max = max(data)
    
    if data_max == data_min:
        return [0.5] * len(data), data_min, data_max
    
    normalized = [(x - data_min) / (data_max - data_min) for x in data]
    return normalized, data_min, data_max

class ModelTrainingPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.model_info = None
        self.df = None
        self.model = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None        # Paramètres de normalisation custom
        self.data_min = None
        self.data_max = None
        self.target_min = None
        self.target_max = None
        self.training_progress = 0
        # Système de persistence
        self.model_persistence = ModelPersistence()
        self.normalization_params = {}
        
        # Variables pour le suivi de progression amélioré
        self.start_time = None
        self.current_epoch = 0
        self.total_epochs = 0
        self.create_widgets()
        
    def create_widgets(self):
        # Cadre principal avec scroll
        main_frame = tk.Frame(self, bg=self.controller.colors["bg_light"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # En-tête
        header_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Button(
            header_frame,
            text="← Retour",
            command=lambda: self.controller.show_page("DataPreviewPage"),
            style='TButton'
        ).pack(side=tk.LEFT)
        
        self.title_label = tk.Label(
            main_frame,
            text="Entraînement du modèle",
            font=('Helvetica', 18, 'bold'),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["primary"]
        )
        self.title_label.pack(pady=(0, 20))
        
        # Container principal avec scroll
        canvas_container = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.main_canvas = tk.Canvas(canvas_container, bg=self.controller.colors["bg_light"])
        scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.main_canvas.yview)
        self.scrollable_content = tk.Frame(self.main_canvas, bg=self.controller.colors["bg_light"])
        
        self.scrollable_content.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        self.main_canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")
        self.main_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Bind mousewheel
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.create_training_content()
        
    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def create_training_content(self):
        # Section 1: Configuration du modèle
        config_section = tk.LabelFrame(
            self.scrollable_content,
            text="Configuration du modèle de réseau de neurones",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15        )
        config_section.pack(fill=tk.X, pady=(0, 20))
        
        # Paramètres d'entraînement
        tk.Label(
            config_section,
            text="Paramètres d'entraînement:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(10, 5))
        
        params_frame = tk.Frame(config_section, bg=self.controller.colors["bg_white"])
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Taux d'apprentissage
        tk.Label(
            params_frame,
            text="Taux d'apprentissage:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.learning_rate_var = tk.DoubleVar(value=0.01)
        learning_rate_scale = tk.Scale(
            params_frame,
            from_=0.001,
            to=0.1,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.learning_rate_var,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            length=200
        )
        learning_rate_scale.grid(row=0, column=1, sticky="w")
        
        # Époques
        tk.Label(
            params_frame,
            text="Nombre d'époques:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        
        self.epochs_var = tk.IntVar(value=200)
        epochs_scale = tk.Scale(
            params_frame,
            from_=50,
            to=1000,
            resolution=50,
            orient=tk.HORIZONTAL,
            variable=self.epochs_var,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            length=200
        )
        epochs_scale.grid(row=1, column=1, sticky="w", pady=(10, 0))
        
        # Test size
        tk.Label(
            params_frame,
            text="Taille du jeu de test (%):",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).grid(row=2, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_scale = tk.Scale(
            params_frame,
            from_=0.1,
            to=0.5,
            resolution=0.05,            orient=tk.HORIZONTAL,
            variable=self.test_size_var,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            length=200
        )
        test_size_scale.grid(row=2, column=1, sticky="w", pady=(10, 0))
        
        # Section 2: Progression de l'entraînement
        self.progress_section = tk.LabelFrame(
            self.scrollable_content,
            text="📊 Progression de l'entraînement",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.progress_section.pack(fill=tk.X, pady=(0, 20))
        
        # Frame pour la barre de progression et le pourcentage
        progress_frame = tk.Frame(self.progress_section, bg=self.controller.colors["bg_white"])
        progress_frame.pack(fill=tk.X, pady=(10, 5))
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=350
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Label pour le pourcentage
        self.progress_percentage_label = tk.Label(
            progress_frame,
            text="0%",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["primary"],
            font=('Helvetica', 12, 'bold'),
            width=5
        )
        self.progress_percentage_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Label pour l'état actuel
        self.progress_label = tk.Label(
            self.progress_section,
            text="🔄 Prêt à commencer l'entraînement",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        )
        self.progress_label.pack(pady=(5, 5))
        
        # Label pour le timing et les détails
        self.timing_label = tk.Label(
            self.progress_section,
            text="⏱️ Temps écoulé: 00:00 | Estimation restante: --:--",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 9)
        )
        self.timing_label.pack(pady=(0, 10))
        
        # Section 3: Résultats
        self.results_section = tk.LabelFrame(
            self.scrollable_content,
            text="Résultats de l'entraînement",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.results_section.pack(fill=tk.X, pady=(0, 20))
        
        self.results_text = tk.Text(
            self.results_section,
            height=8,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Courier', 10),
            state=tk.DISABLED
        )
        self.results_text.pack(fill=tk.X, pady=(10, 10))
        
        # Section 4: Visualisation
        self.viz_section = tk.LabelFrame(
            self.scrollable_content,
            text="Visualisation des résultats",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.viz_section.pack(fill=tk.X, pady=(0, 20))
        
        # Section 5: Sauvegarde
        self.save_section = tk.LabelFrame(
            self.scrollable_content,
            text="Sauvegarde du modèle",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.save_section.pack(fill=tk.X, pady=(0, 20))
        
        # Nom personnalisé pour la sauvegarde
        tk.Label(
            self.save_section,
            text="Nom de sauvegarde (optionnel):",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).pack(anchor="w", pady=(10, 5))
        
        self.save_name_var = tk.StringVar()
        save_name_entry = tk.Entry(
            self.save_section,
            textvariable=self.save_name_var,
            width=40,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"]
        )
        save_name_entry.pack(anchor="w", pady=(0, 10))
        
        # Boutons d'action
        button_frame = tk.Frame(self.scrollable_content, bg=self.controller.colors["bg_light"])
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Bouton Annuler
        ttk.Button(
            button_frame,
            text="Annuler",
            command=lambda: self.controller.show_page("DataPreviewPage"),
            style='TButton',
            cursor="hand2"
        ).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Bouton Entraîner
        self.train_button = ttk.Button(
            button_frame,
            text="🚀 Commencer l'entraînement",
            command=self.start_training,
            style='Add.TButton',
            cursor="hand2"
        )
        self.train_button.pack(side=tk.RIGHT)
        
        # Bouton Sauvegarder (initialement caché)
        self.save_button = ttk.Button(
            button_frame,
            text="💾 Sauvegarder le modèle",
            command=self.save_model,
            style='Add.TButton',
            cursor="hand2"        )
        
    def load_training_data(self, file_path, model_info):
        """Charger les données pour l'entraînement"""
        try:
            self.df = pd.read_csv(file_path, low_memory=False)
            self.model_info = model_info
            
            # Debug: afficher l'architecture reçue
            if 'network_architecture' in model_info:
                print(f"🔍 DEBUG: Architecture reçue dans model_training_page: {model_info['network_architecture']}")
            else:
                print("⚠️ DEBUG: Aucune architecture trouvée dans model_info")
            
            self.title_label.config(text=f"Entraînement - {model_info['name']}")
            
            # Afficher les informations du modèle
            info_text = f"""Configuration du modèle:
🎯 Variable cible: {model_info['target_column']}
📊 Variables explicatives: {len(model_info['feature_columns'])} variables
📋 Données: {model_info['data_shape'][0]} lignes × {model_info['data_shape'][1]} colonnes
📁 Fichier: {model_info['data_file']}
🧠 Type: Réseau de neurones personnalisé"""
            
            self.update_results(info_text)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les données:\n{str(e)}")
    
    def start_training(self):
        """Commencer l'entraînement du modèle"""
        if self.model_info is None:
            messagebox.showwarning("Attention", "Aucune donnée chargée")
            return
        
        # Réinitialiser l'interface
        self.train_button.config(state="disabled")
        self.progress_bar["value"] = 0
        self.progress_percentage_label.config(text="0%")
        
        # Démarrer le timer
        self.start_time = time.time()
        self.current_epoch = 0
        self.total_epochs = self.epochs_var.get()
        
        # Lancer l'entraînement dans un thread séparé
        threading.Thread(target=self.train_model, daemon=True).start()
    
    def train_model(self):
        """Entraîner le modèle (dans un thread séparé)"""
        try:
            # Étape 1: Préparation des données
            self.update_progress(10, "Préparation des données...")
            
            target_col = self.model_info['target_column']
            feature_cols = self.model_info['feature_columns']
            
            # Vérifier et convertir les colonnes en numérique
            data_subset = self.df[feature_cols + [target_col]].copy()
            
            # Convertir la variable cible en numérique
            data_subset[target_col] = pd.to_numeric(data_subset[target_col], errors='coerce')
            
            # Convertir les variables explicatives en numérique
            for col in feature_cols:
                data_subset[col] = pd.to_numeric(data_subset[col], errors='coerce')
            
            # Supprimer les lignes avec des valeurs manquantes après conversion
            clean_data = data_subset.dropna()
            
            if len(clean_data) == 0:
                raise ValueError("Aucune donnée numérique valide trouvée après conversion")
            
            if len(clean_data) < 10:
                raise ValueError(f"Pas assez de données valides ({len(clean_data)} lignes). Minimum requis: 10 lignes")
            
            # Extraire X et y
            X = clean_data[feature_cols].values.tolist()
            y = clean_data[target_col].values.tolist()
            
            self.update_progress(20, f"Données nettoyées: {len(clean_data)} échantillons valides...")
              # Étape 2: Division train/test
            self.update_progress(25, "Division des données...")
            
            test_size = self.test_size_var.get()
            random_state = 42  # Valeur fixe pour la reproductibilité
            
            self.X_train, self.X_test, self.y_train, self.y_test = custom_train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Vérifier qu'on a assez de données pour l'entraînement et le test
            if len(self.X_train) < 5:
                raise ValueError(f"Pas assez de données d'entraînement ({len(self.X_train)} échantillons). Réduisez le pourcentage de test.")
            
            if len(self.X_test) < 3:
                raise ValueError(f"Pas assez de données de test ({len(self.X_test)} échantillons). Augmentez la taille du dataset ou réduisez le pourcentage de test.")
            
            self.update_progress(30, f"Train: {len(self.X_train)} échantillons, Test: {len(self.X_test)} échantillons...")
            
            # Étape 3: Normalisation
            self.update_progress(40, "Normalisation des données...")
            
            # Normaliser les variables explicatives
            X_train_lists = [list(row) for row in self.X_train]
            X_test_lists = [list(row) for row in self.X_test]
            
            # Normaliser chaque caractéristique séparément
            self.X_train_normalized = []
            self.X_test_normalized = []
            self.feature_mins = []
            self.feature_maxs = []
            
            n_features = len(X_train_lists[0])
            
            for i in range(n_features):
                # Extraire la i-ème caractéristique
                feature_train = [row[i] for row in X_train_lists]
                feature_test = [row[i] for row in X_test_lists]
                
                # Normaliser
                normalized_train, feat_min, feat_max = normalize_data(feature_train)
                normalized_test, _, _ = normalize_data(feature_test)  # Utiliser les mêmes min/max
                
                # Stocker les paramètres
                self.feature_mins.append(feat_min)
                self.feature_maxs.append(feat_max)
                
                # Première itération : créer les listes
                if i == 0:
                    self.X_train_normalized = [[val] for val in normalized_train]
                    self.X_test_normalized = [[val] for val in normalized_test]
                else:
                    # Ajouter à chaque échantillon
                    for j, val in enumerate(normalized_train):
                        self.X_train_normalized[j].append(val)
                    for j, val in enumerate(normalized_test):
                        self.X_test_normalized[j].append(val)
            
            # Normaliser la variable cible
            self.y_train_normalized, self.target_min, self.target_max = normalize_data(self.y_train)
            self.y_test_normalized, _, _ = normalize_data(self.y_test)
            
            # Stocker les paramètres de normalisation
            self.normalization_params = {
                'feature_mins': self.feature_mins,
                'feature_maxs': self.feature_maxs,
                'target_min': self.target_min,
                'target_max': self.target_max
            }
            
            # Étape 4: Entraînement du modèle
            self.update_progress(60, "Entraînement du modèle...")            # Configuration du réseau de neurones
            n_features = len(self.X_train_normalized[0])
              # Utiliser l'architecture configurée lors de la création du modèle
            if 'network_architecture' in self.model_info:
                layers_config = self.model_info['network_architecture'].copy()
                configured_inputs = layers_config[0]
                
                # Vérifier si le nombre d'entrées configuré correspond aux données
                if configured_inputs != n_features:
                    print(f"⚠️ ATTENTION: Architecture configurée avec {configured_inputs} entrées, mais {n_features} features détectées")
                    print(f"   🔧 Utilisation de l'architecture configurée: {layers_config}")
                    
                    # Si plus de features que d'entrées configurées, prendre les premières
                    if n_features > configured_inputs:
                        print(f"   📊 Sélection des {configured_inputs} premières features")
                        # Adapter les données pour correspondre à l'architecture
                        self.X_train_normalized = [x[:configured_inputs] for x in self.X_train_normalized]
                        self.X_test_normalized = [x[:configured_inputs] for x in self.X_test_normalized]
                    elif n_features < configured_inputs:
                        print(f"   ❌ ERREUR: Pas assez de features ({n_features}) pour l'architecture configurée ({configured_inputs})")                        # Dans ce cas, on adapte l'architecture
                        layers_config[0] = n_features
                        print(f"   🔧 Architecture adaptée: {layers_config}")
                    else:
                        print(f"✅ Architecture configurée parfaitement adaptée: {layers_config}")
                
                print(f"   Architecture originale: {self.model_info['network_architecture']}")
                print(f"   Architecture finale: {layers_config}")
            else:
                # Si aucune architecture n'est configurée, utiliser une architecture par défaut
                layers_config = [n_features, 8, 1]  # Architecture par défaut simple
                print(f"⚠️ Architecture par défaut utilisée: {layers_config}")
            
            # Créer le réseau de neurones
            self.model = NeuralNetwork(layers_config)
            print(f"🧠 Réseau créé avec layers_config: {self.model.layers_config}")
            
            # Mettre à jour l'architecture dans model_info avec celle réellement utilisée
            self.model_info['network_architecture'] = layers_config.copy()
            self.model_info['layers_config'] = layers_config.copy()  # Alias pour compatibilité            # Entraîner le réseau
            learning_rate = self.learning_rate_var.get()
            epochs = self.epochs_var.get()
            losses = self.model.train(
                self.X_train_normalized, 
                self.y_train_normalized, 
                learning_rate=learning_rate, 
                epochs=epochs, 
                verbose=False,
                progress_callback=self.training_progress_callback
            )
            
            # Stocker l'historique des losses pour la sauvegarde ET la visualisation
            self.losses = losses
            # Aussi stocker dans le modèle pour que la visualisation puisse y accéder
            self.model.losses = losses
            
            # Étape 5: Évaluation
            self.update_progress(80, "Évaluation du modèle...")
            
            # Prédictions sur les données d'entraînement et de test
            y_train_pred_norm = [self.model.predict(x)[0] for x in self.X_train_normalized]
            y_test_pred_norm = [self.model.predict(x)[0] for x in self.X_test_normalized]
              # Dénormaliser les prédictions
            y_train_pred = [pred * (self.target_max - self.target_min) + self.target_min for pred in y_train_pred_norm]
            y_test_pred = [pred * (self.target_max - self.target_min) + self.target_min for pred in y_test_pred_norm]
            
            # Stocker les prédictions pour la sauvegarde
            self.y_test_pred = y_test_pred
            self.y_train_pred = y_train_pred
            
            # Calculer les métriques
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
            # Stocker les métriques pour la sauvegarde
            self.test_mse = mean_squared_error(self.y_test, y_test_pred)
            self.test_mae = test_mae
            self.test_r2 = test_r2
            
            # Étape 6: Résultats avec analyse intelligente
            self.update_progress(95, "Génération des résultats...")
            
            # Analyser la qualité du modèle
            quality_status = "🟢 Excellent" if test_r2 > 0.8 else "🟡 Correct" if test_r2 > 0.6 else "🔴 Faible"
            
            # Diagnostics et recommandations
            diagnostics = []
            recommendations = []
            
            if test_r2 < 0.1:
                diagnostics.append("• R² très faible : le modèle ne capture pas les relations")
                recommendations.append("• Vérifiez la corrélation entre variables")
                recommendations.append("• Essayez d'augmenter le nombre de neurones cachés")
                recommendations.append("• Augmentez le nombre d'époques")
            
            if abs(train_r2 - test_r2) > 0.2:
                diagnostics.append("• Écart important train/test : possible surapprentissage")
                recommendations.append("• Augmentez la taille du jeu de test")
                recommendations.append("• Réduisez le nombre de neurones cachés")
                recommendations.append("• Diminuez le taux d'apprentissage")
            
            if train_rmse > test_rmse:
                diagnostics.append("• RMSE entraînement > test : données de test plus simples")
            
            # Messages d'analyse
            diagnostic_text = "\n".join(diagnostics) if diagnostics else "• Aucun problème majeur détecté"
            recommendation_text = "\n".join(recommendations) if recommendations else "• Le modèle semble correct"
            
            results = f"""✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS

📊 MÉTRIQUES DE PERFORMANCE:
═══════════════════════════════════
• R² Score (Entraînement): {train_r2:.4f}
• R² Score (Test):         {test_r2:.4f}
• RMSE (Entraînement):     {train_rmse:.4f}
• RMSE (Test):             {test_rmse:.4f}
• MAE (Entraînement):      {train_mae:.4f}
• MAE (Test):              {test_mae:.4f}

🎯 QUALITÉ DU MODÈLE:
═══════════════════════════════════
• Coefficient de détermination: {test_r2:.1%}
• Erreur quadratique moyenne: {test_rmse:.2f}
• {quality_status} (R² = {test_r2:.3f})

🧠 ARCHITECTURE DU RÉSEAU:
═══════════════════════════════════
• Configuration: {' → '.join(map(str, layers_config))}
• Nombre de couches: {len(layers_config)}
• Couches cachées: {len(layers_config) - 2}
• Paramètres totaux: {sum(layers_config[i] * (layers_config[i-1] + 1) for i in range(1, len(layers_config)))}
• Taux d'apprentissage: {learning_rate}
• Époques: {epochs}

🔍 DIAGNOSTIC:
═══════════════════════════════════
{diagnostic_text}

💡 RECOMMANDATIONS:
═══════════════════════════════════
{recommendation_text}

📋 INFORMATIONS TECHNIQUES:
═══════════════════════════════════
• Type de régression: Réseau de Neurones Personnalisé
• Données d'entraînement: {len(self.X_train)} échantillons
• Données de test: {len(self.X_test)} échantillons
• Variables utilisées: {len(feature_cols)}
• Loss finale: {losses[-1]:.6f}"""
              # Mettre à jour l'interface dans le thread principal
            self.after(0, lambda: self.training_completed(results, y_test_pred))
            
        except Exception as e:
            error_message = str(e)
            self.after(0, lambda: self.training_failed(error_message))
    
    def update_progress(self, value, text, epoch=None, total_epochs=None):
        """Mettre à jour la barre de progression avec pourcentage et timing"""
        def update():
            # Mettre à jour la barre de progression
            self.progress_bar["value"] = value
            self.progress_percentage_label.config(text=f"{int(value)}%")
            
            # Mettre à jour le texte de statut
            if epoch is not None and total_epochs is not None:
                self.current_epoch = epoch
                self.total_epochs = total_epochs
                epoch_text = f"📈 Époque {epoch}/{total_epochs} - {text}"
            else:
                epoch_text = text
                
            self.progress_label.config(text=epoch_text)
            
            # Calculer et afficher le timing
            if self.start_time is not None:
                elapsed_time = time.time() - self.start_time
                elapsed_str = self.format_time(elapsed_time)
                
                # Estimation du temps restant
                if value > 0:
                    estimated_total = elapsed_time * (100 / value)
                    remaining_time = estimated_total - elapsed_time
                    remaining_str = self.format_time(max(0, remaining_time))
                else:
                    remaining_str = "--:--"
                
                timing_text = f"⏱️ Temps écoulé: {elapsed_str} | Estimation restante: {remaining_str}"
                self.timing_label.config(text=timing_text)
        
        self.after(0, update)
        time.sleep(0.05)  # Délai réduit pour une progression plus fluide
    
    def format_time(self, seconds):
        """Formater le temps en MM:SS"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def training_progress_callback(self, epoch, total_epochs, loss):
        """Callback appelé à chaque époque pour mettre à jour la progression"""
        # Calculer le pourcentage de progression de l'entraînement (entre 60% et 80%)
        training_progress = 60 + (epoch / total_epochs) * 20
        
        # Texte avec détails de l'époque
        progress_text = f"🧠 Entraînement époque {epoch}/{total_epochs} (loss: {loss:.6f})"
        
        # Mettre à jour la progression avec les détails d'époque
        self.update_progress(training_progress, progress_text, epoch, total_epochs)
    
    def training_completed(self, results, y_pred):
        """Actions à effectuer quand l'entraînement est terminé"""
        self.update_progress(100, "✅ Entraînement terminé avec succès!")
        self.update_results(results)
        self.create_visualization(y_pred)
        
        # Activer le bouton de sauvegarde
        self.save_button.pack(side=tk.RIGHT, padx=(0, 10))
        self.train_button.config(state="normal", text="🔄 Réentraîner")
        
        # Mettre à jour les informations du modèle
        test_r2 = r2_score(self.y_test, y_pred)
        self.model_info["accuracy"] = f"{test_r2:.1%}"
        self.model_info["model"] = self.model
        self.model_info["normalization_params"] = self.normalization_params
        
    def training_failed(self, error):
        """Actions à effectuer en cas d'échec"""
        self.progress_bar["value"] = 0
        self.progress_label.config(text="❌ Échec de l'entraînement")
        self.update_results(f"❌ ERREUR LORS DE L'ENTRAÎNEMENT:\n\n{error}")
        self.train_button.config(state="normal")
        messagebox.showerror("Erreur", f"Échec de l'entraînement:\n{error}")
    
    def update_results(self, text):
        """Mettre à jour le texte des résultats"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.results_text.config(state=tk.DISABLED)
    
    def create_visualization(self, y_pred):
        """Créer les graphiques de visualisation optimisés pour les réseaux de neurones"""
        try:
            # Nettoyer la section de visualisation
            for widget in self.viz_section.winfo_children():
                widget.destroy()
            
            # Vérifier qu'on a assez de données
            if len(self.y_test) < 3:
                warning_label = tk.Label(
                    self.viz_section,
                    text=f"⚠️ Attention: Seulement {len(self.y_test)} échantillons de test.\nAugmentez la taille du dataset ou réduisez le pourcentage de test.",
                    bg=self.controller.colors["bg_white"],
                    fg=self.controller.colors["accent"],
                    font=('Helvetica', 11),
                    justify=tk.CENTER
                )
                warning_label.pack(pady=20)
                return
            
            # Dénormaliser les données pour l'affichage si nécessaire
            y_test_display = self.y_test.copy()
            y_pred_display = y_pred.copy()
            
            if hasattr(self, 'target_min') and hasattr(self, 'target_max') and self.target_min is not None:
                y_test_display = [y * (self.target_max - self.target_min) + self.target_min for y in self.y_test]
                y_pred_display = [y * (self.target_max - self.target_min) + self.target_min for y in y_pred]
            
            # Créer la figure avec 3 graphiques optimisés
            fig = plt.figure(figsize=(16, 5))
            fig.patch.set_facecolor('white')
            fig.suptitle('🧠 Analyse de Performance du Réseau de Neurones', fontsize=16, fontweight='bold', y=0.98)
            
            # === GRAPHIQUE 1: Performance de Prédiction (le plus important) ===
            ax1 = plt.subplot(131)
            
            # Calculer les erreurs pour la colorisation
            absolute_errors = [abs(real - pred) for real, pred in zip(y_test_display, y_pred_display)]
            
            # Nuage de points avec colorisation par erreur
            scatter = ax1.scatter(y_test_display, y_pred_display, c=absolute_errors, 
                                 alpha=0.8, s=80, cmap='RdYlGn_r',
                                 edgecolors='black', linewidth=0.5)
            
            # Ligne parfaite (y=x)
            min_val = min(min(y_test_display), min(y_pred_display))
            max_val = max(max(y_test_display), max(y_pred_display))
            perfect_line = [min_val, max_val]
            ax1.plot(perfect_line, perfect_line, 'k--', alpha=0.8, linewidth=2, label='Prédiction parfaite')
            
            # Zones de tolérance
            margin = (max_val - min_val) * 0.05
            x_range = [min_val - margin + i * (max_val - min_val + 2*margin) / 99 for i in range(100)]
            ax1.fill_between(x_range, [x * 0.9 for x in x_range], [x * 1.1 for x in x_range], 
                           alpha=0.2, color='green', label='±10% (Excellent)')
            ax1.fill_between(x_range, [x * 0.8 for x in x_range], [x * 0.9 for x in x_range], 
                           alpha=0.1, color='orange')
            ax1.fill_between(x_range, [x * 1.1 for x in x_range], [x * 1.2 for x in x_range], 
                           alpha=0.1, color='orange', label='±20% (Acceptable)')
            
            # Métriques importantes
            from neural_network.utils import r2_score, mean_absolute_error
            r2 = r2_score(y_test_display, y_pred_display)
            mae = mean_absolute_error(y_test_display, y_pred_display)
            
            ax1.set_xlabel("Valeurs Réelles (€)", fontsize=11, fontweight='bold')
            ax1.set_ylabel("Prédictions du Réseau (€)", fontsize=11, fontweight='bold')
            ax1.set_title(f"Performance de Prédiction\nR² = {r2:.3f} | MAE = {mae:.0f}€", fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=9)
            
            # Colorbar pour les erreurs
            try:
                cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8, pad=0.02)
                cbar.set_label('Erreur Absolue (€)', rotation=270, labelpad=15, fontsize=10)
            except:
                pass
              # === GRAPHIQUE 2: Historique de l'Entraînement (Loss) ===
            ax2 = plt.subplot(132)
            
            # Accéder à l'historique des losses sauvegardé pendant l'entraînement
            if hasattr(self, 'losses') and self.losses:
                losses = self.losses
                epochs = list(range(1, len(losses) + 1))
                
                ax2.plot(epochs, losses, 'b-', linewidth=2, alpha=0.8, label='Loss d\'entraînement')
                ax2.fill_between(epochs, losses, alpha=0.3, color='blue')
                
                # Ligne de tendance lissée
                if len(losses) > 10:
                    window_size = max(3, len(losses) // 10)
                    smoothed_losses = []
                    for i in range(len(losses)):
                        start = max(0, i - window_size // 2)
                        end = min(len(losses), i + window_size // 2 + 1)
                        smoothed_losses.append(sum(losses[start:end]) / (end - start))
                    ax2.plot(epochs, smoothed_losses, 'r--', linewidth=2, alpha=0.8, label='Tendance lissée')
                
                ax2.set_xlabel("Époques", fontsize=11, fontweight='bold')
                ax2.set_ylabel("Loss (MSE)", fontsize=11, fontweight='bold')
                ax2.set_title(f"Convergence du Réseau\n{len(losses)} époques - Final: {losses[-1]:.4f}", fontsize=12, fontweight='bold')
                ax2.grid(True, alpha=0.3)
                ax2.legend(fontsize=9)
                ax2.set_yscale('log')  # Échelle log pour mieux voir la convergence
                
            else:
                # Si pas d'historique, créer un graphique avec les données disponibles
                ax2.text(0.5, 0.5, 'Historique de Loss\nnon disponible\n\nVérifiez que l\'entraînement\na été effectué correctement', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax2.transAxes, fontsize=11, style='italic',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
                ax2.set_title("Convergence du Réseau", fontsize=12, fontweight='bold')
              # === GRAPHIQUE 3: Graphique en Secteurs de la Qualité des Prédictions ===
            ax3 = plt.subplot(133)
            
            # Calcul des erreurs relatives en pourcentage
            relative_errors = [(pred - real) / real * 100 for real, pred in zip(y_test_display, y_pred_display) if real != 0]
            
            if relative_errors:
                # Compter les prédictions par niveau de qualité
                excellent = sum(1 for e in relative_errors if abs(e) <= 5)  # ±5%
                good = sum(1 for e in relative_errors if 5 < abs(e) <= 10)  # ±5-10%
                acceptable = sum(1 for e in relative_errors if 10 < abs(e) <= 20)  # ±10-20%
                poor = sum(1 for e in relative_errors if abs(e) > 20)  # >±20%
                
                total_predictions = len(relative_errors)
                
                # Données pour le graphique en secteurs
                sizes = [excellent, good, acceptable, poor]
                labels = [f'🎯 Excellent (±5%)\n{excellent} prédictions\n({excellent/total_predictions*100:.1f}%)', 
                         f'✅ Bon (±10%)\n{good} prédictions\n({good/total_predictions*100:.1f}%)',
                         f'⚠️ Acceptable (±20%)\n{acceptable} prédictions\n({acceptable/total_predictions*100:.1f}%)', 
                         f'❌ Médiocre (>±20%)\n{poor} prédictions\n({poor/total_predictions*100:.1f}%)']
                colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']  # Vert, Orange, Orange foncé, Rouge
                explode = (0.05, 0, 0, 0.1 if poor > 0 else 0)  # Faire ressortir excellent et médiocre si il y en a
                
                # Filtrer les sections vides pour éviter les erreurs
                non_zero_data = [(size, label, color, exp) for size, label, color, exp in zip(sizes, labels, colors, explode) if size > 0]
                if non_zero_data:
                    sizes_filtered, labels_filtered, colors_filtered, explode_filtered = zip(*non_zero_data)
                    
                    # Créer le graphique en secteurs
                    wedges, texts, autotexts = ax3.pie(sizes_filtered, labels=labels_filtered, colors=colors_filtered, 
                                                      explode=explode_filtered, autopct='%1.1f%%', startangle=90,
                                                      textprops={'fontsize': 8, 'weight': 'bold'})
                    
                    # Améliorer l'apparence du texte
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                        autotext.set_fontsize(9)
                else:
                    # Si toutes les sections sont vides, afficher un message
                    ax3.text(0.5, 0.5, 'Aucune donnée\nd\'erreur disponible', 
                            horizontalalignment='center', verticalalignment='center',
                            transform=ax3.transAxes, fontsize=12, style='italic')
                
                ax3.set_title('🎯 Qualité des Prédictions du Réseau\n(Répartition par niveau de précision)', 
                             fontsize=11, fontweight='bold', pad=15)
                
                # Ajouter un résumé statistique en bas
                mean_error = sum(relative_errors) / len(relative_errors)
                std_error = (sum((e - mean_error)**2 for e in relative_errors) / len(relative_errors))**0.5
                precision_5 = (excellent/total_predictions*100) if total_predictions > 0 else 0
                precision_10 = ((excellent+good)/total_predictions*100) if total_predictions > 0 else 0
                
                stats_text = f"""📈 RÉSUMÉ:
Erreur moyenne: {mean_error:.1f}%
Précision ±5%: {precision_5:.1f}%
Précision ±10%: {precision_10:.1f}%"""
                
                ax3.text(0.5, -0.15, stats_text, transform=ax3.transAxes, fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.3),
                        verticalalignment='top', horizontalalignment='center')
                
            else:
                ax3.text(0.5, 0.5, 'Pas de données\nd\'erreur disponibles', 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax3.transAxes, fontsize=12, style='italic')
            
            plt.tight_layout()
            
            # Intégrer dans l'interface
            canvas = FigureCanvasTkAgg(fig, master=self.viz_section)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
            
            # === SECTION STATISTIQUES AMÉLIORÉE ===
            stats_frame = tk.LabelFrame(
                self.viz_section, 
                text="📊 Métriques de Performance du Réseau",
                font=('Helvetica', 12, 'bold'),
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["primary"],
                padx=20, pady=15
            )
            stats_frame.pack(fill=tk.X, pady=10, padx=20)
            
            # Calculer des métriques avancées
            mse = mean_squared_error(y_test_display, y_pred_display)
            rmse = mse ** 0.5
            mape = sum(abs((real - pred) / real) for real, pred in zip(y_test_display, y_pred_display) if real != 0) / len(y_test_display) * 100
            
            # Pourcentage de prédictions dans différentes tolérances
            tolerance_5 = sum(1 for real, pred in zip(y_test_display, y_pred_display) if abs((pred - real) / real) <= 0.05) / len(y_test_display) * 100
            tolerance_10 = sum(1 for real, pred in zip(y_test_display, y_pred_display) if abs((pred - real) / real) <= 0.10) / len(y_test_display) * 100
            tolerance_20 = sum(1 for real, pred in zip(y_test_display, y_pred_display) if abs((pred - real) / real) <= 0.20) / len(y_test_display) * 100
            
            # Organisez les statistiques en colonnes
            stats_left = tk.Frame(stats_frame, bg=self.controller.colors["bg_white"])
            stats_left.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            stats_right = tk.Frame(stats_frame, bg=self.controller.colors["bg_white"])
            stats_right.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            left_text = f"""🎯 PRÉCISION DU MODÈLE:
• R² (Coefficient de détermination): {r2:.4f}
• MAE (Erreur absolue moyenne): {mae:.0f} €
• RMSE (Racine de l'erreur quadratique): {rmse:.0f} €
• MAPE (Erreur absolue moyenne en %): {mape:.2f}%"""
            
            right_text = f"""✅ TOLÉRANCE DE PRÉDICTION:
• ±5% de précision: {tolerance_5:.1f}% des prédictions
• ±10% de précision: {tolerance_10:.1f}% des prédictions  
• ±20% de précision: {tolerance_20:.1f}% des prédictions
• Échantillons de test: {len(y_test_display)}"""
            
            tk.Label(stats_left, text=left_text, bg=self.controller.colors["bg_white"],
                    fg=self.controller.colors["text"], font=('Helvetica', 10), justify=tk.LEFT).pack(anchor="w")
            
            tk.Label(stats_right, text=right_text, bg=self.controller.colors["bg_white"],
                    fg=self.controller.colors["text"], font=('Helvetica', 10), justify=tk.LEFT).pack(anchor="w")
            
        except Exception as e:
            error_label = tk.Label(
                self.viz_section,
                text=f"Erreur lors de la création des graphiques: {str(e)}",
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["accent"],
                font=('Helvetica', 10)
            )
            error_label.pack(pady=10)
            print(f"Erreur détaillée: {e}")  # Pour debug
    
    def save_model(self):
        """Sauvegarder le modèle entraîné avec le système de persistence"""
        try:
            if not self.model:
                messagebox.showwarning("Attention", "Aucun modèle entraîné à sauvegarder")
                return
            
            # Nom personnalisé ou utiliser le nom du modèle
            save_name = self.save_name_var.get().strip()
            if save_name:
                self.model_info['name'] = save_name
            
            # Ajouter la date de création
            from datetime import datetime
            self.model_info['created_date'] = datetime.now().strftime("%d/%m/%Y à %H:%M")
              # Préparer les résultats d'entraînement pour la sauvegarde
            training_results = {}
            
            # Sauvegarder l'historique des losses si disponible
            if hasattr(self, 'losses') and self.losses:
                training_results['loss_history'] = self.losses
            
            # Sauvegarder les données de test pour recréer les graphiques identiques
            if hasattr(self, 'y_test') and hasattr(self, 'y_test_pred'):
                training_results['y_test_real'] = self.y_test  # Vraies valeurs (dénormalisées)
                training_results['y_test_pred'] = self.y_test_pred  # Prédictions (dénormalisées)
            
            # Sauvegarder les métriques calculées
            if hasattr(self, 'test_mse'):
                training_results['metrics'] = {
                    'mse': getattr(self, 'test_mse', 0),
                    'mae': getattr(self, 'test_mae', 0),
                    'r2': getattr(self, 'test_r2', 0)
                }
            
            # Sauvegarder des statistiques sur les données
            if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
                training_results['training_data_stats'] = {
                    'n_train_samples': len(self.X_train),
                    'n_test_samples': len(self.y_test) if hasattr(self, 'y_test') else 0,
                    'n_features': len(self.X_train[0]) if self.X_train else 0
                }
            
            # Sauvegarder avec le système de persistence
            save_result = self.model_persistence.save_model(
                self.model_info,
                self.model,
                self.normalization_params,
                training_results  # Nouveau paramètre
            )
            
            if save_result["success"]:
                # Ajouter le modèle à la liste de l'application
                if self.model_info not in self.controller.model_list:
                    self.controller.model_list.append(self.model_info)
                
                # Mettre à jour le statut
                if hasattr(self.controller, 'status_label'):
                    self.controller.status_label.config(
                        text=f"Modèle '{self.model_info['name']}' entraîné et sauvegardé avec succès"
                    )
                
                # Afficher un message de succès détaillé
                messagebox.showinfo(
                    "Succès",
                    f"Modèle '{self.model_info['name']}' sauvegardé avec succès!\n\n"
                    f"Précision: {self.model_info['accuracy']}\n"
                    f"Fichier: {save_result['filename']}\n"
                    f"Emplacement: {save_result['metadata_path']}"
                )
                
                # Retourner à la page d'accueil
                self.controller.show_page("HomePage")
                
            else:
                messagebox.showerror("Erreur", f"Impossible de sauvegarder le modèle:\n{save_result['error']}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder le modèle:\n{str(e)}")
