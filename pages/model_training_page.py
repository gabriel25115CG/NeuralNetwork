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
    
    def _on_horizontal_mousewheel(self, event):
        """Scroll horizontal avec Shift+molette"""
        self.viz_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
        
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
        
        # Section 4: Visualisation avec scroll horizontal
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
        
        # Container avec scroll horizontal pour les graphiques
        self.viz_container = tk.Frame(self.viz_section, bg=self.controller.colors["bg_white"])
        self.viz_container.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.viz_canvas = tk.Canvas(self.viz_container, bg=self.controller.colors["bg_white"], height=600)
        self.viz_scrollbar_h = ttk.Scrollbar(self.viz_container, orient="horizontal", command=self.viz_canvas.xview)
        self.viz_scrollable_frame = tk.Frame(self.viz_canvas, bg=self.controller.colors["bg_white"])
        
        self.viz_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.viz_canvas.configure(scrollregion=self.viz_canvas.bbox("all"))
        )
        
        self.viz_canvas.create_window((0, 0), window=self.viz_scrollable_frame, anchor="nw")
        self.viz_canvas.configure(xscrollcommand=self.viz_scrollbar_h.set)
        
        self.viz_canvas.pack(side="top", fill="both", expand=True)
        self.viz_scrollbar_h.pack(side="bottom", fill="x")
        
        # Bind mousewheel horizontal
        self.viz_canvas.bind("<Shift-MouseWheel>", self._on_horizontal_mousewheel)
        self.viz_canvas.bind("<Button-4>", lambda e: self.viz_canvas.xview_scroll(-1, "units"))
        self.viz_canvas.bind("<Button-5>", lambda e: self.viz_canvas.xview_scroll(1, "units"))
        
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
        """Créer les graphiques de visualisation optimisés en 2x2 avec scroll horizontal"""
        try:
            # Nettoyer la section de visualisation
            for widget in self.viz_scrollable_frame.winfo_children():
                widget.destroy()
            
            # Vérifier qu'on a assez de données
            if len(self.y_test) < 3:
                warning_label = tk.Label(
                    self.viz_scrollable_frame,
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
            
            # Créer la figure avec 2 graphiques principaux, plus grands et plus lisibles
            fig = plt.figure(figsize=(16, 8))  # Plus large pour avoir 2 graphiques côte à côte
            fig.patch.set_facecolor('white')
            fig.suptitle('🧠 Analyse de Performance du Réseau de Neurones', fontsize=16, fontweight='bold', y=0.95)
            
            # === GRAPHIQUE 1: Performance de Prédiction (gauche, plus grand) ===
            ax1 = plt.subplot(1, 2, 1)  # 1 ligne, 2 colonnes, position 1
            
            # Calculer les erreurs pour la colorisation
            absolute_errors = [abs(real - pred) for real, pred in zip(y_test_display, y_pred_display)]
            
            # Nuage de points avec colorisation par erreur, points plus gros
            scatter = ax1.scatter(y_test_display, y_pred_display, c=absolute_errors, 
                                 alpha=0.8, s=80, cmap='RdYlGn_r',  # Points plus gros
                                 edgecolors='black', linewidth=0.5)
            
            # Ligne parfaite (y=x) plus visible
            min_val = min(min(y_test_display), min(y_pred_display))
            max_val = max(max(y_test_display), max(y_pred_display))
            perfect_line = [min_val, max_val]
            ax1.plot(perfect_line, perfect_line, 'k--', alpha=0.9, linewidth=2.5, label='Prédiction Parfaite')
            
            # Métriques importantes
            from neural_network.utils import r2_score, mean_absolute_error
            r2 = r2_score(y_test_display, y_pred_display)
            mae = mean_absolute_error(y_test_display, y_pred_display)
            
            # Colorbar pour les erreurs
            cbar = plt.colorbar(scatter, ax=ax1, shrink=0.8)
            cbar.set_label('Erreur Absolue (€)', fontsize=11)
            
            ax1.set_xlabel("Valeurs Réelles (€)", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Prédictions (€)", fontsize=12, fontweight='bold')
            ax1.set_title(f"Performance de Prédiction\nR² = {r2:.3f} | MAE = {mae:,.0f}€", fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
            ax1.legend(fontsize=11, loc='upper left')
            ax1.tick_params(labelsize=11)
            
            # Formatage des axes avec des milliers
            ax1.ticklabel_format(style='plain', axis='both')
            
            # === GRAPHIQUE 2: Historique de l'Entraînement (droite, plus grand) ===
            ax2 = plt.subplot(1, 2, 2)  # 1 ligne, 2 colonnes, position 2
            
            
            if hasattr(self, 'losses') and self.losses:
                losses = self.losses
                epochs = list(range(1, len(losses) + 1))
                
                # Graphique principal de la loss
                ax2.plot(epochs, losses, 'b-', linewidth=2.5, alpha=0.9, label='Loss', color='#2E86AB')
                ax2.fill_between(epochs, losses, alpha=0.3, color='#2E86AB')
                
                # Ligne de tendance lissée avec couleur différente
                if len(losses) > 10:
                    window_size = max(3, len(losses) // 10)
                    smoothed_losses = []
                    for i in range(len(losses)):
                        start = max(0, i - window_size // 2)
                        end = min(len(losses), i + window_size // 2 + 1)
                        smoothed_losses.append(sum(losses[start:end]) / (end - start))
                    ax2.plot(epochs, smoothed_losses, '--', linewidth=2.5, alpha=0.9, label='Tendance', color='#F18F01')
                
                # Marqueurs pour début et fin
                ax2.plot(1, losses[0], 'ro', markersize=8, label=f'Début: {losses[0]:.4f}')
                ax2.plot(len(losses), losses[-1], 'go', markersize=8, label=f'Fin: {losses[-1]:.4f}')
                
                ax2.set_xlabel("Époques d'Entraînement", fontsize=12, fontweight='bold')
                ax2.set_ylabel("Loss (Erreur)", fontsize=12, fontweight='bold')
                ax2.set_title(f"Convergence de l'Entraînement\nAmélioration: {((losses[0]-losses[-1])/losses[0]*100):.1f}%", 
                             fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
                ax2.legend(fontsize=11, loc='upper right')
                ax2.set_yscale('log')
                ax2.tick_params(labelsize=11)
                
                # Annotations importantes
                if len(losses) > 1:
                    improvement = ((losses[0] - losses[-1]) / losses[0]) * 100
                    if improvement > 50:
                        ax2.annotate(f'Excellent\napprentissage!', 
                                   xy=(len(losses)*0.7, losses[-1]*2), 
                                   fontsize=10, fontweight='bold', 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
            else:
                ax2.text(0.5, 0.5, '⚠️ Historique de Loss\nnon disponible', 
                        ha='center', va='center', transform=ax2.transAxes, 
                        fontsize=14, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"))
                ax2.set_title("Convergence de l'Entraînement", fontsize=14, fontweight='bold')
            
            # Ajustement final de l'espacement pour 2 graphiques
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, wspace=0.3)  # Plus d'espace horizontal
            
            # Intégrer dans l'interface avec scroll
            canvas = FigureCanvasTkAgg(fig, master=self.viz_scrollable_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(side=tk.TOP, padx=20, pady=10)  # Graphiques en haut
            
            # === SECTION STATISTIQUES DÉTAILLÉE ET ATTRACTIVE ===
            # Frame principal pour les statistiques
            stats_main_frame = tk.Frame(self.viz_scrollable_frame, bg=self.controller.colors["bg_white"])
            stats_main_frame.pack(side=tk.TOP, fill=tk.X, padx=20, pady=10)
            
            # Titre de la section
            stats_title = tk.Label(
                stats_main_frame,
                text="📊 RAPPORT DE PERFORMANCE DÉTAILLÉ",
                font=('Helvetica', 14, 'bold'),
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["primary"]
            )
            stats_title.pack(pady=(10, 15))
            
            # Container pour 3 colonnes de métriques
            metrics_container = tk.Frame(stats_main_frame, bg=self.controller.colors["bg_white"])
            metrics_container.pack(fill=tk.X, pady=10)
            
            # Calculer des métriques avancées
            mse = mean_squared_error(y_test_display, y_pred_display)
            rmse = mse ** 0.5
            mape = sum(abs((real - pred) / real) for real, pred in zip(y_test_display, y_pred_display) if real != 0) / len(y_test_display) * 100
            
            # Pourcentage de prédictions dans différentes tolérances
            tolerance_5 = sum(1 for real, pred in zip(y_test_display, y_pred_display) if abs((pred - real) / real) <= 0.05) / len(y_test_display) * 100
            tolerance_10 = sum(1 for real, pred in zip(y_test_display, y_pred_display) if abs((pred - real) / real) <= 0.10) / len(y_test_display) * 100
            tolerance_20 = sum(1 for real, pred in zip(y_test_display, y_pred_display) if abs((pred - real) / real) <= 0.20) / len(y_test_display) * 100
            
            # COLONNE 1: Métriques de base
            col1_frame = tk.LabelFrame(
                metrics_container,
                text="🎯 PRÉCISION GÉNÉRALE",
                font=('Helvetica', 11, 'bold'),
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["primary"],
                padx=15, pady=10
            )
            col1_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, expand=True)
            
            # Déterminer la qualité du R²
            r2_quality = "Excellent" if r2 > 0.9 else "Bon" if r2 > 0.7 else "Moyen" if r2 > 0.5 else "Faible"
            r2_color = "#27ae60" if r2 > 0.9 else "#f39c12" if r2 > 0.7 else "#e67e22" if r2 > 0.5 else "#e74c3c"
            
            col1_text = f"""• R² Score: {r2:.4f} ({r2_quality})
• Erreur Absolue Moyenne: {mae:,.0f} €
• Erreur Quadratique: {rmse:,.0f} €
• Erreur Relative Moyenne: {mape:.1f}%

💡 Interprétation:
Le modèle explique {r2*100:.1f}% de la 
variance des prix."""
            
            tk.Label(col1_frame, text=col1_text, bg=self.controller.colors["bg_white"],
                    fg=self.controller.colors["text"], font=('Helvetica', 10), justify=tk.LEFT).pack(anchor="w")
            
            # COLONNE 2: Qualité des prédictions
            col2_frame = tk.LabelFrame(
                metrics_container,
                text="✅ NIVEAUX DE TOLÉRANCE",
                font=('Helvetica', 11, 'bold'),
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["success"],
                padx=15, pady=10
            )
            col2_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, expand=True)
            
            # Évaluation qualitative
            overall_quality = "Excellent" if tolerance_10 >= 80 else "Bon" if tolerance_10 >= 60 else "Moyen" if tolerance_10 >= 40 else "À améliorer"
            
            col2_text = f"""• Prédictions ±5%: {tolerance_5:.0f}%
• Prédictions ±10%: {tolerance_10:.0f}%
• Prédictions ±20%: {tolerance_20:.0f}%

🏆 Qualité globale: {overall_quality}

� Pour l'immobilier:
±10% est considéré comme 
une excellente précision."""
            
            tk.Label(col2_frame, text=col2_text, bg=self.controller.colors["bg_white"],
                    fg=self.controller.colors["text"], font=('Helvetica', 10), justify=tk.LEFT).pack(anchor="w")
            
            # COLONNE 3: Informations sur les données
            col3_frame = tk.LabelFrame(
                metrics_container,
                text="�📊 DONNÉES D'ENTRAÎNEMENT",
                font=('Helvetica', 11, 'bold'),
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["accent"],
                padx=15, pady=10
            )
            col3_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, expand=True)
            
            # Calculs supplémentaires
            total_samples = len(self.X_train) + len(y_test_display)
            test_ratio = len(y_test_display) / total_samples * 100
            
            # Info sur la convergence
            convergence_info = ""
            if hasattr(self, 'losses') and self.losses:
                improvement = ((self.losses[0] - self.losses[-1]) / self.losses[0]) * 100
                convergence_info = f"• Amélioration Loss: {improvement:.1f}%"
            
            col3_text = f"""• Échantillons totaux: {total_samples:,}
• Données d'entraînement: {len(self.X_train):,}
• Données de test: {len(y_test_display):,}
• Ratio test: {test_ratio:.1f}%
{convergence_info}

🧠 Architecture:
{len(self.model.layers)} couches
{sum(len(layer.neurons) for layer in self.model.layers):,} neurones"""
            
            tk.Label(col3_frame, text=col3_text, bg=self.controller.colors["bg_white"],
                    fg=self.controller.colors["text"], font=('Helvetica', 10), justify=tk.LEFT).pack(anchor="w")
            
        except Exception as e:
            error_label = tk.Label(
                self.viz_scrollable_frame,
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
