import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time

class ModelTrainingPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.model_info = None
        self.df = None
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.training_progress = 0
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
            text="Configuration du modèle",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        config_section.pack(fill=tk.X, pady=(0, 20))
        
        # Type de régression
        tk.Label(
            config_section,
            text="Type de régression:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(10, 5))
        
        self.regression_type_var = tk.StringVar(value="linear")
        regression_frame = tk.Frame(config_section, bg=self.controller.colors["bg_white"])
        regression_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Radiobutton(
            regression_frame,
            text="Régression linéaire simple",
            variable=self.regression_type_var,
            value="linear"
        ).pack(anchor="w")
        
        ttk.Radiobutton(
            regression_frame,
            text="Régression Ridge (avec régularisation)",
            variable=self.regression_type_var,
            value="ridge"
        ).pack(anchor="w")
        
        ttk.Radiobutton(
            regression_frame,
            text="Régression Lasso (sélection de variables)",
            variable=self.regression_type_var,
            value="lasso"
        ).pack(anchor="w")
        
        # Paramètres d'entraînement
        tk.Label(
            config_section,
            text="Paramètres d'entraînement:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(15, 5))
        
        params_frame = tk.Frame(config_section, bg=self.controller.colors["bg_white"])
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Test size
        tk.Label(
            params_frame,
            text="Taille du jeu de test (%):",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.test_size_var = tk.DoubleVar(value=0.2)
        test_size_scale = tk.Scale(
            params_frame,
            from_=0.1,
            to=0.5,
            resolution=0.05,
            orient=tk.HORIZONTAL,
            variable=self.test_size_var,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            length=200
        )
        test_size_scale.grid(row=0, column=1, sticky="w")
        
        # Random state
        tk.Label(
            params_frame,
            text="Graine aléatoire:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        
        self.random_state_var = tk.IntVar(value=42)
        random_state_entry = tk.Entry(
            params_frame,
            textvariable=self.random_state_var,
            width=10,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"]
        )
        random_state_entry.grid(row=1, column=1, sticky="w", pady=(10, 0))
        
        # Section 2: Progression de l'entraînement
        self.progress_section = tk.LabelFrame(
            self.scrollable_content,
            text="Progression de l'entraînement",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.progress_section.pack(fill=tk.X, pady=(0, 20))
        
        self.progress_bar = ttk.Progressbar(
            self.progress_section,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(pady=(10, 5))
        
        self.progress_label = tk.Label(
            self.progress_section,
            text="Prêt à commencer l'entraînement",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        )
        self.progress_label.pack(pady=(0, 10))
        
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
            cursor="hand2"
        )
        
    def load_training_data(self, file_path, model_info):
        """Charger les données pour l'entraînement"""
        try:
            self.df = pd.read_csv(file_path, low_memory=False)
            self.model_info = model_info
            
            self.title_label.config(text=f"Entraînement - {model_info['name']}")
            
            # Afficher les informations du modèle
            info_text = f"""Configuration du modèle:
🎯 Variable cible: {model_info['target_column']}
📊 Variables explicatives: {len(model_info['feature_columns'])} variables
📋 Données: {model_info['data_shape'][0]} lignes × {model_info['data_shape'][1]} colonnes
📁 Fichier: {model_info['data_file']}"""
            
            self.update_results(info_text)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les données:\n{str(e)}")
    
    def start_training(self):
        """Commencer l'entraînement du modèle"""
        if self.model_info is None:
            messagebox.showwarning("Attention", "Aucune donnée chargée")
            return
        
        self.train_button.config(state="disabled")
        self.progress_bar["value"] = 0
        
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
            
            X = clean_data[feature_cols]
            y = clean_data[target_col]
            
            self.update_progress(20, f"Données nettoyées: {len(clean_data)} échantillons valides...")
            
            # Étape 2: Division train/test
            self.update_progress(25, "Division des données...")
            
            test_size = self.test_size_var.get()
            random_state = self.random_state_var.get()
            
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Étape 3: Normalisation
            self.update_progress(40, "Normalisation des données...")
            
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            
            # Étape 4: Entraînement du modèle
            self.update_progress(60, "Entraînement du modèle...")
            
            regression_type = self.regression_type_var.get()
            
            if regression_type == "linear":
                self.model = LinearRegression()
            elif regression_type == "ridge":
                self.model = Ridge(alpha=1.0)
            elif regression_type == "lasso":
                self.model = Lasso(alpha=1.0)
            
            self.model.fit(self.X_train_scaled, self.y_train)
            
            # Étape 5: Évaluation
            self.update_progress(80, "Évaluation du modèle...")
            
            y_train_pred = self.model.predict(self.X_train_scaled)
            y_test_pred = self.model.predict(self.X_test_scaled)
            
            # Calculer les métriques
            train_r2 = r2_score(self.y_train, y_train_pred)
            test_r2 = r2_score(self.y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            train_mae = mean_absolute_error(self.y_train, y_train_pred)
            test_mae = mean_absolute_error(self.y_test, y_test_pred)
            
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
                recommendations.append("• Essayez d'ajouter plus de variables explicatives")
                recommendations.append("• Vérifiez la qualité des données (valeurs aberrantes)")
            
            if abs(train_r2 - test_r2) > 0.2:
                diagnostics.append("• Écart important train/test : possible surapprentissage")
                recommendations.append("• Augmentez la taille du jeu de test")
                recommendations.append("• Essayez la régularisation (Ridge/Lasso)")
            
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

🔍 DIAGNOSTIC:
═══════════════════════════════════
{diagnostic_text}

💡 RECOMMANDATIONS:
═══════════════════════════════════
{recommendation_text}

📋 INFORMATIONS TECHNIQUES:
═══════════════════════════════════
• Type de régression: {regression_type.title()}
• Données d'entraînement: {len(self.X_train)} échantillons
• Données de test: {len(self.X_test)} échantillons
• Variables utilisées: {len(feature_cols)}"""
            
            # Mettre à jour l'interface dans le thread principal
            self.after(0, lambda: self.training_completed(results, y_test_pred))
            
        except Exception as e:
            error_message = str(e)
            self.after(0, lambda: self.training_failed(error_message))
    
    def update_progress(self, value, text):
        """Mettre à jour la barre de progression"""
        def update():
            self.progress_bar["value"] = value
            self.progress_label.config(text=text)
        
        self.after(0, update)
        time.sleep(0.1)  # Petit délai pour visualiser la progression
    
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
        self.model_info["scaler"] = self.scaler
        
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
        """Créer les graphiques de visualisation"""
        try:
            # Nettoyer la section de visualisation
            for widget in self.viz_section.winfo_children():
                widget.destroy()
            
            # Créer la figure matplotlib
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            fig.patch.set_facecolor('white')
            
            # Graphique 1: Valeurs réelles vs prédites
            ax1.scatter(self.y_test, y_pred, alpha=0.6, color=self.controller.colors["primary"])
            ax1.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
            ax1.set_xlabel("Valeurs réelles")
            ax1.set_ylabel("Valeurs prédites")
            ax1.set_title("Prédictions vs Réalité")
            ax1.grid(True, alpha=0.3)
            
            # Graphique 2: Résidus
            residuals = self.y_test - y_pred
            ax2.scatter(y_pred, residuals, alpha=0.6, color=self.controller.colors["accent"])
            ax2.axhline(y=0, color='r', linestyle='--')
            ax2.set_xlabel("Valeurs prédites")
            ax2.set_ylabel("Résidus")
            ax2.set_title("Analyse des résidus")
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Intégrer dans l'interface
            canvas = FigureCanvasTkAgg(fig, master=self.viz_section)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
            
        except Exception as e:
            error_label = tk.Label(
                self.viz_section,
                text=f"Erreur lors de la création des graphiques: {str(e)}",
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["accent"],
                font=('Helvetica', 10)
            )
            error_label.pack(pady=10)
    
    def save_model(self):
        """Sauvegarder le modèle entraîné"""
        try:
            # Ajouter la date de création
            from datetime import datetime
            self.model_info['created_date'] = datetime.now().strftime("%d/%m/%Y à %H:%M")
            
            # Ajouter le modèle à la liste
            if self.model_info not in self.controller.model_list:
                self.controller.model_list.append(self.model_info)
            
            self.controller.status_label.config(
                text=f"Modèle '{self.model_info['name']}' entraîné et sauvegardé avec succès"
            )
            
            messagebox.showinfo(
                "Succès",
                f"Modèle '{self.model_info['name']}' sauvegardé avec succès!\n\n"
                f"Précision: {self.model_info['accuracy']}"
            )
            
            # Retourner à la page d'accueil
            self.controller.show_page("HomePage")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder le modèle:\n{str(e)}")
