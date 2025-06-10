import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ModelDetailsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.model_info = None
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
            command=lambda: self.controller.show_page("HomePage"),
            style='TButton'
        ).pack(side=tk.LEFT)
        
        self.title_label = tk.Label(
            main_frame,
            text="Détails du modèle",
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
        
        self.create_details_content()
        
    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
    def create_details_content(self):
        # Section 1: Informations générales
        self.info_section = tk.LabelFrame(
            self.scrollable_content,
            text="Informations générales",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.info_section.pack(fill=tk.X, pady=(0, 20))
        
        self.info_text = tk.Text(
            self.info_section,
            height=8,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11),
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.info_text.pack(fill=tk.X, pady=(10, 10))
        
        # Section 2: Variables et configuration
        self.config_section = tk.LabelFrame(
            self.scrollable_content,
            text="Configuration du modèle",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.config_section.pack(fill=tk.X, pady=(0, 20))
        
        self.config_text = tk.Text(
            self.config_section,
            height=6,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Courier', 10),
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.config_text.pack(fill=tk.X, pady=(10, 10))
        
        # Section 3: Métriques de performance
        self.metrics_section = tk.LabelFrame(
            self.scrollable_content,
            text="Métriques de performance",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.metrics_section.pack(fill=tk.X, pady=(0, 20))
        
        self.metrics_text = tk.Text(
            self.metrics_section,
            height=8,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Courier', 11),
            state=tk.DISABLED
        )
        self.metrics_text.pack(fill=tk.X, pady=(10, 10))
        
        # Section 4: Actions
        self.actions_section = tk.LabelFrame(
            self.scrollable_content,
            text="Actions",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.actions_section.pack(fill=tk.X, pady=(0, 20))
        
        actions_frame = tk.Frame(self.actions_section, bg=self.controller.colors["bg_white"])
        actions_frame.pack(fill=tk.X, pady=10)
        
        # Bouton Prédiction
        self.predict_button = ttk.Button(
            actions_frame,
            text="🔮 Faire une prédiction",
            command=self.make_prediction,
            style='Add.TButton',
            cursor="hand2"
        )
        self.predict_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Bouton Réentraîner
        self.retrain_button = ttk.Button(
            actions_frame,
            text="🔄 Réentraîner le modèle",
            command=self.retrain_model,
            style='TButton',
            cursor="hand2"
        )
        self.retrain_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Bouton Supprimer
        self.delete_button = ttk.Button(
            actions_frame,
            text="🗑️ Supprimer le modèle",
            command=self.delete_model,
            style='TButton',
            cursor="hand2"
        )
        self.delete_button.pack(side=tk.LEFT)
        
        # Section 5: Zone de prédiction
        self.prediction_section = tk.LabelFrame(
            self.scrollable_content,
            text="Zone de prédiction",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.prediction_section.pack(fill=tk.X, pady=(0, 20))
        
        # Cette section sera remplie dynamiquement
        
    def load_model_details(self, model_info):
        """Charger les détails d'un modèle"""
        self.model_info = model_info
        
        # Mettre à jour le titre
        self.title_label.config(text=f"Détails - {model_info['name']}")
        
        # Informations générales
        info_text = f"""📋 Nom: {model_info['name']}
📊 Type: Modèle de régression
🎯 Précision: {model_info.get('accuracy', 'Non disponible')}
📁 Fichier de données: {model_info['data_file']}
📅 Créé: {model_info.get('created_date', 'Date inconnue')}
📝 Description: {model_info.get('description', 'Aucune description')}

🏆 STATUT: {'✅ Modèle entraîné et prêt' if 'model' in model_info else '⚠️ Modèle non entraîné'}"""
        
        self.update_text_widget(self.info_text, info_text)
        
        # Configuration
        if 'target_column' in model_info:
            config_text = f"""🎯 VARIABLE CIBLE:
    • {model_info['target_column']}

📊 VARIABLES EXPLICATIVES ({len(model_info.get('feature_columns', []))})"""
            
            if 'feature_columns' in model_info:
                config_text += ":\n"
                for i, feature in enumerate(model_info['feature_columns'], 1):
                    config_text += f"    {i:2d}. {feature}\n"
            
            if 'data_shape' in model_info:
                rows, cols = model_info['data_shape']
                config_text += f"\n📋 DONNÉES:\n    • {rows:,} lignes × {cols} colonnes"
                
        else:
            config_text = "⚠️ Configuration non disponible - modèle non configuré"
            
        self.update_text_widget(self.config_text, config_text)
        
        # Métriques
        if 'model' in model_info:
            # Si le modèle est entraîné, afficher les métriques détaillées
            metrics_text = f"""🎯 PERFORMANCE DU MODÈLE:
═══════════════════════════════════

✅ Modèle entraîné avec succès
🎯 Précision globale: {model_info['accuracy']}

📊 Type de régression: {model_info.get('regression_type', 'Linéaire').title()}

⚡ UTILISATION:
  • Modèle prêt pour les prédictions
  • Variables normalisées automatiquement
  • Résultats en temps réel

🔧 PARAMÈTRES:
  • Algorithme: {model_info.get('regression_type', 'linear').title()}
  • Variables: {len(model_info.get('feature_columns', []))}
  • Échantillons: {model_info.get('data_shape', [0])[0]:,}"""
        else:
            metrics_text = """⚠️ MODÈLE NON ENTRAÎNÉ

Le modèle n'a pas encore été entraîné.
Utilisez le bouton 'Réentraîner' pour commencer l'entraînement."""
            
        self.update_text_widget(self.metrics_text, metrics_text)
        
        # Zone de prédiction
        self.create_prediction_form()
        
    def update_text_widget(self, widget, text):
        """Mettre à jour un widget texte"""
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(1.0, text)
        widget.config(state=tk.DISABLED)
        
    def create_prediction_form(self):
        """Créer le formulaire de prédiction"""
        # Nettoyer la section
        for widget in self.prediction_section.winfo_children():
            widget.destroy()
            
        if 'feature_columns' not in self.model_info or 'model' not in self.model_info:
            tk.Label(
                self.prediction_section,
                text="⚠️ Prédiction non disponible - modèle non entraîné",
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text"],
                font=('Helvetica', 11)
            ).pack(pady=10)
            return
            
        tk.Label(
            self.prediction_section,
            text="Entrez les valeurs pour faire une prédiction:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11, 'bold')
        ).pack(anchor="w", pady=(10, 5))
        
        # Créer les champs de saisie
        self.prediction_vars = {}
        
        input_frame = tk.Frame(self.prediction_section, bg=self.controller.colors["bg_white"])
        input_frame.pack(fill=tk.X, pady=5)
        
        for i, feature in enumerate(self.model_info['feature_columns']):
            row = i // 2
            col = i % 2
            
            feature_frame = tk.Frame(input_frame, bg=self.controller.colors["bg_white"])
            feature_frame.grid(row=row, column=col, sticky="ew", padx=5, pady=2)
            
            tk.Label(
                feature_frame,
                text=f"{feature}:",
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text"],
                font=('Helvetica', 10),
                width=15
            ).pack(side=tk.LEFT)
            
            var = tk.StringVar()
            entry = tk.Entry(
                feature_frame,
                textvariable=var,
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text"],
                width=15
            )
            entry.pack(side=tk.LEFT, padx=(5, 0))
            
            self.prediction_vars[feature] = var
            
        # Configurer les colonnes pour qu'elles s'étendent
        input_frame.grid_columnconfigure(0, weight=1)
        input_frame.grid_columnconfigure(1, weight=1)
        
        # Bouton prédire et résultat
        action_frame = tk.Frame(self.prediction_section, bg=self.controller.colors["bg_white"])
        action_frame.pack(fill=tk.X, pady=(15, 5))
        
        ttk.Button(
            action_frame,
            text="🎯 Prédire",
            command=self.calculate_prediction,
            style='Add.TButton'
        ).pack(side=tk.LEFT)
        
        self.prediction_result = tk.Label(
            action_frame,
            text="",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["primary"],
            font=('Helvetica', 12, 'bold')
        )
        self.prediction_result.pack(side=tk.LEFT, padx=(20, 0))
        
    def calculate_prediction(self):
        """Calculer une prédiction"""
        try:
            if 'model' not in self.model_info or 'scaler' not in self.model_info:
                messagebox.showwarning("Attention", "Modèle non entraîné")
                return
                
            # Récupérer les valeurs
            values = []
            for feature in self.model_info['feature_columns']:
                value_str = self.prediction_vars[feature].get().strip()
                if not value_str:
                    messagebox.showwarning("Attention", f"Veuillez entrer une valeur pour {feature}")
                    return
                try:
                    value = float(value_str)
                    values.append(value)
                except ValueError:
                    messagebox.showwarning("Attention", f"Valeur invalide pour {feature}: {value_str}")
                    return
            
            # Normaliser et prédire
            values_array = np.array(values).reshape(1, -1)
            values_scaled = self.model_info['scaler'].transform(values_array)
            prediction = self.model_info['model'].predict(values_scaled)[0]
            
            # Afficher le résultat
            self.prediction_result.config(
                text=f"🎯 Prédiction: {prediction:.2f}",
                fg=self.controller.colors["success"]
            )
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la prédiction:\n{str(e)}")
            
    def make_prediction(self):
        """Faire une prédiction simple"""
        if 'model' not in self.model_info:
            messagebox.showinfo("Information", "Le modèle doit être entraîné avant de pouvoir faire des prédictions.")
            return
            
        # Scroller vers la zone de prédiction
        self.main_canvas.yview_moveto(1.0)
        
    def retrain_model(self):
        """Réentraîner le modèle"""
        if 'full_path' in self.model_info:
            self.controller.show_model_training(self.model_info['full_path'], self.model_info)
        else:
            messagebox.showwarning("Attention", "Impossible de réentraîner: fichier de données non trouvé")
            
    def delete_model(self):
        """Supprimer le modèle"""
        result = messagebox.askyesno(
            "Confirmation",
            f"Êtes-vous sûr de vouloir supprimer le modèle '{self.model_info['name']}'?\n\nCette action est irréversible."
        )
        
        if result:
            # Supprimer de la liste
            if self.model_info in self.controller.model_list:
                self.controller.model_list.remove(self.model_info)
                
            self.controller.status_label.config(
                text=f"Modèle '{self.model_info['name']}' supprimé"
            )
            
            messagebox.showinfo("Succès", f"Modèle '{self.model_info['name']}' supprimé avec succès")
            
            # Retourner à la page d'accueil
            self.controller.show_page("HomePage")
