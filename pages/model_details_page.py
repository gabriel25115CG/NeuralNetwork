import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import os

# Ajouter le chemin vers neural_network
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_network.model_persistence import ModelPersistence

class ModelDetailsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.model_info = None
        self.loaded_model_data = None
        self.neural_network = None
        self.normalization_params = {}
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
          # Container principal avec scroll horizontal et vertical
        canvas_container = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        # Canvas principal pour le scrolling
        self.main_canvas = tk.Canvas(canvas_container, bg=self.controller.colors["bg_light"])
        
        # Scrollbars verticale et horizontale
        v_scrollbar = ttk.Scrollbar(canvas_container, orient="vertical", command=self.main_canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_container, orient="horizontal", command=self.main_canvas.xview)
        
        # Frame pour le contenu scrollable
        self.scrollable_content = tk.Frame(self.main_canvas, bg=self.controller.colors["bg_light"])
        
        # Configuration du scrolling
        self.scrollable_content.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        # Créer la fenêtre scrollable
        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_content, anchor="nw")
        
        # Configurer les scrollbars
        self.main_canvas.configure(
            yscrollcommand=v_scrollbar.set,
            xscrollcommand=h_scrollbar.set
        )
        
        # Placement avec grid pour mieux gérer les scrollbars
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        # Configuration du redimensionnement
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)
          # Bind events pour le scrolling
        self.main_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.main_canvas.bind("<Shift-MouseWheel>", self._on_horizontal_mousewheel)
        self.bind_all("<MouseWheel>", self._on_mousewheel)
        self.bind_all("<Shift-MouseWheel>", self._on_horizontal_mousewheel)
        
        # Ajuster la largeur du contenu quand le canvas change de taille
        def on_canvas_configure(event):
            # S'assurer que le contenu fait au moins la largeur du canvas
            canvas_width = event.width
            self.main_canvas.itemconfig(self.canvas_window, width=max(canvas_width, 1200))  # Largeur minimale réduite pour les graphiques
        
        self.main_canvas.bind('<Configure>', on_canvas_configure)
        
        # Instructions de navigation
        nav_info = tk.Label(
            main_frame,
            text="💡 Navigation: Molette souris = Vertical | Shift+Molette = Horizontal | Barres de défilement disponibles",
            font=('Helvetica', 9),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["text"],
            justify="center"
        )
        nav_info.pack(pady=(5, 0))
        
        # Créer les sections
        self.create_details_content()
        
        # Créer une section pour les graphiques
        self.charts_frame = tk.Frame(self.scrollable_content, bg=self.controller.colors["bg_white"])
        self.charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def _on_mousewheel(self, event):
        self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def _on_horizontal_mousewheel(self, event):
        self.main_canvas.xview_scroll(int(-1*(event.delta/120)), "units")
    
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
        
        self.info_text = self._create_scrollable_text(self.info_section, height=10)
        
        # Section 2: Configuration
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
        
        self.config_text = self._create_scrollable_text(self.config_section, height=8)
        
        # Section 3: Métriques de performance
        self.metrics_section = tk.LabelFrame(
            self.scrollable_content,
            text="Performance et métriques",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        self.metrics_section.pack(fill=tk.X, pady=(0, 20))
        
        self.metrics_text = self._create_scrollable_text(self.metrics_section, height=8)
        
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
        
        # Boutons d'action
        action_frame = tk.Frame(self.actions_section, bg=self.controller.colors["bg_white"])
        action_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            action_frame,
            text="🎯 Faire une prédiction",
            command=self.make_prediction,
            style='Add.TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            action_frame,
            text="🔄 Réentraîner",
            command=self.retrain_model,
            style='TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            action_frame,
            text="🗑️ Supprimer",
            command=self.delete_model,
            style='TButton'
        ).pack(side=tk.LEFT)
        
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

    def _create_scrollable_text(self, parent, height=6):
        """Créer un widget Text avec scrollbar intégrée"""
        # Frame conteneur pour le text et la scrollbar
        text_frame = tk.Frame(parent, bg=self.controller.colors["bg_white"])
        text_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 10))
        
        # Widget Text
        text_widget = tk.Text(
            text_frame,
            height=height,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11),
            state=tk.DISABLED,
            wrap=tk.WORD,
            relief=tk.FLAT,
            borderwidth=1
        )
        
        # Scrollbar verticale
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Placement
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return text_widget

    def load_model_details(self, loaded_model_data):
        """Charger et afficher les détails d'un modèle sauvegardé avec visualisations"""
        try:
            if isinstance(loaded_model_data, dict) and "model_info" in loaded_model_data:
                # Modèle chargé depuis un fichier sauvegardé
                model_info = loaded_model_data["model_info"]
                neural_network = loaded_model_data["neural_network"]
                normalization_params = loaded_model_data.get("normalization_params", {})
                  # Stocker les données du modèle chargé
                self.loaded_model_data = loaded_model_data
                self.model_info = model_info  # Les vraies infos du modèle sont ici
                self.neural_network = neural_network
                self.normalization_params = normalization_params
                
                # Debug: vérifier que les informations importantes sont présentes
                print(f"🔍 DEBUG: feature_columns = {model_info.get('feature_columns', 'Non trouvées')}")
                print(f"🔍 DEBUG: target_column = {model_info.get('target_column', 'Non trouvée')}")
                print(f"🔍 DEBUG: neural_network chargé = {neural_network is not None}")
                print(f"🔍 DEBUG: normalization_params chargés = {len(normalization_params)} paramètres")
                
                # Mettre à jour le titre
                self.title_label.config(text=f"📊 Détails - {model_info['name']}")
                  # Informations générales - utiliser l'architecture sauvegardée en priorité
                saved_architecture = loaded_model_data.get("network_architecture", {})
                if 'layers_config' in saved_architecture:
                    layers_config = saved_architecture['layers_config']
                    print(f"Architecture récupérée des métadonnées: {layers_config}")
                else:
                    layers_config = neural_network.layers_config
                    print(f"Architecture récupérée du neural_network: {layers_config}")
                
                info_text = f"""📋 Nom: {model_info['name']}
📊 Type: Réseau de Neurones - Régression
🎯 Variable cible: {model_info.get('target_column', 'Non définie')}
📁 Fichier source: {model_info.get('data_file', 'Non spécifié')}
📅 Créé le: {model_info.get('created_date', 'Date inconnue')}

🧠 ARCHITECTURE DU RÉSEAU:
   • Configuration: {layers_config}
   • Entrées: {layers_config[0]} variables
   • Couches cachées: {len(layers_config) - 2} ({layers_config[1:-1] if len(layers_config) > 2 else 'Aucune'})
   • Sortie: {layers_config[-1]} valeur(s)
   • Total paramètres: {neural_network.get_info().get('total_params', 'Non calculé')}

🏆 STATUT: ✅ Modèle entraîné et prêt à utiliser"""
                
                self.update_text_widget(self.info_text, info_text)
                
                # Configuration détaillée
                config_text = f"""🎯 VARIABLE CIBLE:
    • {model_info.get('target_column', 'Non définie')}

📊 VARIABLES EXPLICATIVES ({len(model_info.get('feature_columns', []))})"""
                
                if 'feature_columns' in model_info:
                    config_text += ":\n"
                    for i, feature in enumerate(model_info['feature_columns'], 1):
                        config_text += f"    {i:2d}. {feature}\n"
                
                if 'data_shape' in model_info:
                    rows, cols = model_info['data_shape']
                    config_text += f"\n📋 DONNÉES:\n    • {rows:,} lignes × {cols} colonnes"
                
                self.update_text_widget(self.config_text, config_text)
                  # Métriques de performance - utiliser les vraies métriques sauvegardées
                training_results = loaded_model_data.get("training_results", {})
                metrics = training_results.get("metrics", {})
                
                # Formater les métriques avec les vraies valeurs
                mse_value = metrics.get('mse', 'Non calculé')
                mae_value = metrics.get('mae', 'Non calculé')
                r2_value = metrics.get('r2', 'Non calculé')
                
                # Formater les valeurs numériques
                if isinstance(mse_value, (int, float)):
                    mse_str = f"{mse_value:.4f}"
                else:
                    mse_str = str(mse_value)
                    
                if isinstance(mae_value, (int, float)):
                    mae_str = f"{mae_value:.4f}"
                else:
                    mae_str = str(mae_value)
                    
                if isinstance(r2_value, (int, float)):
                    r2_str = f"{r2_value:.4f}"
                else:
                    r2_str = str(r2_value)
                
                metrics_text = f"""🎯 PERFORMANCE DU MODÈLE:
═══════════════════════════════════

✅ Modèle entraîné avec succès
📊 Architecture: Réseau de Neurones
🎯 MSE: {mse_str}
📈 MAE: {mae_str}
📊 R²: {r2_str}

⚡ PARAMÈTRES D'ENTRAÎNEMENT:
   • Couches: {len(layers_config)} ({layers_config})
   • Paramètres totaux: {neural_network.get_info().get('total_params', 'Non calculé')}
   • Normalisation: {'✅ Activée' if normalization_params else '⚠️ Non disponible'}"""
                
                self.update_text_widget(self.metrics_text, metrics_text)
                
                # Créer les visualisations
                self.create_model_visualizations(loaded_model_data)
                
            else:
                # Format de modèle ancien ou différent
                self.load_model_info(loaded_model_data)
                
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les détails du modèle:\n{str(e)}")
            
    def create_model_visualizations(self, loaded_model_data):
        """Créer les 3 graphiques demandés pour le modèle chargé"""
        try:
            # Nettoyer les anciens graphiques
            for widget in self.charts_frame.winfo_children():
                widget.destroy()
            
            model_info = loaded_model_data["model_info"]
            neural_network = loaded_model_data["neural_network"]
            training_results = loaded_model_data.get("training_results", {})            # Créer une figure avec 3 subplots optimisée et plus compacte
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Style général pour tous les graphiques
            plt.style.use('default')  # Reset pour éviter les conflits
            fig.patch.set_facecolor('white')
            
            # Titre principal avec style amélioré mais plus compact
            fig.suptitle(f'📊 Analyse du Modèle: {model_info["name"]}', 
                        fontsize=14, fontweight='bold', y=0.95, 
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.3))
              # Ajuster l'espacement pour une meilleure lisibilité en format plus compact
            plt.subplots_adjust(left=0.06, right=0.94, top=0.85, bottom=0.15, wspace=0.4)
            
            try:
                # Récupérer la vraie architecture sauvegardée
                saved_architecture = loaded_model_data.get("network_architecture", {})
                if 'layers_config' in saved_architecture:
                    correct_layers_config = saved_architecture['layers_config']
                    print(f"Utilisation de l'architecture des métadonnées pour le graphique: {correct_layers_config}")
                else:
                    correct_layers_config = neural_network.layers_config
                    print(f"Utilisation de l'architecture du neural_network pour le graphique: {correct_layers_config}")
                
                # 1. Architecture du réseau (gauche) - avec la vraie architecture
                self.plot_network_architecture_enhanced(ax1, neural_network, correct_layers_config)
                
                # 2. Courbe de Loss (centre)
                self.plot_training_loss_curve(ax2, training_results)
                  # 3. Prédictions vs Réelles (droite)
                self.plot_predictions_vs_actual(ax3, training_results, neural_network)
                
                # Améliorer l'apparence générale
                for ax in [ax1, ax2, ax3]:
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_linewidth(0.5)
                    ax.spines['bottom'].set_linewidth(0.5)
                
            except Exception as e:
                print(f"Erreur lors de la création des graphiques: {e}")
                # En cas d'erreur, créer des graphiques de base
                self.create_fallback_charts(ax1, ax2, ax3, model_info)
            
            # Optimiser le layout final
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            
            # Intégrer dans Tkinter avec scrolling
            canvas = FigureCanvasTkAgg(fig, self.charts_frame)
            canvas.draw()
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Ajouter une barre d'outils pour le zoom et la navigation
            toolbar_frame = tk.Frame(self.charts_frame, bg=self.controller.colors["bg_white"])
            toolbar_frame.pack(fill=tk.X, pady=(5, 0))
            
            from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()
              # Statut des données avec plus d'informations et actions
            has_real_data = 'loss_history' in training_results and 'y_test_real' in training_results
            has_partial_data = 'loss_history' in training_results or 'y_test_real' in training_results
            
            status_frame = tk.Frame(self.charts_frame, bg=self.controller.colors["bg_white"])
            status_frame.pack(fill=tk.X, pady=(10, 5))
            
            if has_real_data:
                status_text = "✅ Graphiques basés sur les données d'entraînement sauvegardées"
                status_color = self.controller.colors["success"]
                detail_text = f"📊 {len(training_results.get('loss_history', []))} époques • {len(training_results.get('y_test_real', []))} points de test"
            elif has_partial_data:
                status_text = "⚠️ Données partielles disponibles - Certains graphiques simulés"
                status_color = self.controller.colors["accent"]
                detail_text = "💡 Réentraînez le modèle pour obtenir des visualisations complètes"
            else:
                status_text = "ℹ️ Graphiques de démonstration - Données simulées"
                status_color = self.controller.colors["text"]
                detail_text = "🔄 Entraînez le modèle pour voir les vraies données"
            
            status_label = tk.Label(
                status_frame,
                text=status_text,
                bg=self.controller.colors["bg_white"],
                fg=status_color,
                font=('Helvetica', 11, 'bold')
            )
            status_label.pack()
            
            detail_label = tk.Label(
                status_frame,
                text=detail_text,
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text"],
                font=('Helvetica', 9)
            )
            detail_label.pack()
            
            # Boutons d'action rapide pour les graphiques
            action_frame = tk.Frame(status_frame, bg=self.controller.colors["bg_white"])
            action_frame.pack(pady=(5, 0))
            
            ttk.Button(
                action_frame,
                text="📊 Sauvegarder graphiques",
                command=lambda: self.save_charts(fig),
                style='TButton'
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            if not has_real_data:
                ttk.Button(
                    action_frame,
                    text="🔄 Réentraîner pour données réelles",
                    command=self.retrain_model,
                    style='Add.TButton'
                ).pack(side=tk.LEFT)
            
        except Exception as e:
            # En cas d'erreur, afficher un message informatif plus détaillé
            error_frame = tk.Frame(self.charts_frame, bg=self.controller.colors["bg_white"])
            error_frame.pack(expand=True, fill=tk.BOTH, pady=20)
            
            error_title = tk.Label(
                error_frame,
                text="⚠️ Erreur lors de la création des visualisations",
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["accent"],
                font=('Helvetica', 14, 'bold')
            )
            error_title.pack(pady=(20, 10))
            
            error_detail = tk.Label(
                error_frame,
                text=f"Détail de l'erreur: {str(e)}\n\n"
                     "Le modèle reste fonctionnel pour les prédictions.\n"
                     "Vous pouvez essayer de :\n"
                     "• Réentraîner le modèle\n"
                     "• Vérifier l'intégrité des données sauvegardées\n"
                     "• Contacter le support si le problème persiste",
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text"],
                font=('Helvetica', 11),
                justify=tk.CENTER,
                wraplength=600
            )
            error_detail.pack(pady=10)
            
            ttk.Button(
                error_frame,
                text="🔄 Réessayer",
                command=lambda: self.create_model_visualizations(loaded_model_data),
                style='Add.TButton'
            ).pack(pady=10)

    def create_fallback_charts(self, ax1, ax2, ax3, model_info):
        """Créer des graphiques de fallback en cas d'erreur"""
        # Graphique 1: Message d'erreur pour l'architecture
        ax1.text(0.5, 0.5, '⚠️ Erreur\narchitecture', ha='center', va='center', 
                transform=ax1.transAxes, fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        ax1.set_title("Architecture du Réseau", fontweight='bold')
        ax1.axis('off')
        
        # Graphique 2: Message d'erreur pour la loss
        ax2.text(0.5, 0.5, '⚠️ Erreur\ncourbe de loss', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        ax2.set_title("Courbe de Loss", fontweight='bold')
        ax2.axis('off')
        
        # Graphique 3: Message d'erreur pour les prédictions
        ax3.text(0.5, 0.5, '⚠️ Erreur\nprédictions', ha='center', va='center', 
                transform=ax3.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
        ax3.set_title("Prédictions vs Réelles", fontweight='bold')
        ax3.axis('off')

    def save_charts(self, fig):
        """Sauvegarder les graphiques en tant qu'image"""
        try:
            from tkinter import filedialog
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("SVG files", "*.svg")],
                title="Sauvegarder les graphiques"
            )
            if filename:
                fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
                self.controller.status_label.config(text=f"Graphiques sauvegardés: {filename}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder: {str(e)}")

    def plot_network_architecture_enhanced(self, ax, neural_network, layers_config=None):
        """Dessiner l'architecture du réseau de façon améliorée et centrée"""
        ax.clear()
        
        # Utiliser la configuration passée en paramètre ou celle du réseau
        if layers_config is None:
            layers_config = neural_network.layers_config
        
        print(f"Dessin du graphique avec l'architecture: {layers_config}")
          # Calculer les paramètres totaux pour le titre
        total_params = sum(layers_config[i] * layers_config[i+1] + layers_config[i+1] 
                          for i in range(len(layers_config)-1))
        arch_str = ' → '.join(map(str, layers_config))
        
        ax.set_title(f"Architecture du Réseau\n{arch_str} ({total_params:,} paramètres)", 
                    fontsize=10, fontweight='bold', pad=15)
        
        # Paramètres de dessin adaptatifs
        n_layers = len(layers_config)
        max_neurons = max(layers_config)
        
        # Adapter l'espacement selon le nombre de couches
        if n_layers <= 3:
            layer_spacing = 3.0
        elif n_layers <= 5:
            layer_spacing = 2.5
        else:
            layer_spacing = 2.0
            
        # Adapter la taille des neurones selon le nombre max
        if max_neurons <= 5:
            neuron_radius = 0.4
        elif max_neurons <= 10:
            neuron_radius = 0.3
        else:
            neuron_radius = 0.25
            
        # Couleurs dégradées pour les couches
        layer_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
        
        # Calculer les positions pour centrer le réseau
        total_width = (n_layers - 1) * layer_spacing
        start_x = -total_width / 2
        
        all_positions = []  # Stocker toutes les positions pour les connexions
        
        for i, n_neurons in enumerate(layers_config):
            x = start_x + i * layer_spacing
            
            # Calculer les positions Y centrées
            if n_neurons == 1:
                y_positions = [0]  # Centré
            else:
                # Espacement vertical adaptatif
                if max_neurons <= 8:
                    vertical_spacing = 1.0
                else:
                    vertical_spacing = min(1.0, 8.0 / max_neurons)
                
                total_height = (n_neurons - 1) * vertical_spacing
                y_positions = [j * vertical_spacing - total_height/2 for j in range(n_neurons)]
            
            all_positions.append([(x, y) for y in y_positions])
            
            # Couleur de la couche avec variation
            base_color = layer_colors[i % len(layer_colors)]
            
            # Dessiner les neurones
            for j, y in enumerate(y_positions):
                circle = plt.Circle((x, y), neuron_radius, color=base_color, alpha=0.8, 
                                  edgecolor='white', linewidth=1.5)
                ax.add_patch(circle)
                
                # Ajouter un petit numéro dans le neurone si pas trop de neurones
                if n_neurons <= 10:
                    ax.text(x, y, str(j+1), ha='center', va='center', 
                           fontsize=8, fontweight='bold', color='white')
            
            # Étiquette de la couche avec noms adaptatifs
            if i == 0:
                layer_name = "Entrée"
            elif i == len(layers_config) - 1:
                layer_name = "Sortie"
            else:
                layer_name = f"Cachée {i}"
                
            # Position de l'étiquette adaptée (en bas du graphique)
            label_y = min(y_positions) - neuron_radius - 0.7
            ax.text(x, label_y, f"{layer_name}\n({n_neurons})", 
                   ha='center', va='top', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=base_color, alpha=0.3))
        
        # Dessiner toutes les connexions entre couches adjacentes
        for i in range(len(all_positions) - 1):
            current_layer = all_positions[i]
            next_layer = all_positions[i + 1]
            
            # Dessiner les connexions avec gestion intelligente de la densité
            total_connections = len(current_layer) * len(next_layer)
            
            if total_connections <= 50:
                # Dessiner toutes les connexions
                alpha = 0.4
                linewidth = 0.8
                for x1, y1 in current_layer:
                    for x2, y2 in next_layer:
                        ax.plot([x1 + neuron_radius, x2 - neuron_radius], [y1, y2], 
                               'gray', alpha=alpha, linewidth=linewidth)
            elif total_connections <= 200:
                # Dessiner une sélection de connexions
                alpha = 0.25
                linewidth = 0.6
                step = max(1, len(current_layer) // 8)  # Prendre 1 neurone sur N
                for i_curr in range(0, len(current_layer), step):
                    x1, y1 = current_layer[i_curr]
                    for x2, y2 in next_layer:
                        ax.plot([x1 + neuron_radius, x2 - neuron_radius], [y1, y2], 
                               'gray', alpha=alpha, linewidth=linewidth)
            else:
                # Pour beaucoup de connexions, juste quelques lignes représentatives
                alpha = 0.15
                linewidth = 0.4
                # Connecter les neurones du haut, milieu, bas
                for i_curr in [0, len(current_layer)//2, len(current_layer)-1]:
                    if i_curr < len(current_layer):
                        x1, y1 = current_layer[i_curr]
                        for i_next in [0, len(next_layer)//2, len(next_layer)-1]:
                            if i_next < len(next_layer):
                                x2, y2 = next_layer[i_next]
                                ax.plot([x1 + neuron_radius, x2 - neuron_radius], [y1, y2], 
                                       'gray', alpha=alpha, linewidth=linewidth)
        
        # Définir les limites pour centrer parfaitement
        margin = 1.0
        ax.set_xlim(start_x - margin, start_x + total_width + margin)
          # Calculer les limites Y en tenant compte des étiquettes en bas
        max_y = max(max(y for _, y in layer_pos) for layer_pos in all_positions)
        min_y = min(min(y for _, y in layer_pos) for layer_pos in all_positions)
        
        # Ajouter de l'espace pour les étiquettes en bas
        y_margin_top = max(1.5, (max_y - min_y) * 0.2)
        y_margin_bottom = max(2.5, (max_y - min_y) * 0.4)  # Plus d'espace en bas pour les étiquettes
        ax.set_ylim(min_y - y_margin_bottom, max_y + y_margin_top)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def plot_training_loss_curve(self, ax, training_results):
        """Dessiner une courbe de loss d'entraînement belle et informative"""
        ax.clear()
        ax.set_title("📈 Évolution de la Loss d'Entraînement", fontsize=11, fontweight='bold', pad=15)
        
        if 'loss_history' in training_results and len(training_results['loss_history']) > 0:
            # Données réelles d'entraînement
            losses = training_results['loss_history']
            epochs = list(range(1, len(losses) + 1))
            
            print(f"🎯 Affichage de {len(losses)} époques de loss réelles")
            
            # Calculer les statistiques
            initial_loss = losses[0]
            final_loss = losses[-1]
            min_loss = min(losses)
            max_loss = max(losses)
            improvement = ((initial_loss - final_loss) / initial_loss) * 100 if initial_loss > 0 else 0
            
            # Couleur selon la performance
            if improvement > 90:
                color = '#27ae60'  # Vert foncé - Excellent
                performance = "🟢 Excellente"
            elif improvement > 75:
                color = '#2ecc71'  # Vert - Très bon
                performance = "🟢 Très bonne"
            elif improvement > 50:
                color = '#3498db'  # Bleu - Bon
                performance = "🔵 Bonne"
            elif improvement > 25:
                color = '#f39c12'  # Orange - Moyen
                performance = "🟡 Moyenne"
            else:
                color = '#e74c3c'  # Rouge - Faible
                performance = "🔴 Faible"
              # Graphique principal avec une belle courbe
            ax.plot(epochs, losses, color=color, linewidth=2.5, alpha=0.9, 
                   label=f'Loss d\'entraînement', marker='o', markersize=3, markevery=max(1, len(epochs)//20))
            
            # Points de référence importants
            ax.scatter([1], [initial_loss], color='red', s=100, zorder=5, 
                      label=f'Début: {initial_loss:.4f}', alpha=0.8, edgecolors='white', linewidth=1)
            ax.scatter([len(losses)], [final_loss], color='green', s=100, zorder=5, 
                      label=f'Fin: {final_loss:.4f}', alpha=0.8, edgecolors='white', linewidth=1)
            
            # Point de loss minimale si différent du point final
            min_loss_epoch = losses.index(min_loss) + 1
            if min_loss_epoch != len(losses):
                ax.scatter([min_loss_epoch], [min_loss], color='gold', s=80, zorder=5, 
                          label=f'Min: {min_loss:.4f}', alpha=0.8, edgecolors='white', linewidth=1)
            
            # Configuration des axes avec un style moderne
            ax.set_xlabel('Époque', fontsize=10, fontweight='bold')
            ax.set_ylabel('Loss (MSE)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#fafafa')
            
            # Améliorer l'échelle Y pour mieux voir la courbe
            y_margin = (max_loss - min_loss) * 0.1
            ax.set_ylim(min_loss - y_margin, max_loss + y_margin)
              # Légende moderne
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper right', 
                     framealpha=0.9, edgecolor='gray')
            
            # Boîte d'informations compacte
            info_text = f'''Convergence: {performance}
📊 Initial: {initial_loss:.6f}
🎯 Final: {final_loss:.6f}
📉 Amélioration: {improvement:.1f}%
⭐ Min: {min_loss:.6f}
📈 Époques: {len(losses)}'''
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes, va='top', ha='left',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                   alpha=0.95, edgecolor=color, linewidth=1.5))
            
            # Ajouter une annotation pour le taux de convergence si assez d'époques
            if len(losses) > 10:
                # Calculer la dérivée pour voir la vitesse de convergence
                mid_point = len(losses) // 2
                mid_loss = losses[mid_point]
                convergence_rate = (initial_loss - mid_loss) / mid_point if mid_point > 0 else 0
                
                if convergence_rate > 0:
                    ax.annotate(f'Convergence rapide\n({convergence_rate:.4f}/époque)', 
                               xy=(mid_point, mid_loss), 
                               xytext=(mid_point + len(losses)*0.3, mid_loss + (max_loss - min_loss)*0.2),
                               arrowprops=dict(arrowstyle='->', color=color, alpha=0.7, lw=1.5),
                               fontsize=8, ha='center', 
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))            
            # Configuration finale du titre et des axes
            ax.set_title("📈 Courbe de Loss d'Entraînement", fontsize=11, fontweight='bold', pad=15)
        
        else:
            # Pas de données réelles - afficher un message informatif
            print("⚠️ Aucune donnée de loss trouvée, affichage d'un exemple")
            
            # Créer un exemple réaliste de courbe de loss
            epochs = list(range(1, 101))
            # Courbe exponentielle décroissante avec du bruit réaliste
            base_loss = 2.0
            decay_rate = 0.03
            noise_factor = 0.05            
            np.random.seed(42)
            losses = []
            for i, epoch in enumerate(epochs):
                # Loss exponentielle décroissante avec plateaux occasionnels
                exponential_decay = base_loss * np.exp(-decay_rate * i)
                # Ajouter du bruit réaliste et quelques plateaux
                noise = noise_factor * exponential_decay * (2 * np.random.random() - 1)
                plateau_factor = 1.0 if i % 30 != 0 else 1.2  # Plateaux occasionnels
                loss_value = max(0.001, exponential_decay * plateau_factor + noise)
                losses.append(loss_value)
            
            ax.plot(epochs, losses, color='#3498db', linewidth=2.5, alpha=0.8, 
                   label='Exemple de courbe de loss', linestyle='--', marker='o', markersize=2, markevery=10)
            
            ax.set_xlabel('Époque', fontsize=10, fontweight='bold')
            ax.set_ylabel('Loss (MSE)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#fafafa')
            ax.legend(frameon=True, fancybox=True, shadow=True)
            
            # Message explicatif
            ax.text(0.5, 0.7, '⚠️ Aucune donnée d\'entraînement\n\n🔄 Entraînez le modèle pour voir\nla vraie courbe de convergence\n\n📈 Exemple de courbe typique\navec décroissance exponentielle',                   transform=ax.transAxes, va='center', ha='center', fontsize=10,
                   bbox=dict(boxstyle='round,pad=1', facecolor='#fff3cd', alpha=0.9,                            edgecolor='#f39c12', linewidth=2))
            ax.set_title("📈 Courbe de Loss d'Entraînement (Exemple)", fontsize=11, fontweight='bold', pad=15)
    
    def plot_predictions_vs_actual(self, ax, training_results, neural_network):
        """Dessiner un graphique moderne des prédictions vs valeurs réelles pour réseau neuronal"""
        ax.clear()
        ax.set_title("🧠 Réseau de Neurones: Performance de Prédiction", fontsize=11, fontweight='bold', pad=15)
        
        if ('y_test_real' in training_results and 'y_test_pred' in training_results and 
            len(training_results['y_test_real']) > 0):
            # Données réelles
            y_true = np.array(training_results['y_test_real'])
            y_pred = np.array(training_results['y_test_pred'])
            
            print(f"🧠 Affichage de {len(y_true)} prédictions neuronales")
            
            # Calculer les métriques
            from neural_network.utils import r2_score
            r2 = r2_score(y_true.tolist(), y_pred.tolist())
            mae = np.mean(np.abs(y_true - y_pred))
            mse = np.mean((y_true - y_pred)**2)
            rmse = np.sqrt(mse)
              # Calculer les erreurs absolues pour la couleur (plus approprié pour les réseaux de neurones)
            absolute_errors = np.abs(y_true - y_pred)
            
            # Ajouter un jitter (dispersion) sur l'axe des abscisses pour éviter la superposition
            jitter_strength = (np.max(y_true) - np.min(y_true)) * 0.005  # 0.5% de la plage
            y_true_jittered = y_true + np.random.normal(0, jitter_strength, len(y_true))
            
            # Nuage de points avec couleur selon l'erreur (style neuronal)
            scatter = ax.scatter(y_true_jittered, y_pred, c=absolute_errors, s=60, alpha=0.7, 
                               edgecolors='black', linewidth=0.5, zorder=3, 
                               cmap='RdYlGn_r', 
                               label=f'Prédictions neuronales ({len(y_true)} points)')
            
            # Zones de performance au lieu de la droite parfaite
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            margin = (max_val - min_val) * 0.05
            
            # Zone d'erreur acceptable (±10%)
            x_range = np.linspace(min_val - margin, max_val + margin, 100)
            ax.fill_between(x_range, x_range * 0.9, x_range * 1.1, alpha=0.2, color='green', 
                           label='Zone excellente (±10%)', zorder=1)
            ax.fill_between(x_range, x_range * 0.8, x_range * 0.9, alpha=0.15, color='orange', zorder=1)
            ax.fill_between(x_range, x_range * 1.1, x_range * 1.2, alpha=0.15, color='orange', 
                           label='Zone acceptable (±20%)', zorder=1)
            
            # Configuration des axes avec style neuronal
            ax.set_xlabel('Valeurs Réelles (Prix en milliers €)', fontsize=10, fontweight='bold')
            ax.set_ylabel('Prédictions Neuronales (Prix en milliers €)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#fafafa')
            
            # Égaliser les axes
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)
            ax.set_aspect('equal', adjustable='box')
            
            # Légende moderne
            ax.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', 
                     framealpha=0.9, edgecolor='gray', fontsize=9)
              # Calculer les pourcentages dans les zones de tolérance
            tolerance_10 = np.sum(np.abs((y_true - y_pred) / y_true) <= 0.1) / len(y_true) * 100
            tolerance_20 = np.sum(np.abs((y_true - y_pred) / y_true) <= 0.2) / len(y_true) * 100
            
            # Déterminer la qualité du modèle neuronal
            if r2 > 0.90 and tolerance_10 > 70:
                quality = "🟢 Réseau très performant"
                quality_color = '#27ae60'
            elif r2 > 0.80 and tolerance_10 > 50:
                quality = "🟡 Réseau performant"
                quality_color = '#2ecc71'
            elif r2 > 0.70 and tolerance_20 > 70:
                quality = "🔵 Réseau correct"
                quality_color = '#3498db'
            elif r2 > 0.50:
                quality = "🟠 Réseau en apprentissage"
                quality_color = '#f39c12'
            else:
                quality = "🔴 Réseau à améliorer"
                quality_color = '#e74c3c'
            
        else:
            # Pas de données réelles - créer un exemple informatif pour réseau neuronal
            print("⚠️ Aucune donnée de prédiction trouvée, affichage d'un exemple neuronal")
            
            # Créer un exemple réaliste de réseau neuronal
            np.random.seed(42)
            n_points = 80
            
            # Générer des valeurs vraies avec distribution réaliste
            y_true = np.random.gamma(2, 50, n_points)
            
            # Simuler les erreurs typiques d'un réseau de neurones
            # 1. Erreur non-linéaire (le réseau a des difficultés sur certaines plages)
            nonlinear_error = 0.08 * y_true * np.sin(y_true / 50) * np.random.normal(0, 1, n_points)
            # 2. Bruit d'apprentissage
            learning_noise = np.random.normal(0, 8, n_points)
            # 3. Biais léger du réseau
            network_bias = 5 * np.tanh(y_true / 100)
            
            y_pred = y_true + nonlinear_error + learning_noise + network_bias
            
            # Visualisation exemple pour réseau neuronal
            absolute_errors = np.abs(y_true - y_pred)
            scatter = ax.scatter(y_true, y_pred, c=absolute_errors, s=50, alpha=0.7, 
                               edgecolors='black', linewidth=0.5, cmap='RdYlGn_r',
                               label='Exemple: Prédictions neuronales')
            
            # Zones de tolérance
            min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
            x_range = np.linspace(min_val, max_val, 100)
            ax.fill_between(x_range, x_range * 0.9, x_range * 1.1, alpha=0.2, color='green', 
                           label='Zone excellente (±10%)')
            ax.fill_between(x_range, x_range * 0.8, x_range * 0.9, alpha=0.15, color='orange')
            ax.fill_between(x_range, x_range * 1.1, x_range * 1.2, alpha=0.15, color='orange', 
                           label='Zone acceptable (±20%)')
            
            ax.set_xlabel('Valeurs Réelles (Prix en milliers €)', fontsize=10)
            ax.set_ylabel('Prédictions Neuronales (Prix en milliers €)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
            ax.set_aspect('equal', adjustable='box')
            
            # Message d'exemple
            ax.text(0.98, 0.02, 'Exemple de réseau neuronal\n(données simulées)', 
                   transform=ax.transAxes, va='bottom', ha='right',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            relative_errors = np.abs((y_true - y_pred) / (y_true + 1e-8)) * 100
              # Créer le scatter plot
            scatter = ax.scatter(y_true, y_pred, c=relative_errors, cmap='RdYlGn_r', 
                               alpha=0.7, s=60, edgecolors='white', linewidth=0.8)
            
            # Ligne de prédiction parfaite (diagonale y=x)
            min_val, max_val = min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3, 
                   label='Prédictions parfaites (y=x)', alpha=0.8)
            
            ax.set_xlabel('Valeurs Réelles', fontsize=10, fontweight='bold')
            ax.set_ylabel('Prédictions du Modèle', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#fafafa')
            ax.legend(frameon=True, fancybox=True, shadow=True)
            
            # Message explicatif
            ax.text(0.5, 0.7, '⚠️ Aucune donnée de test\n\n🔄 Entraînez le modèle pour voir\nles vraies prédictions\n\n📊 Exemple : nuage de points\navec diagonale y=x parfaite', 
                   transform=ax.transAxes, va='center', ha='center', fontsize=10,
                   bbox=dict(boxstyle='round,pad=1', facecolor='#fff3cd', alpha=0.9,
                            edgecolor='#f39c12', linewidth=2))
            
            ax.set_title("🎯 Prédictions vs Valeurs Réelles (Exemple)", fontsize=11, fontweight='bold', pad=15)
        
    def update_text_widget(self, widget, text):
        """Mettre à jour un widget texte"""
        widget.config(state=tk.NORMAL)
        widget.delete(1.0, tk.END)
        widget.insert(1.0, text)
        widget.config(state=tk.DISABLED)
        
    def load_model_info(self, model_info):
        """Charger les informations d'un modèle (ancien format)"""
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
        
    def make_prediction(self):
        """Ouvrir l'interface de prédiction avec les variables du modèle"""
        if not self.model_info:
            messagebox.showinfo("Information", "Aucun modèle chargé.")
            return
        
        # Créer une fenêtre de prédiction
        prediction_window = tk.Toplevel(self)
        prediction_window.title("🎯 Faire une prédiction avec le modèle")
        prediction_window.geometry("700x800")
        prediction_window.configure(bg=self.controller.colors["bg_light"])
        prediction_window.grab_set()  # Rendre la fenêtre modale
        
        # Centrer la fenêtre
        prediction_window.update_idletasks()
        x = (prediction_window.winfo_screenwidth() // 2) - (700 // 2)
        y = (prediction_window.winfo_screenheight() // 2) - (800 // 2)
        prediction_window.geometry(f"700x800+{x}+{y}")
        
        # Titre principal
        title_label = tk.Label(
            prediction_window,
            text="🎯 Prédiction avec Réseau de Neurones",
            font=('Helvetica', 18, 'bold'),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["primary"]
        )
        title_label.pack(pady=(20, 10))
        
        # Sous-titre avec nom du modèle
        model_name = self.model_info.get('model_info', {}).get('name', 'Modèle inconnu')
        subtitle_label = tk.Label(
            prediction_window,
            text=f"Modèle: {model_name}",
            font=('Helvetica', 12, 'italic'),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["text"]
        )
        subtitle_label.pack(pady=(0, 10))        # Récupérer les variables d'entrée depuis les métadonnées du modèle
        all_feature_columns = self.model_info.get('feature_columns', [])
        target_column = self.model_info.get('target_column', 'valeur_cible')
        
        # Filtrer les variables non pertinentes pour la prédiction de prix
        excluded_variables = ['longitude', 'latitude', 'code_postal', 'code_commune', 'code_departement', 'numero_disposition']
        feature_columns = [col for col in all_feature_columns if col not in excluded_variables]
        
        if not feature_columns:
            messagebox.showerror("Erreur", "Aucune variable d'entrée pertinente trouvée après filtrage.")
            prediction_window.destroy()
            return
          # Informations sur le modèle et limites
        info_frame = tk.LabelFrame(
            prediction_window,
            text="Informations du modèle et limites d'entraînement",
            font=('Helvetica', 12, 'bold'),            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["primary"],
            padx=15,
            pady=10
        )
        info_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        # Informations générales sur le modèle
        info_text = f"Variables d'entrée: {len(feature_columns)}\nVariable cible: {target_column}\nArchitecture: {self.loaded_model_data.get('network_architecture', {}).get('layers_config', 'Inconnue')}"
        
        # Ajouter un résumé des limites d'entraînement
        feature_mins = self.normalization_params.get('feature_mins', [])
        feature_maxs = self.normalization_params.get('feature_maxs', [])
        all_feature_columns = self.model_info.get('feature_columns', [])
        
        if feature_mins and feature_maxs:
            info_text += "\n\n📊 RÉSUMÉ DES LIMITES D'ENTRAÎNEMENT:"
            for feature_name in feature_columns:
                if feature_name in all_feature_columns:
                    feature_index = all_feature_columns.index(feature_name)
                    if feature_index < len(feature_mins) and feature_index < len(feature_maxs):
                        min_val = feature_mins[feature_index]
                        max_val = feature_maxs[feature_index]
                        info_text += f"\n• {feature_name}: {min_val:.1f} - {max_val:.1f}"
            
            info_text += "\n\n💡 Les valeurs par défaut sont calculées à 40% dans la plage d'entraînement."
            info_text += "\n⚠️ Les valeurs hors de ces limites peuvent donner des prédictions moins fiables."
        
        tk.Label(
            info_frame,
            text=info_text,
            font=('Helvetica', 10),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            justify=tk.LEFT
        ).pack(anchor=tk.W)
        
        # Frame principal pour les entrées (avec scroll si nécessaire)
        main_canvas = tk.Canvas(prediction_window, bg=self.controller.colors["bg_light"])
        scrollbar = ttk.Scrollbar(prediction_window, orient="vertical", command=main_canvas.yview)
        scrollable_frame = tk.Frame(main_canvas, bg=self.controller.colors["bg_light"])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Section des variables d'entrée
        input_frame = tk.LabelFrame(
            scrollable_frame,
            text="Variables d'entrée du modèle",
            font=('Helvetica', 14, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["primary"],
            padx=20,
            pady=15
        )
        input_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Variables pour stocker les entrées
        self.prediction_vars = {}
          # Créer les champs d'entrée dynamiquement basés sur les métadonnées du modèle
        for i, feature_name in enumerate(feature_columns):
            # Frame pour chaque entrée
            field_frame = tk.Frame(input_frame, bg=self.controller.colors["bg_white"])
            field_frame.pack(fill=tk.X, pady=8)
            
            # Label avec le nom de la variable
            label_frame = tk.Frame(field_frame, bg=self.controller.colors["bg_white"])
            label_frame.pack(fill=tk.X)
            
            tk.Label(
                label_frame,
                text=f"{feature_name}:",
                font=('Helvetica', 11, 'bold'),
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text"]
            ).pack(side=tk.LEFT)
            
            # Description générique basée sur le nom de la variable
            description = self._get_variable_description(feature_name)
            tk.Label(
                label_frame,
                text=f"({description})",
                font=('Helvetica', 9, 'italic'),
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text_invisible"]
            ).pack(side=tk.LEFT, padx=(10, 0))
            
            # Récupérer et afficher les limites d'entraînement
            feature_mins = self.normalization_params.get('feature_mins', [])
            feature_maxs = self.normalization_params.get('feature_maxs', [])
            all_feature_columns = self.model_info.get('feature_columns', [])
            
            # Trouver l'index de cette variable dans la liste complète des features
            if feature_name in all_feature_columns:
                feature_index = all_feature_columns.index(feature_name)
                if feature_index < len(feature_mins) and feature_index < len(feature_maxs):
                    min_val = feature_mins[feature_index]
                    max_val = feature_maxs[feature_index]
                    
                    # Afficher les limites d'entraînement
                    limits_label = tk.Label(
                        label_frame,
                        text=f"[Limites: {min_val:.1f} - {max_val:.1f}]",
                        font=('Helvetica', 8, 'bold'),
                        bg=self.controller.colors["bg_white"],
                        fg=self.controller.colors["primary"]
                    )
                    limits_label.pack(side=tk.RIGHT, padx=(10, 0))
            
            # Frame pour le champ d'entrée et les boutons
            entry_frame = tk.Frame(field_frame, bg=self.controller.colors["bg_white"])
            entry_frame.pack(fill=tk.X, pady=(5, 0))
            
            # Champ d'entrée avec valeur par défaut
            default_value = self._get_default_value(feature_name)
            self.prediction_vars[feature_name] = tk.StringVar(value=str(default_value))
            entry = ttk.Entry(
                entry_frame,
                textvariable=self.prediction_vars[feature_name],
                font=('Helvetica', 11),
                width=20
            )
            entry.pack(side=tk.LEFT, padx=(0, 10))
            
            # Boutons pour définir les valeurs min/max rapidement
            if feature_name in all_feature_columns:
                feature_index = all_feature_columns.index(feature_name)
                if feature_index < len(feature_mins) and feature_index < len(feature_maxs):
                    min_val = feature_mins[feature_index]
                    max_val = feature_maxs[feature_index]
                    
                    # Bouton Min
                    min_button = ttk.Button(
                        entry_frame,
                        text=f"Min ({min_val:.1f})",
                        command=lambda v=min_val, var=feature_name: self.prediction_vars[var].set(str(v)),
                        width=12
                    )
                    min_button.pack(side=tk.LEFT, padx=(0, 5))
                    
                    # Bouton Max
                    max_button = ttk.Button(
                        entry_frame,
                        text=f"Max ({max_val:.1f})",
                        command=lambda v=max_val, var=feature_name: self.prediction_vars[var].set(str(v)),
                        width=12
                    )
                    max_button.pack(side=tk.LEFT, padx=(0, 5))
            
            # Validation en temps réel
            entry.bind('<KeyRelease>', lambda e, var=feature_name: self.validate_input(var))
        
        # Section des résultats
        self.result_frame = tk.LabelFrame(
            scrollable_frame,
            text=f"Résultat de la prédiction ({target_column})",
            font=('Helvetica', 14, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["success"],
            padx=20,
            pady=15
        )
        self.result_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Label pour afficher le résultat
        self.prediction_result_label = tk.Label(
            self.result_frame,
            text="Cliquez sur 'Prédire' pour obtenir une estimation",
            font=('Helvetica', 12),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            wraplength=600
        )
        self.prediction_result_label.pack(pady=10)
        
        # Frame pour les boutons d'action
        button_frame = tk.Frame(scrollable_frame, bg=self.controller.colors["bg_light"])
        button_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Bouton de prédiction
        predict_button = ttk.Button(
            button_frame,
            text="🎯 Prédire avec le réseau neuronal",
            command=lambda: self.perform_prediction(prediction_window, feature_columns, target_column),
            style='Add.TButton'
        )
        predict_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Bouton de réinitialisation
        ttk.Button(
            button_frame,
            text="🔄 Réinitialiser",
            command=lambda: self._reset_prediction_values(feature_columns),
            style='TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        # Bouton de fermeture
        ttk.Button(
            button_frame,
            text="Fermer",
            command=prediction_window.destroy,
            style='TButton'
        ).pack(side=tk.RIGHT)
        
        # Pack le canvas et scrollbar
        main_canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=(0, 20))
        scrollbar.pack(side="right", fill="y", pady=(0, 20))
    
    def _get_variable_description(self, variable_name):
        """Générer une description générique basée sur le nom de la variable"""
        descriptions = {
            'surface_reelle_bati': 'Surface en m²',
            'nombre_pieces_principales': 'Nombre entier',
            'surface_terrain': 'Surface en m²', 
            'longitude': 'Coordonnée géographique',
            'latitude': 'Coordonnée géographique',
            'prix': 'Valeur monétaire',
            'valeur_fonciere': 'Valeur monétaire',
            'age': 'Nombre d\'années',
            'etage': 'Numéro d\'étage',
            'superficie': 'Surface en m²'
        }
        
        # Recherche par mots-clés dans le nom de la variable
        variable_lower = variable_name.lower()
        
        if 'surface' in variable_lower or 'superficie' in variable_lower:
            return 'Surface en m²'
        elif 'prix' in variable_lower or 'valeur' in variable_lower or 'montant' in variable_lower:
            return 'Valeur monétaire'
        elif 'piece' in variable_lower or 'chambre' in variable_lower:
            return 'Nombre entier'
        elif 'longitude' in variable_lower or 'latitude' in variable_lower:
            return 'Coordonnée géographique'
        elif 'age' in variable_lower or 'annee' in variable_lower:
            return 'Nombre d\'années'
        elif 'etage' in variable_lower or 'niveau' in variable_lower:
            return 'Numéro d\'étage'
        else:
            return 'Valeur numérique'
    
    def _get_default_value(self, variable_name):
        """Générer une valeur par défaut basée sur les limites d'entraînement ou des valeurs moyennes réalistes"""
        # D'abord, essayer d'utiliser les statistiques du modèle si disponibles
        feature_mins = self.normalization_params.get('feature_mins', [])
        feature_maxs = self.normalization_params.get('feature_maxs', [])
        all_feature_columns = self.model_info.get('feature_columns', [])
        
        # Si nous avons les données d'entraînement, utiliser une valeur dans la plage
        if variable_name in all_feature_columns:
            feature_index = all_feature_columns.index(variable_name)
            if feature_index < len(feature_mins) and feature_index < len(feature_maxs):
                min_val = feature_mins[feature_index]
                max_val = feature_maxs[feature_index]
                # Utiliser une valeur entre le 30% et 70% de la plage (plus réaliste qu'une moyenne simple)
                return min_val + (max_val - min_val) * 0.4  # 40% dans la plage
        
        # Valeurs par défaut de secours plus réalistes basées sur les données immobilières typiques
        defaults = {
            'surface_reelle_bati': 90,
            'nombre_pieces_principales': 4,
            'surface_terrain': 800,
            'longitude': 6.35,
            'latitude': 47.1,
            'prix': 180000,
            'valeur_fonciere': 180000,
            'age': 15,
            'etage': 1
        }
        
        # Recherche par nom exact d'abord
        if variable_name in defaults:
            return defaults[variable_name]
        
        # Recherche par mots-clés
        variable_lower = variable_name.lower()
        
        if 'surface' in variable_lower and 'bati' in variable_lower:
            return 90
        elif 'surface' in variable_lower and 'terrain' in variable_lower:
            return 800
        elif 'piece' in variable_lower:
            return 4
        elif 'longitude' in variable_lower:
            return 6.35
        elif 'latitude' in variable_lower:
            return 47.1
        elif 'prix' in variable_lower or 'valeur' in variable_lower:
            return 180000
        else:
            return 1
    
    def _reset_prediction_values(self, feature_columns):
        """Réinitialiser les valeurs de prédiction aux valeurs par défaut"""
        for feature_name in feature_columns:
            if feature_name in self.prediction_vars:
                default_value = self._get_default_value(feature_name)
                self.prediction_vars[feature_name].set(str(default_value))
    
    def validate_input(self, var_name):
        """Valider les entrées en temps réel avec feedback visuel"""
        try:
            value_str = self.prediction_vars[var_name].get()
            if not value_str:
                return
                
            value = float(value_str)
            
            # Vérifier les limites d'entraînement
            feature_mins = self.normalization_params.get('feature_mins', [])
            feature_maxs = self.normalization_params.get('feature_maxs', [])
            all_feature_columns = self.model_info.get('feature_columns', [])
            
            if var_name in all_feature_columns:
                feature_index = all_feature_columns.index(var_name)
                if feature_index < len(feature_mins) and feature_index < len(feature_maxs):
                    min_val = feature_mins[feature_index]
                    max_val = feature_maxs[feature_index]
                    
                    # Créer un feedback visuel sur la validité
                    if value < min_val or value > max_val:
                        # Valeur hors limites - feedback rouge
                        self.prediction_vars[var_name].set(value_str)  # Garder la valeur mais signaler
                    else:
                        # Valeur dans les limites - feedback vert implicite
                        pass
                        
        except ValueError:
            # Valeur non numérique - laisser l'utilisateur corriger
            pass
    
    def load_example(self, example_values):
        """Charger un exemple prédéfini"""
        for var_name, value in example_values.items():
            if var_name in self.prediction_vars:
                self.prediction_vars[var_name].set(value)
    
    def perform_prediction(self, window, feature_columns, target_column):
        """Effectuer la prédiction avec le modèle chargé en utilisant les variables dynamiques"""
        try:
            # Récupérer les valeurs d'entrée pour les variables filtrées (celles affichées)
            user_input_values = {}
            
            for feature_name in feature_columns:
                try:
                    value = float(self.prediction_vars[feature_name].get())
                    user_input_values[feature_name] = value
                except (ValueError, KeyError):
                    messagebox.showerror("Erreur", f"Valeur invalide pour {feature_name}")
                    return
            
            # Récupérer toutes les variables d'entrée originales du modèle
            all_feature_columns = self.model_info.get('feature_columns', [])
              # Construire le vecteur d'entrée complet avec des valeurs par défaut pour les variables exclues
            full_input_values = []
            
            # Utiliser des valeurs par défaut plus intelligentes basées sur les statistiques d'entraînement
            feature_mins = self.normalization_params.get('feature_mins', [])
            feature_maxs = self.normalization_params.get('feature_maxs', [])
            
            for i, feature_name in enumerate(all_feature_columns):
                if feature_name in user_input_values:
                    # Utiliser la valeur saisie par l'utilisateur
                    full_input_values.append(user_input_values[feature_name])
                else:
                    # Utiliser une valeur par défaut intelligente pour les variables exclues
                    if i < len(feature_mins) and i < len(feature_maxs):
                        # Utiliser une valeur dans la plage d'entraînement (40% de la plage)
                        default_val = feature_mins[i] + (feature_maxs[i] - feature_mins[i]) * 0.4
                    else:
                        # Valeurs de secours spécifiques par type de variable
                        if 'longitude' in feature_name.lower():
                            default_val = 6.35
                        elif 'latitude' in feature_name.lower():
                            default_val = 47.1
                        elif 'code' in feature_name.lower():
                            default_val = 25000
                        else:
                            default_val = 1.0
                    
                    full_input_values.append(default_val)
              # Debug minimal: vérifier si les variables sont dans les limites
            print(f"Variables utilisées: {list(user_input_values.keys())}")
            
            # Utiliser le réseau de neurones et les paramètres de normalisation chargés
            neural_network = self.neural_network
            normalization_params = self.normalization_params
            
            if not neural_network:
                messagebox.showerror("Erreur", "Réseau de neurones non trouvé dans le modèle")
                return
              # Normaliser les données d'entrée
            normalized_inputs = full_input_values.copy()
            if normalization_params:
                feature_mins = normalization_params.get('feature_mins')
                feature_maxs = normalization_params.get('feature_maxs')
                
                if feature_mins is not None and feature_maxs is not None:
                    # Vérifier si les valeurs sont dans les limites d'entraînement
                    warnings = []
                    
                    # Normalisation min-max
                    for i, value in enumerate(full_input_values):
                        if i < len(feature_mins) and i < len(feature_maxs):
                            # Vérifier les limites
                            if value < feature_mins[i] or value > feature_maxs[i]:
                                var_name = feature_columns[i] if i < len(feature_columns) else f"Variable {i+1}"
                                warnings.append(f"⚠️ {var_name}: {value} (limites d'entraînement: {feature_mins[i]:.1f} - {feature_maxs[i]:.1f})")
                            
                            if feature_maxs[i] != feature_mins[i]:
                                normalized_value = (value - feature_mins[i]) / (feature_maxs[i] - feature_mins[i])
                            else:
                                normalized_value = 0.0
                            normalized_inputs[i] = normalized_value
                    
                    if warnings:
                        warning_text = "Les valeurs suivantes sont hors des limites d'entraînement :\n" + "\n".join(warnings) + "\n\nLes prédictions peuvent être moins fiables."
                        messagebox.showwarning("Valeurs hors limites", warning_text)
            
            # Faire la prédiction
            prediction = neural_network.predict(normalized_inputs)
            predicted_value = prediction[0] if isinstance(prediction, list) else prediction
            
            # Dénormaliser la prédiction
            if normalization_params:
                target_min = normalization_params.get('target_min')
                target_max = normalization_params.get('target_max')
                
                if target_min is not None and target_max is not None:
                    predicted_value = predicted_value * (target_max - target_min) + target_min
            
            # Afficher le résultat
            predicted_value = max(0, predicted_value)  # S'assurer que la valeur est positive
              # Créer un message détaillé avec les variables saisies par l'utilisateur
            details_lines = []
            for feature_name, value in user_input_values.items():
                # Formatage adapté selon le type de variable
                if 'prix' in feature_name.lower() or 'valeur' in feature_name.lower():
                    details_lines.append(f"• {feature_name}: {value:,.0f} €")
                elif any(keyword in feature_name.lower() for keyword in ['longitude', 'latitude']):
                    details_lines.append(f"• {feature_name}: {value:.4f}")
                elif 'piece' in feature_name.lower() or 'nombre' in feature_name.lower():
                    details_lines.append(f"• {feature_name}: {int(value)}")
                elif 'surface' in feature_name.lower():
                    details_lines.append(f"• {feature_name}: {value:.0f} m²")
                else:
                    details_lines.append(f"• {feature_name}: {value}")
            
            # Formatage de la valeur prédite selon la variable cible
            if 'prix' in target_column.lower() or 'valeur' in target_column.lower():
                formatted_prediction = f"{predicted_value:,.0f} €"
            else:
                formatted_prediction = f"{predicted_value:.2f}"
            
            result_text = f"""
🎯 {target_column} estimé: {formatted_prediction}

📊 Variables d'entrée utilisées:
{chr(10).join(details_lines)}

🧠 Modèle: Réseau de Neurones
Architecture: {self.loaded_model_data.get('network_architecture', {}).get('layers_config', 'Inconnue')}

⚠️ Cette estimation est générée par un réseau de neurones et doit être considérée comme indicative.
"""
            
            self.prediction_result_label.configure(
                text=result_text,
                fg=self.controller.colors["success"],
                font=('Helvetica', 11, 'bold')
            )
            
        except Exception as e:
            error_msg = f"Erreur lors de la prédiction: {str(e)}"
            messagebox.showerror("Erreur", error_msg)
            print(f"Erreur de prédiction: {e}")
            import traceback
            traceback.print_exc()
        
    def retrain_model(self):
        """Réentraîner le modèle"""
        if not self.model_info:
            messagebox.showwarning("Attention", "Aucun modèle chargé")
            return
            
        model_info = self.model_info.get('model_info', self.model_info)
        if 'data_file' in model_info:
            data_file = model_info['data_file']
            
            # Essayer plusieurs chemins possibles
            possible_paths = [
                data_file,  # Chemin direct
                os.path.join("data", data_file),  # Dans le dossier data relatif
                os.path.join(os.path.dirname(__file__), "..", "data", data_file),  # Data depuis le répertoire pages
                os.path.join(os.getcwd(), "data", data_file),  # Data depuis le répertoire courant
                os.path.join(os.getcwd(), data_file),  # Fichier dans le répertoire courant
            ]
            
            file_path = None
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    file_path = abs_path
                    break
            
            if file_path:
                print(f"Fichier de données trouvé: {file_path}")
                self.controller.show_model_training(file_path, model_info)
            else:
                # Proposer à l'utilisateur de sélectionner le fichier
                from tkinter import filedialog
                result = messagebox.askyesno(
                    "Fichier non trouvé", 
                    f"Fichier de données '{data_file}' non trouvé.\n\nVoulez-vous sélectionner manuellement le fichier de données?"
                )
                
                if result:
                    file_path = filedialog.askopenfilename(
                        title="Sélectionner le fichier de données",
                        filetypes=[
                            ("Fichiers CSV", "*.csv"),
                            ("Tous les fichiers", "*.*")
                        ],
                        initialdir=os.path.join(os.getcwd(), "data") if os.path.exists(os.path.join(os.getcwd(), "data")) else os.getcwd()
                    )
                    
                    if file_path:
                        # Mettre à jour les métadonnées du modèle avec le nouveau chemin
                        model_info['data_file'] = os.path.basename(file_path)
                        self.controller.show_model_training(file_path, model_info)
                    else:
                        messagebox.showinfo("Annulé", "Réentraînement annulé")
                else:
                    messagebox.showwarning("Attention", f"Impossible de réentraîner: fichier '{data_file}' non trouvé")
        else:
            # Pas de fichier de données dans les métadonnées, proposer de sélectionner un fichier
            result = messagebox.askyesno(
                "Fichier de données manquant", 
                "Aucun fichier de données n'est associé à ce modèle.\n\nVoulez-vous sélectionner un fichier de données pour le réentraîner?"
            )
            
            if result:
                from tkinter import filedialog
                file_path = filedialog.askopenfilename(
                    title="Sélectionner le fichier de données pour le réentraînement",
                    filetypes=[
                        ("Fichiers CSV", "*.csv"),
                        ("Tous les fichiers", "*.*")
                    ],
                    initialdir=os.path.join(os.getcwd(), "data") if os.path.exists(os.path.join(os.getcwd(), "data")) else os.getcwd()
                )
                
                if file_path:
                    # Mettre à jour les métadonnées du modèle avec le nouveau chemin
                    model_info['data_file'] = os.path.basename(file_path)
                    self.controller.show_model_training(file_path, model_info)
                else:
                    messagebox.showinfo("Annulé", "Réentraînement annulé")
            else:
                messagebox.showwarning("Attention", "Impossible de réentraîner sans fichier de données")
            
    def delete_model(self):
        """Supprimer le modèle"""
        if not self.model_info:
            messagebox.showwarning("Attention", "Aucun modèle chargé")
            return
            
        model_info = self.model_info.get('model_info', self.model_info)
        model_name = model_info.get('name', 'Modèle inconnu')
        
        result = messagebox.askyesno(
            "Confirmation",
            f"Êtes-vous sûr de vouloir supprimer le modèle '{model_name}'?\n\nCette action est irréversible."
        )
        
        if result:
            try:
                # Utiliser ModelPersistence pour supprimer le modèle
                persistence = ModelPersistence()
                # Trouver le nom de fichier correspondant
                for model in persistence.list_saved_models():
                    if model['model_info']['name'] == model_name:
                        persistence.delete_model(model['filename'])
                        break
                
                self.controller.status_label.config(
                    text=f"Modèle '{model_name}' supprimé"
                )
                
                messagebox.showinfo("Succès", f"Modèle '{model_name}' supprimé avec succès")
                
                # Retourner à la page d'accueil et actualiser
                self.controller.show_page("HomePage")
                self.controller.refresh_all_pages()
                
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la suppression: {str(e)}")
