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
        
        # Créer les sections
        self.create_details_content()
        
        # Créer une section pour les graphiques
        self.charts_frame = tk.Frame(self.scrollable_content, bg=self.controller.colors["bg_white"])
        self.charts_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
        
        self.config_text = tk.Text(
            self.config_section,
            height=6,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11),
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.config_text.pack(fill=tk.X, pady=(10, 10))
        
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
        
        self.metrics_text = tk.Text(
            self.metrics_section,
            height=6,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11),
            state=tk.DISABLED,
            wrap=tk.WORD
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

    def load_model_details(self, loaded_model_data):
        """Charger et afficher les détails d'un modèle sauvegardé avec visualisations"""
        try:
            if isinstance(loaded_model_data, dict) and "model_info" in loaded_model_data:
                # Modèle chargé depuis un fichier sauvegardé
                model_info = loaded_model_data["model_info"]
                neural_network = loaded_model_data["neural_network"]
                normalization_params = loaded_model_data.get("normalization_params", {})
                
                self.model_info = loaded_model_data
                
                # Mettre à jour le titre
                self.title_label.config(text=f"📊 Détails - {model_info['name']}")
                
                # Informations générales
                info_text = f"""📋 Nom: {model_info['name']}
📊 Type: Réseau de Neurones - Régression
🎯 Variable cible: {model_info.get('target_column', 'Non définie')}
📁 Fichier source: {model_info.get('data_file', 'Non spécifié')}
📅 Créé le: {model_info.get('created_date', 'Date inconnue')}

🧠 ARCHITECTURE DU RÉSEAU:
   • Configuration: {neural_network.layers_config}
   • Entrées: {neural_network.layers_config[0]} variables
   • Couches cachées: {len(neural_network.layers_config) - 2} ({neural_network.layers_config[1:-1] if len(neural_network.layers_config) > 2 else 'Aucune'})
   • Sortie: {neural_network.layers_config[-1]} valeur(s)
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
                
                # Métriques de performance
                metrics_text = f"""🎯 PERFORMANCE DU MODÈLE:
═══════════════════════════════════

✅ Modèle entraîné avec succès
📊 Architecture: Réseau de Neurones
🎯 MSE: {model_info.get('mse', 'Non calculé')}
📈 MAE: {model_info.get('mae', 'Non calculé')}
📊 R²: {model_info.get('r2_score', 'Non calculé')}

⚡ PARAMÈTRES D'ENTRAÎNEMENT:
   • Couches: {len(neural_network.layers_config)} ({neural_network.layers_config})
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
            training_results = loaded_model_data.get("training_results", {})
            
            # Créer une figure avec 3 subplots (1 ligne, 3 colonnes)
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Analyse du Modèle: {model_info["name"]}', fontsize=14, fontweight='bold')
            
            # 1. Architecture du réseau (style création de modèle)
            self.plot_network_architecture_enhanced(ax1, neural_network)
            
            # 2. Courbe de Loss (données réelles si disponibles)
            self.plot_training_loss_curve(ax2, training_results)
            
            # 3. Distribution des prédictions (graphique réel)
            self.plot_predictions_vs_actual(ax3, training_results, neural_network)
            
            plt.tight_layout()
            
            # Intégrer dans Tkinter
            canvas = FigureCanvasTkAgg(fig, self.charts_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Afficher le statut des données
            has_real_data = 'loss_history' in training_results and 'y_test_real' in training_results
            status_text = "✅ Graphiques basés sur les données d'entraînement sauvegardées" if has_real_data else "⚠️ Certains graphiques sont simulés (données partielles)"
            status_label = tk.Label(
                self.charts_frame,
                text=status_text,
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["success"] if has_real_data else self.controller.colors["accent"],
                font=('Helvetica', 10)
            )
            status_label.pack(pady=5)
            
        except Exception as e:
            # En cas d'erreur, afficher un message informatif
            error_label = tk.Label(
                self.charts_frame,
                text=f"⚠️ Impossible de créer les visualisations:\n{str(e)}\n\nLe modèle reste fonctionnel pour les prédictions.",
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["accent"],
                font=('Helvetica', 11),
                justify=tk.CENTER
            )
            error_label.pack(expand=True, pady=20)
    
    def plot_network_architecture_enhanced(self, ax, neural_network):
        """Dessiner l'architecture du réseau de façon améliorée"""
        ax.clear()
        layers_config = neural_network.layers_config
        
        ax.set_title("Architecture du Réseau", fontsize=12, fontweight='bold')
        
        # Paramètres de dessin
        n_layers = len(layers_config)
        max_neurons = max(layers_config)
        layer_spacing = 2.0
        neuron_radius = 0.3
        max_display_neurons = 8  # Limite d'affichage pour éviter l'encombrement
        
        # Couleurs pour chaque type de couche
        layer_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        for i, n_neurons in enumerate(layers_config):
            x = i * layer_spacing
            
            # Limiter l'affichage pour éviter l'encombrement
            display_neurons = min(n_neurons, max_display_neurons)
            
            # Calculer les positions Y
            if display_neurons == n_neurons:
                y_positions = np.linspace(-max_neurons/2, max_neurons/2, n_neurons)
            else:
                # Afficher quelques neurones au début, au milieu et à la fin
                y_positions = np.linspace(-max_neurons/2, max_neurons/2, display_neurons)
            
            # Couleur de la couche
            color = layer_colors[min(i, len(layer_colors)-1)]
            
            # Dessiner les neurones
            for j, y in enumerate(y_positions):
                circle = plt.Circle((x, y), neuron_radius, color=color, alpha=0.7)
                ax.add_patch(circle)
                
                # Si on a tronqué l'affichage, ajouter "..." au milieu
                if display_neurons < n_neurons and j == display_neurons // 2:
                    ax.text(x, y, '...', ha='center', va='center', fontsize=8, fontweight='bold')
            
            # Dessiner les connexions vers la couche suivante
            if i < len(layers_config) - 1:
                next_x = (i + 1) * layer_spacing
                next_n_neurons = layers_config[i + 1]
                next_display_neurons = min(next_n_neurons, max_display_neurons)
                next_y_positions = np.linspace(-max_neurons/2, max_neurons/2, next_display_neurons)
                
                # Dessiner quelques connexions représentatives
                for y1 in y_positions[::2]:  # Prendre 1 neurone sur 2
                    for y2 in next_y_positions[::2]:
                        ax.plot([x + neuron_radius, next_x - neuron_radius], [y1, y2], 
                               'gray', alpha=0.3, linewidth=0.5)
            
            # Étiquette de la couche
            layer_name = ['Entrée', 'Cachée', 'Cachée 2', 'Sortie'][min(i, 3)]
            ax.text(x, max_neurons/2 + 1, f"{layer_name}\n({n_neurons})", 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlim(-0.5, (n_layers-1) * layer_spacing + 0.5)
        ax.set_ylim(-max_neurons/2 - 1.5, max_neurons/2 + 2)
        ax.set_aspect('equal')
        ax.axis('off')
    
    def plot_training_loss_curve(self, ax, training_results):
        """Dessiner la courbe de loss d'entraînement"""
        ax.clear()
        ax.set_title("Courbe de Loss d'Entraînement", fontsize=12, fontweight='bold')
        
        if 'loss_history' in training_results and len(training_results['loss_history']) > 0:
            # Données réelles
            losses = training_results['loss_history']
            epochs = range(1, len(losses) + 1)
            ax.plot(epochs, losses, 'b-', linewidth=2, label='Loss réelle')
            ax.set_xlabel('Époque')
            ax.set_ylabel('Loss (MSE)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Ajouter des informations sur la convergence
            final_loss = losses[-1]
            ax.text(0.02, 0.98, f'Loss finale: {final_loss:.6f}', 
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        else:
            # Simulation de données pour la démonstration
            epochs = range(1, 101)
            # Générer une courbe de loss qui décroît
            losses = [10 * np.exp(-i/20) + 0.1 + 0.05 * np.random.random() for i in epochs]
            ax.plot(epochs, losses, 'r--', linewidth=2, alpha=0.7, label='Loss simulée')
            ax.set_xlabel('Époque')
            ax.set_ylabel('Loss (MSE)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.text(0.02, 0.98, 'Données simulées\n(pas de données d\'entraînement)', 
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
    
    def plot_predictions_vs_actual(self, ax, training_results, neural_network):
        """Dessiner le graphique des prédictions vs valeurs réelles"""
        ax.clear()
        ax.set_title("Prédictions vs Valeurs Réelles", fontsize=12, fontweight='bold')
        
        if ('y_test_real' in training_results and 'y_test_pred' in training_results and 
            len(training_results['y_test_real']) > 0):
            # Données réelles
            y_true = training_results['y_test_real']
            y_pred = training_results['y_test_pred']
            
            ax.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50)
            
            # Ligne de régression parfaite (y=x)
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
            
            ax.set_xlabel('Valeurs Réelles')
            ax.set_ylabel('Prédictions')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Calculer et afficher R²
            from neural_network.utils import r2_score
            r2 = r2_score(y_true, y_pred)
            ax.text(0.02, 0.98, f'R² = {r2:.3f}', 
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        else:
            # Simulation de données
            np.random.seed(42)
            n_points = 50
            y_true = np.random.normal(100, 20, n_points)
            noise = np.random.normal(0, 5, n_points)
            y_pred = y_true + noise  # Simulation d'un modèle avec un peu de bruit
            
            ax.scatter(y_true, y_pred, alpha=0.6, color='orange', s=50)
            
            # Ligne de régression parfaite
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Prédiction parfaite')
            
            ax.set_xlabel('Valeurs Réelles (simulées)')
            ax.set_ylabel('Prédictions (simulées)')
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.text(0.02, 0.98, 'Données simulées\n(pas de données de test)', 
                   transform=ax.transAxes, va='top', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))
        
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
        """Faire une prédiction simple"""
        if not self.model_info:
            messagebox.showinfo("Information", "Aucun modèle chargé.")
            return
            
        # Scroller vers la zone de prédiction
        self.main_canvas.yview_moveto(1.0)
        
    def retrain_model(self):
        """Réentraîner le modèle"""
        if not self.model_info:
            messagebox.showwarning("Attention", "Aucun modèle chargé")
            return
            
        model_info = self.model_info.get('model_info', self.model_info)
        if 'data_file' in model_info:
            # Construire le chemin complet du fichier
            data_file = model_info['data_file']
            file_path = os.path.join("data", data_file)
            if os.path.exists(file_path):
                self.controller.show_model_training(file_path, model_info)
            else:
                messagebox.showwarning("Attention", f"Fichier de données non trouvé: {data_file}")
        else:
            messagebox.showwarning("Attention", "Impossible de réentraîner: fichier de données non trouvé")
            
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
