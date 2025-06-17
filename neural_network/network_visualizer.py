import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import numpy as np
import math

class NetworkArchitectureVisualizer:
    """Classe pour visualiser l'architecture d'un réseau de neurones"""
    
    def __init__(self, parent_frame, colors):
        self.parent_frame = parent_frame
        self.colors = colors
        self.fig = None
        self.canvas = None
        self.ax = None
        
    def create_visualization(self, layers_config, title="Architecture du Réseau"):
        """Créer la visualisation de l'architecture du réseau"""
        # Nettoyer le frame parent
        for widget in self.parent_frame.winfo_children():
            widget.destroy()
            
        # Créer la figure matplotlib
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.fig.patch.set_facecolor('white')
        
        # Dessiner le réseau
        self._draw_network(layers_config, title)
        
        # Intégrer dans tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def _draw_network(self, layers_config, title):
        """Dessiner l'architecture du réseau"""
        self.ax.clear()
        self.ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        n_layers = len(layers_config)
        max_neurons = max(layers_config)
        
        # Paramètres de dessin
        layer_spacing = 2.0
        neuron_radius = 0.3
        max_display_neurons = 10  # Limite d'affichage pour éviter l'encombrement
        
        # Couleurs pour chaque type de couche
        layer_colors = {
            'input': '#3498db',      # Bleu pour l'entrée
            'hidden': '#2ecc71',     # Vert pour les couches cachées
            'output': '#e74c3c'      # Rouge pour la sortie
        }
        
        # Dessiner chaque couche
        for layer_idx, n_neurons in enumerate(layers_config):
            x = layer_idx * layer_spacing
            
            # Déterminer le type de couche
            if layer_idx == 0:
                layer_type = 'input'
                layer_name = 'Entrée'
            elif layer_idx == n_layers - 1:
                layer_type = 'output'
                layer_name = 'Sortie'
            else:
                layer_type = 'hidden'
                layer_name = f'Cachée {layer_idx}'
            
            color = layer_colors[layer_type]
            
            # Calculer les positions Y des neurones
            if n_neurons <= max_display_neurons:
                # Afficher tous les neurones
                y_positions = np.linspace(0, max_neurons - 1, n_neurons)
                display_neurons = n_neurons
                show_dots = False
            else:
                # Afficher seulement quelques neurones avec des points de suspension
                display_neurons = min(6, max_display_neurons)
                y_positions = np.linspace(0, max_neurons - 1, display_neurons)
                show_dots = True
              # Dessiner les neurones visibles
            for i, y in enumerate(y_positions):
                if show_dots and i == display_neurons // 2:
                    # Dessiner des points de suspension au milieu
                    self.ax.text(x, y, '⋮', fontsize=20, ha='center', va='center', 
                               color=color, fontweight='bold')
                else:
                    circle = plt.Circle((x, y), neuron_radius, facecolor=color, 
                                      alpha=0.7, linewidth=2, edgecolor='darkblue')
                    self.ax.add_patch(circle)
                    
                    # Numéroter les neurones pour les petites couches
                    if n_neurons <= 5:
                        neuron_num = i + 1 if not show_dots or i < display_neurons // 2 else i + (n_neurons - display_neurons) + 1
                        self.ax.text(x, y, str(neuron_num), fontsize=8, ha='center', va='center', 
                                   color='white', fontweight='bold')
            
            # Ajouter le label de la couche
            self.ax.text(x, max_neurons + 0.5, f'{layer_name}\n({n_neurons} neurones)', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.3))
            
            # Dessiner les connexions vers la couche suivante
            if layer_idx < n_layers - 1:
                self._draw_connections(x, x + layer_spacing, 
                                     layers_config[layer_idx], 
                                     layers_config[layer_idx + 1],
                                     max_neurons, max_display_neurons)
        
        # Ajouter des informations sur le réseau
        self._add_network_info(layers_config, max_neurons)
        
        # Configuration des axes
        self.ax.set_xlim(-0.5, (n_layers - 1) * layer_spacing + 0.5)
        self.ax.set_ylim(-1, max_neurons + 1.5)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
    def _draw_connections(self, x1, x2, n_neurons1, n_neurons2, max_neurons, max_display):
        """Dessiner les connexions entre deux couches"""
        # Limiter le nombre de connexions affichées pour éviter l'encombrement
        max_connections = 20
        
        # Positions des neurones de la première couche
        if n_neurons1 <= max_display:
            y1_positions = np.linspace(0, max_neurons - 1, n_neurons1)
        else:
            y1_positions = np.linspace(0, max_neurons - 1, min(6, max_display))
            
        # Positions des neurones de la deuxième couche
        if n_neurons2 <= max_display:
            y2_positions = np.linspace(0, max_neurons - 1, n_neurons2)
        else:
            y2_positions = np.linspace(0, max_neurons - 1, min(6, max_display))
        
        # Calculer le nombre total de connexions théoriques
        total_connections = n_neurons1 * n_neurons2
        
        # Si trop de connexions, n'en afficher qu'un échantillon
        if total_connections > max_connections:
            # Afficher seulement quelques connexions représentatives
            sample_connections = min(max_connections, len(y1_positions) * len(y2_positions))
            connection_alpha = 0.1
        else:
            connection_alpha = 0.3
            
        # Dessiner les connexions
        connections_drawn = 0
        for y1 in y1_positions:
            for y2 in y2_positions:
                if connections_drawn >= max_connections:
                    break
                    
                self.ax.plot([x1 + 0.3, x2 - 0.3], [y1, y2], 
                           color='gray', alpha=connection_alpha, linewidth=0.5)
                connections_drawn += 1
                
            if connections_drawn >= max_connections:
                break
        
        # Ajouter une note si toutes les connexions ne sont pas affichées
        if total_connections > max_connections:
            mid_y = (max(y1_positions) + min(y1_positions)) / 2
            self.ax.text((x1 + x2) / 2, mid_y - 1, 
                        f'({total_connections} connexions\ntotales)', 
                        ha='center', va='center', fontsize=8, 
                        style='italic', color='gray',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    def _add_network_info(self, layers_config, max_neurons):
        """Ajouter des informations sur le réseau"""
        total_neurons = sum(layers_config)
        total_params = 0
        
        # Calculer le nombre total de paramètres
        for i in range(len(layers_config) - 1):
            # Poids + biais
            total_params += layers_config[i] * layers_config[i + 1] + layers_config[i + 1]
        
        info_text = f"""Informations du réseau:
• Total neurones: {total_neurons}
• Total paramètres: {total_params:,}
• Couches: {len(layers_config)}
• Architecture: {' → '.join(map(str, layers_config))}"""
        
        self.ax.text(-0.3, -0.5, info_text, fontsize=10, va='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.3))

class TrainingProgressVisualizer:
    """Classe pour visualiser les progrès d'entraînement en temps réel"""
    
    def __init__(self, parent_frame, colors):
        self.parent_frame = parent_frame
        self.colors = colors
        self.fig = None
        self.canvas = None
        self.ax_loss = None
        self.ax_pred = None
        self.losses = []
        self.epochs = []
        
    def create_visualization(self):
        """Créer la visualisation des progrès d'entraînement"""
        # Nettoyer le frame parent
        for widget in self.parent_frame.winfo_children():
            widget.destroy()
            
        # Créer la figure avec subplots
        self.fig, (self.ax_loss, self.ax_pred) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.patch.set_facecolor('white')
        
        # Initialiser les graphiques
        self._setup_loss_plot()
        self._setup_prediction_plot()
        
        # Intégrer dans tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def _setup_loss_plot(self):
        """Configurer le graphique de la perte"""
        self.ax_loss.set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
        self.ax_loss.set_xlabel('Époque')
        self.ax_loss.set_ylabel('Loss')
        self.ax_loss.grid(True, alpha=0.3)
        self.ax_loss.set_xlim(0, 100)  # Initial range
        self.ax_loss.set_ylim(0, 1)    # Initial range
        
    def _setup_prediction_plot(self):
        """Configurer le graphique des prédictions"""
        self.ax_pred.set_title('Prédictions vs Réalité', fontsize=14, fontweight='bold')
        self.ax_pred.set_xlabel('Valeurs réelles')
        self.ax_pred.set_ylabel('Valeurs prédites')
        self.ax_pred.grid(True, alpha=0.3)
        
    def update_loss(self, epoch, loss):
        """Mettre à jour le graphique de la perte"""
        self.epochs.append(epoch)
        self.losses.append(loss)
        
        # Mettre à jour le graphique
        self.ax_loss.clear()
        self._setup_loss_plot()
        
        if len(self.losses) > 1:
            self.ax_loss.plot(self.epochs, self.losses, 'b-', linewidth=2, label='Loss')
            self.ax_loss.legend()
            
            # Ajuster les limites
            self.ax_loss.set_xlim(0, max(self.epochs) + 10)
            self.ax_loss.set_ylim(0, max(self.losses) * 1.1)
            
            # Ajouter la valeur actuelle
            self.ax_loss.text(epoch, loss, f'{loss:.4f}', ha='center', va='bottom',
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow', alpha=0.7))
        
        self.canvas.draw()
        
    def update_predictions(self, y_true, y_pred):
        """Mettre à jour le graphique des prédictions"""
        self.ax_pred.clear()
        self._setup_prediction_plot()
        
        if len(y_true) > 0 and len(y_pred) > 0:
            # Scatter plot des prédictions
            self.ax_pred.scatter(y_true, y_pred, alpha=0.6, color=self.colors["primary"])
            
            # Ligne de prédiction parfaite
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            self.ax_pred.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Prédiction parfaite')
            
            # Calculer R²
            try:
                from neural_network.utils import r2_score
                r2 = r2_score(y_true, y_pred)
                self.ax_pred.text(0.05, 0.95, f'R² = {r2:.3f}', transform=self.ax_pred.transAxes,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7),
                                fontsize=12, fontweight='bold')
            except:
                pass
                
            self.ax_pred.legend()
            
        self.canvas.draw()
        
    def reset(self):
        """Réinitialiser les données de visualisation"""
        self.losses = []
        self.epochs = []
        if self.ax_loss:
            self.ax_loss.clear()
            self._setup_loss_plot()
        if self.ax_pred:
            self.ax_pred.clear()
            self._setup_prediction_plot()
        if self.canvas:
            self.canvas.draw()

class InteractiveNetworkDesigner:
    """Interface interactive pour concevoir l'architecture d'un réseau"""
    
    def __init__(self, parent_frame, colors, callback=None):
        self.parent_frame = parent_frame
        self.colors = colors
        self.callback = callback  # Fonction à appeler quand l'architecture change
        self.layers_config = [2, 5, 1]  # Configuration par défaut
        self.layer_widgets = []
        self.create_interface()
        
    def create_interface(self):
        """Créer l'interface de conception"""
        # Frame principal
        main_frame = tk.Frame(self.parent_frame, bg=self.colors["bg_white"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Titre
        title_label = tk.Label(
            main_frame,
            text="🔧 Concepteur d'Architecture",
            font=('Helvetica', 14, 'bold'),
            bg=self.colors["bg_white"],
            fg=self.colors["primary"]
        )
        title_label.pack(pady=(10, 20))
        
        # Frame pour les contrôles
        controls_frame = tk.Frame(main_frame, bg=self.colors["bg_white"])
        controls_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Boutons pour ajouter/supprimer des couches
        tk.Button(
            controls_frame,
            text="➕ Ajouter couche cachée",
            command=self.add_hidden_layer,
            bg=self.colors["success"],
            fg="white",
            font=('Helvetica', 10, 'bold'),
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            controls_frame,
            text="➖ Supprimer couche",
            command=self.remove_hidden_layer,
            bg=self.colors["accent"],
            fg="white",
            font=('Helvetica', 10, 'bold'),
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            controls_frame,
            text="🔄 Réinitialiser",
            command=self.reset_architecture,
            bg=self.colors["primary"],
            fg="white",
            font=('Helvetica', 10, 'bold'),
            cursor="hand2"
        ).pack(side=tk.LEFT)
        
        # Frame pour les couches
        self.layers_frame = tk.Frame(main_frame, bg=self.colors["bg_white"])
        self.layers_frame.pack(fill=tk.BOTH, expand=True)
        
        # Créer les widgets pour les couches initiales
        self.create_layer_widgets()
        
    def create_layer_widgets(self):
        """Créer les widgets pour configurer chaque couche"""
        # Nettoyer les widgets existants
        for widget in self.layers_frame.winfo_children():
            widget.destroy()
        self.layer_widgets = []
        
        for i, n_neurons in enumerate(self.layers_config):
            layer_frame = tk.LabelFrame(
                self.layers_frame,
                text=self.get_layer_name(i),
                font=('Helvetica', 11, 'bold'),
                bg=self.colors["bg_white"],
                fg=self.colors["text"],
                padx=15,
                pady=10
            )
            layer_frame.pack(fill=tk.X, pady=(0, 10))
            
            # Widgets pour cette couche
            controls_frame = tk.Frame(layer_frame, bg=self.colors["bg_white"])
            controls_frame.pack(fill=tk.X)
            
            # Label et slider pour le nombre de neurones
            tk.Label(
                controls_frame,
                text="Nombre de neurones:",
                bg=self.colors["bg_white"],
                fg=self.colors["text"],
                font=('Helvetica', 10)
            ).pack(side=tk.LEFT, padx=(0, 10))
            
            # Variables et contraintes selon le type de couche
            if i == 0:  # Couche d'entrée
                min_neurons, max_neurons = 1, 20
                default_val = n_neurons
            elif i == len(self.layers_config) - 1:  # Couche de sortie
                min_neurons, max_neurons = 1, 5
                default_val = n_neurons
            else:  # Couches cachées
                min_neurons, max_neurons = 1, 50
                default_val = n_neurons
            
            var = tk.IntVar(value=default_val)
            scale = tk.Scale(
                controls_frame,
                from_=min_neurons,
                to=max_neurons,
                orient=tk.HORIZONTAL,
                variable=var,
                bg=self.colors["bg_white"],
                fg=self.colors["text"],
                length=200,
                command=lambda val, idx=i: self.update_layer(idx, int(val))
            )
            scale.pack(side=tk.LEFT, padx=(0, 10))
            
            # Label pour afficher la valeur actuelle
            value_label = tk.Label(
                controls_frame,
                text=str(default_val),
                bg=self.colors["bg_white"],
                fg=self.colors["primary"],
                font=('Helvetica', 10, 'bold')
            )
            value_label.pack(side=tk.LEFT)
            
            self.layer_widgets.append({
                'frame': layer_frame,
                'var': var,
                'scale': scale,
                'label': value_label,
                'index': i
            })
            
        # Afficher les informations sur l'architecture
        self.update_architecture_info()
        
    def get_layer_name(self, index):
        """Obtenir le nom d'une couche selon son index"""
        if index == 0:
            return "🔵 Couche d'entrée"
        elif index == len(self.layers_config) - 1:
            return "🔴 Couche de sortie"
        else:
            return f"🟢 Couche cachée {index}"
            
    def update_layer(self, layer_index, n_neurons):
        """Mettre à jour le nombre de neurones d'une couche"""
        self.layers_config[layer_index] = n_neurons
        
        # Mettre à jour le label de valeur
        if layer_index < len(self.layer_widgets):
            self.layer_widgets[layer_index]['label'].config(text=str(n_neurons))
        
        # Mettre à jour les informations
        self.update_architecture_info()
        
        # Appeler le callback si défini
        if self.callback:
            self.callback(self.layers_config.copy())
            
    def add_hidden_layer(self):
        """Ajouter une couche cachée"""
        # Insérer une nouvelle couche cachée avant la sortie
        self.layers_config.insert(-1, 5)  # 5 neurones par défaut
        self.create_layer_widgets()
        
        if self.callback:
            self.callback(self.layers_config.copy())
            
    def remove_hidden_layer(self):
        """Supprimer la dernière couche cachée"""
        if len(self.layers_config) > 2:  # Garder au moins entrée + sortie
            self.layers_config.pop(-2)  # Supprimer l'avant-dernière (dernière cachée)
            self.create_layer_widgets()
            
            if self.callback:
                self.callback(self.layers_config.copy())
                
    def reset_architecture(self):
        """Réinitialiser l'architecture par défaut"""
        self.layers_config = [2, 5, 1]
        self.create_layer_widgets()
        
        if self.callback:
            self.callback(self.layers_config.copy())
            
    def update_architecture_info(self):
        """Mettre à jour les informations sur l'architecture"""
        # Chercher s'il y a déjà un label d'info
        info_label = None
        for widget in self.layers_frame.winfo_children():
            if hasattr(widget, 'info_label_marker'):
                info_label = widget
                break
                
        # Calculer les informations
        total_neurons = sum(self.layers_config)
        total_params = 0
        for i in range(len(self.layers_config) - 1):
            total_params += self.layers_config[i] * self.layers_config[i + 1] + self.layers_config[i + 1]
        
        info_text = f"📊 Total: {total_neurons} neurones | {total_params:,} paramètres | Architecture: {' → '.join(map(str, self.layers_config))}"
        
        if info_label:
            info_label.config(text=info_text)
        else:
            info_label = tk.Label(
                self.layers_frame,
                text=info_text,
                bg=self.colors["bg_white"],
                fg=self.colors["text"],
                font=('Helvetica', 9),
                wraplength=600
            )
            info_label.info_label_marker = True  # Marqueur pour identification
            info_label.pack(pady=(10, 0))
            
    def get_architecture(self):
        """Obtenir l'architecture actuelle"""
        return self.layers_config.copy()
        
    def set_architecture(self, layers_config):
        """Définir une nouvelle architecture"""
        self.layers_config = layers_config.copy()
        self.create_layer_widgets()
