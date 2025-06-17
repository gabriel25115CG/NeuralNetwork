import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os

# Ajouter le chemin vers neural_network
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_network.network_visualizer import NetworkArchitectureVisualizer, TrainingProgressVisualizer, InteractiveNetworkDesigner
from neural_network.network import NeuralNetwork

class NetworkVisualizationPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.current_architecture = [2, 5, 1]
        self.current_network = None
        self.architecture_visualizer = None
        self.training_visualizer = None
        self.network_designer = None
        self.create_widgets()
        
    def create_widgets(self):
        # Cadre principal
        main_frame = tk.Frame(self, bg=self.controller.colors["bg_light"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # En-tête
        header_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Button(
            header_frame,
            text="← Retour à l'accueil",
            command=lambda: self.controller.show_page("HomePage"),
            style='TButton'
        ).pack(side=tk.LEFT)
        
        # Titre
        title_label = tk.Label(
            main_frame,
            text="🧠 Visualisateur de Réseaux de Neurones",
            font=('Helvetica', 18, 'bold'),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["primary"]
        )
        title_label.pack(pady=(0, 20))
        
        # Notebook pour les onglets
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Créer les onglets
        self.create_architecture_tab()
        self.create_designer_tab()
        self.create_training_tab()
        self.create_comparison_tab()
        
    def create_architecture_tab(self):
        """Créer l'onglet de visualisation d'architecture"""
        arch_frame = tk.Frame(self.notebook, bg=self.controller.colors["bg_white"])
        self.notebook.add(arch_frame, text="🏗️ Architecture")
        
        # Section de contrôles
        controls_frame = tk.LabelFrame(
            arch_frame,
            text="Configuration rapide",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=15,
            pady=10
        )
        controls_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Architectures prédéfinies
        tk.Label(
            controls_frame,
            text="Architectures prédéfinies:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(5, 10))
        
        presets_frame = tk.Frame(controls_frame, bg=self.controller.colors["bg_white"])
        presets_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Boutons pour les architectures prédéfinies
        presets = [
            ("Simple", [2, 3, 1]),
            ("Classique", [3, 5, 3, 1]),
            ("Profond", [4, 10, 8, 5, 1]),
            ("Large", [5, 20, 15, 10, 1]),
            ("XOR", [2, 4, 1]),
            ("Multi-sortie", [3, 8, 6, 3])
        ]
        
        for name, config in presets:
            tk.Button(
                presets_frame,
                text=name,
                command=lambda c=config: self.set_architecture(c),
                bg=self.controller.colors["primary"],
                fg="white",
                font=('Helvetica', 9, 'bold'),
                cursor="hand2",
                padx=10,
                pady=5
            ).pack(side=tk.LEFT, padx=(0, 5))
        
        # Configuration manuelle
        manual_frame = tk.Frame(controls_frame, bg=self.controller.colors["bg_white"])
        manual_frame.pack(fill=tk.X, pady=(10, 0))
        
        tk.Label(
            manual_frame,
            text="Configuration manuelle (séparée par des virgules):",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(0, 5))
        
        config_frame = tk.Frame(manual_frame, bg=self.controller.colors["bg_white"])
        config_frame.pack(fill=tk.X)
        
        self.config_entry = tk.Entry(
            config_frame,
            font=('Helvetica', 11),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            width=30
        )
        self.config_entry.pack(side=tk.LEFT, padx=(0, 10))
        self.config_entry.insert(0, "2, 5, 1")
        
        tk.Button(
            config_frame,
            text="Appliquer",
            command=self.apply_manual_config,
            bg=self.controller.colors["success"],
            fg="white",
            font=('Helvetica', 10, 'bold'),
            cursor="hand2"
        ).pack(side=tk.LEFT)
        
        # Frame pour la visualisation
        self.arch_viz_frame = tk.Frame(arch_frame, bg=self.controller.colors["bg_white"])
        self.arch_viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Initialiser la visualisation
        self.architecture_visualizer = NetworkArchitectureVisualizer(
            self.arch_viz_frame, 
            self.controller.colors
        )
        self.update_architecture_visualization()
        
    def create_designer_tab(self):
        """Créer l'onglet de conception interactive"""
        designer_frame = tk.Frame(self.notebook, bg=self.controller.colors["bg_white"])
        self.notebook.add(designer_frame, text="🎨 Concepteur")
        
        # Créer un PanedWindow pour diviser l'espace
        paned = tk.PanedWindow(designer_frame, orient=tk.HORIZONTAL, bg=self.controller.colors["bg_white"])
        paned.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Frame gauche pour le concepteur
        designer_left = tk.Frame(paned, bg=self.controller.colors["bg_white"], width=400)
        paned.add(designer_left)
        
        # Frame droite pour la visualisation
        designer_right = tk.Frame(paned, bg=self.controller.colors["bg_white"])
        paned.add(designer_right)
        
        # Créer le concepteur interactif
        self.network_designer = InteractiveNetworkDesigner(
            designer_left, 
            self.controller.colors,
            callback=self.on_architecture_changed
        )
        
        # Créer la visualisation mise à jour automatiquement
        self.designer_viz_frame = tk.Frame(designer_right, bg=self.controller.colors["bg_white"])
        self.designer_viz_frame.pack(fill=tk.BOTH, expand=True)
        
        self.designer_visualizer = NetworkArchitectureVisualizer(
            self.designer_viz_frame,
            self.controller.colors
        )
        
    def create_training_tab(self):
        """Créer l'onglet de visualisation d'entraînement"""
        training_frame = tk.Frame(self.notebook, bg=self.controller.colors["bg_white"])
        self.notebook.add(training_frame, text="📈 Entraînement")
        
        # Section de contrôles
        controls_frame = tk.LabelFrame(
            training_frame,
            text="Simulation d'entraînement",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=15,
            pady=10
        )
        controls_frame.pack(fill=tk.X, padx=20, pady=20)
        
        # Paramètres de simulation
        params_frame = tk.Frame(controls_frame, bg=self.controller.colors["bg_white"])
        params_frame.pack(fill=tk.X, pady=(10, 15))
        
        # Taux d'apprentissage
        tk.Label(
            params_frame,
            text="Taux d'apprentissage:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.lr_var = tk.DoubleVar(value=0.01)
        lr_scale = tk.Scale(
            params_frame,
            from_=0.001,
            to=0.1,
            resolution=0.001,
            orient=tk.HORIZONTAL,
            variable=self.lr_var,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            length=200
        )
        lr_scale.grid(row=0, column=1, sticky="w")
        
        # Époques
        tk.Label(
            params_frame,
            text="Époques:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(10, 0))
        
        self.epochs_var = tk.IntVar(value=100)
        epochs_scale = tk.Scale(
            params_frame,
            from_=10,
            to=500,
            resolution=10,
            orient=tk.HORIZONTAL,
            variable=self.epochs_var,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            length=200
        )
        epochs_scale.grid(row=1, column=1, sticky="w", pady=(10, 0))
        
        # Boutons
        buttons_frame = tk.Frame(controls_frame, bg=self.controller.colors["bg_white"])
        buttons_frame.pack(fill=tk.X, pady=(15, 0))
        
        tk.Button(
            buttons_frame,
            text="🚀 Simuler Entraînement",
            command=self.simulate_training,
            bg=self.controller.colors["success"],
            fg="white",
            font=('Helvetica', 11, 'bold'),
            cursor="hand2",
            padx=15
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            buttons_frame,
            text="⏹️ Arrêter",
            command=self.stop_simulation,
            bg=self.controller.colors["accent"],
            fg="white",
            font=('Helvetica', 11, 'bold'),
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Button(
            buttons_frame,
            text="🔄 Réinitialiser",
            command=self.reset_training_viz,
            bg=self.controller.colors["primary"],
            fg="white",
            font=('Helvetica', 11, 'bold'),
            cursor="hand2"
        ).pack(side=tk.LEFT)
        
        # Frame pour la visualisation d'entraînement
        self.training_viz_frame = tk.Frame(training_frame, bg=self.controller.colors["bg_white"])
        self.training_viz_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Initialiser la visualisation d'entraînement
        self.training_visualizer = TrainingProgressVisualizer(
            self.training_viz_frame,
            self.controller.colors
        )
        self.training_visualizer.create_visualization()
        
        # Variable pour contrôler la simulation
        self.simulation_running = False
        
    def create_comparison_tab(self):
        """Créer l'onglet de comparaison d'architectures"""
        comparison_frame = tk.Frame(self.notebook, bg=self.controller.colors["bg_white"])
        self.notebook.add(comparison_frame, text="⚖️ Comparaison")
        
        # Instructions
        instructions = """
        Comparez différentes architectures de réseaux de neurones:
        
        1. Ajoutez des architectures à comparer
        2. Visualisez leurs caractéristiques
        3. Comparez leurs performances théoriques
        4. Analysez la complexité computationnelle
        """
        
        tk.Label(
            comparison_frame,
            text=instructions,
            font=('Helvetica', 11),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            justify="left"
        ).pack(pady=20, padx=20)
        
        # Section d'ajout d'architectures
        add_frame = tk.LabelFrame(
            comparison_frame,
            text="Ajouter une architecture",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=15,
            pady=10
        )
        add_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        input_frame = tk.Frame(add_frame, bg=self.controller.colors["bg_white"])
        input_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            input_frame,
            text="Architecture:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        self.comp_entry = tk.Entry(
            input_frame,
            font=('Helvetica', 11),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            width=20
        )
        self.comp_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        tk.Entry(
            input_frame,
            font=('Helvetica', 11),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            width=15
        ).pack(side=tk.LEFT, padx=(0, 10))  # Nom de l'architecture
        
        tk.Button(
            input_frame,
            text="Ajouter",
            command=self.add_architecture_to_comparison,
            bg=self.controller.colors["primary"],
            fg="white",
            font=('Helvetica', 10, 'bold'),
            cursor="hand2"
        ).pack(side=tk.LEFT)
        
        # Tableau de comparaison
        self.comparison_table_frame = tk.Frame(comparison_frame, bg=self.controller.colors["bg_white"])
        self.comparison_table_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Créer le tableau initial
        self.create_comparison_table()
        
    def set_architecture(self, layers_config):
        """Définir une nouvelle architecture"""
        self.current_architecture = layers_config.copy()
        self.config_entry.delete(0, tk.END)
        self.config_entry.insert(0, ", ".join(map(str, layers_config)))
        self.update_architecture_visualization()
        
        # Mettre à jour le concepteur si il existe
        if self.network_designer:
            self.network_designer.set_architecture(layers_config)
            
    def apply_manual_config(self):
        """Appliquer la configuration manuelle"""
        try:
            config_str = self.config_entry.get()
            layers = [int(x.strip()) for x in config_str.split(',')]
            
            if len(layers) < 2:
                raise ValueError("Au moins 2 couches nécessaires")
            if any(n <= 0 for n in layers):
                raise ValueError("Tous les nombres doivent être positifs")
                
            self.set_architecture(layers)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Configuration invalide:\n{str(e)}")
            
    def update_architecture_visualization(self):
        """Mettre à jour la visualisation de l'architecture"""
        if self.architecture_visualizer:
            title = f"Architecture: {' → '.join(map(str, self.current_architecture))}"
            self.architecture_visualizer.create_visualization(
                self.current_architecture, 
                title
            )
            
    def on_architecture_changed(self, new_architecture):
        """Callback appelé quand l'architecture change dans le concepteur"""
        self.current_architecture = new_architecture.copy()
        
        # Mettre à jour la visualisation du concepteur
        if hasattr(self, 'designer_visualizer'):
            title = f"Architecture: {' → '.join(map(str, new_architecture))}"
            self.designer_visualizer.create_visualization(new_architecture, title)
            
    def simulate_training(self):
        """Simuler un entraînement pour la visualisation"""
        if self.simulation_running:
            return
            
        self.simulation_running = True
        
        # Réinitialiser la visualisation
        self.training_visualizer.reset()
        
        # Simuler des données d'entraînement
        import random
        import math
        
        epochs = self.epochs_var.get()
        learning_rate = self.lr_var.get()
        
        # Générer des données fictives
        n_samples = 50
        X_data = [[random.uniform(0, 1), random.uniform(0, 1)] for _ in range(n_samples)]
        y_data = [x[0] * 0.5 + x[1] * 0.3 + random.uniform(-0.1, 0.1) for x in X_data]
        
        # Simuler l'entraînement epoch par epoch
        def simulate_epoch(epoch):
            if not self.simulation_running or epoch >= epochs:
                self.simulation_running = False
                return
                
            # Simuler une loss qui diminue
            base_loss = 1.0 * math.exp(-epoch * learning_rate * 2)
            noise = random.uniform(-0.05, 0.05)
            loss = max(0.001, base_loss + noise)
            
            # Mettre à jour la visualisation de la loss
            self.training_visualizer.update_loss(epoch + 1, loss)
            
            # Tous les 10 epochs, mettre à jour les prédictions
            if (epoch + 1) % 10 == 0:
                # Simuler des prédictions qui s'améliorent
                noise_factor = max(0.1, 1.0 - epoch / epochs)
                y_pred = [y + random.uniform(-noise_factor, noise_factor) for y in y_data]
                self.training_visualizer.update_predictions(y_data, y_pred)
            
            # Programmer la prochaine epoch
            self.after(50, lambda: simulate_epoch(epoch + 1))
            
        # Commencer la simulation
        simulate_epoch(0)
        
    def stop_simulation(self):
        """Arrêter la simulation d'entraînement"""
        self.simulation_running = False
        
    def reset_training_viz(self):
        """Réinitialiser la visualisation d'entraînement"""
        self.stop_simulation()
        if self.training_visualizer:
            self.training_visualizer.reset()
            
    def add_architecture_to_comparison(self):
        """Ajouter une architecture à la comparaison"""
        try:
            config_str = self.comp_entry.get()
            layers = [int(x.strip()) for x in config_str.split(',')]
            
            if len(layers) < 2:
                raise ValueError("Au moins 2 couches nécessaires")
            if any(n <= 0 for n in layers):
                raise ValueError("Tous les nombres doivent être positifs")
                
            # Ajouter à la table de comparaison
            self.update_comparison_table(layers)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Architecture invalide:\n{str(e)}")
            
    def create_comparison_table(self):
        """Créer le tableau de comparaison"""
        # Headers
        headers = ['Architecture', 'Neurones', 'Paramètres', 'Couches', 'Complexité']
        
        # Créer le treeview
        self.comparison_tree = ttk.Treeview(
            self.comparison_table_frame, 
            columns=headers, 
            show='headings',
            height=10
        )
        
        # Configurer les colonnes
        for header in headers:
            self.comparison_tree.heading(header, text=header)
            self.comparison_tree.column(header, width=150, anchor='center')
            
        self.comparison_tree.pack(fill=tk.BOTH, expand=True)
        
        # Ajouter quelques exemples
        examples = [
            [2, 3, 1],
            [3, 5, 3, 1],
            [4, 10, 8, 5, 1]
        ]
        
        for arch in examples:
            self.update_comparison_table(arch)
            
    def update_comparison_table(self, layers_config):
        """Mettre à jour le tableau de comparaison"""
        arch_str = ' → '.join(map(str, layers_config))
        total_neurons = sum(layers_config)
        
        # Calculer les paramètres
        total_params = 0
        for i in range(len(layers_config) - 1):
            total_params += layers_config[i] * layers_config[i + 1] + layers_config[i + 1]
            
        n_layers = len(layers_config)
        
        # Estimer la complexité
        if total_params < 100:
            complexity = "Faible"
        elif total_params < 1000:
            complexity = "Moyenne"
        elif total_params < 10000:
            complexity = "Élevée"
        else:
            complexity = "Très élevée"
            
        # Ajouter à la table
        self.comparison_tree.insert('', 'end', values=(
            arch_str,
            total_neurons,
            f"{total_params:,}",
            n_layers,
            complexity
        ))
