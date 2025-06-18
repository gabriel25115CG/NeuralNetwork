import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
import numpy as np

class CreateModelPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
          # Variables pour l'architecture du réseau
        self.layers_config = [5, 12, 8, 1]  # Configuration optimisée pour l'immobilier : [entrée(5 vars), hidden1(12), hidden2(8), sortie(1 prix)]self.network_fig = None
        self.network_canvas = None
        self.network_ax = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Cadre principal avec scroll
        main_frame = tk.Frame(self, bg=self.controller.colors["bg_light"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # En-tête avec bouton retour
        header_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Bouton retour
        back_button = ttk.Button(
            header_frame,
            text="← Retour à l'accueil",
            command=lambda: self.controller.show_page("HomePage"),
            style='TButton',
            cursor="hand2"
        )
        back_button.pack(side=tk.LEFT)
          # Titre de la page
        title_label = tk.Label(
            main_frame,
            text="🧠 Créer un nouveau modèle neuronal",
            font=('Helvetica', 18, 'bold'),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["primary"]
        )
        title_label.pack(pady=(0, 10))
          # Sous-titre explicatif
        subtitle_label = tk.Label(
            main_frame,
            text="Réseau de neurones optimisé pour les données complexes (immobilier, relations non-linéaires)",
            font=('Helvetica', 11, 'italic'),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["text"]
        )
        subtitle_label.pack(pady=(0, 20))
        
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
        
        # Ajuster la largeur
        def on_canvas_configure(event):
            canvas_width = event.width
            self.main_canvas.itemconfig(self.main_canvas.find_all()[0], width=canvas_width-20)
        
        self.main_canvas.bind('<Configure>', on_canvas_configure)
        
        # Gérer le scroll avec la molette
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        self.main_canvas.bind("<MouseWheel>", _on_mousewheel)
        
        self.main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Cadre principal du formulaire (maintenant dans le contenu scrollable)
        form_frame = tk.Frame(self.scrollable_content, bg=self.controller.colors["bg_white"], padx=40, pady=30)
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Section 1: Informations du modèle
        info_section = tk.LabelFrame(
            form_frame,
            text="Informations du modèle",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        info_section.pack(fill=tk.X, pady=(0, 20))
        
        # Nom du modèle
        tk.Label(
            info_section,
            text="Nom du modèle:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(10, 5))
        
        self.name_entry = tk.Entry(
            info_section,
            font=('Helvetica', 11),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            width=50
        )
        self.name_entry.pack(fill=tk.X, pady=(0, 15))
        self.name_entry.focus()
        
        # Description du modèle
        tk.Label(
            info_section,
            text="Description (optionnelle):",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(0, 5))
        
        self.description_text = tk.Text(
            info_section,
            height=3,
            font=('Helvetica', 11),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            wrap=tk.WORD
        )
        self.description_text.pack(fill=tk.X, pady=(0, 10))
        
        # Section 2: Données d'entraînement
        data_section = tk.LabelFrame(
            form_frame,
            text="Données d'entraînement",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        data_section.pack(fill=tk.X, pady=(0, 20))
        
        # Sélection du fichier de données
        file_frame = tk.Frame(data_section, bg=self.controller.colors["bg_white"])
        file_frame.pack(fill=tk.X, pady=(10, 15))
        
        tk.Label(
            file_frame,
            text="Fichier de données (CSV):",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(0, 5))
        
        file_select_frame = tk.Frame(file_frame, bg=self.controller.colors["bg_white"])
        file_select_frame.pack(fill=tk.X)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = tk.Entry(
            file_select_frame,
            textvariable=self.file_path_var,
            font=('Helvetica', 11),
            bg=self.controller.colors["bg_white"],
            fg="black",
            readonlybackground=self.controller.colors["bg_white"],
            state="readonly"
        )
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_button = ttk.Button(
            file_select_frame,
            text="Parcourir...",
            command=self.browse_file,
            style='TButton',
            cursor="hand2"
        )
        browse_button.pack(side=tk.RIGHT)
          # Informations sur le fichier
        self.file_info_label = tk.Label(
            data_section,
            text="Aucun fichier sélectionné",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10),
            justify="left"
        )
        self.file_info_label.pack(anchor="w", pady=(5, 10))
        
        # Recommandations pour les datasets
        recommendations_frame = tk.Frame(data_section, bg="#f8f9fa", relief="groove", bd=1)
        recommendations_frame.pack(fill=tk.X, pady=(5, 15))
        
        
    
        
        # Section 3: Paramètres du modèle
        params_section = tk.LabelFrame(
            form_frame,
            text="Paramètres du modèle",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15        )
        params_section.pack(fill=tk.X, pady=(0, 30))
        
        # Type de régression (fixé à Neuronal)
        regression_info_frame = tk.Frame(params_section, bg=self.controller.colors["bg_white"])
        regression_info_frame.pack(fill=tk.X, pady=(10, 15))
        
        tk.Label(
            regression_info_frame,
            text="🧠 Type de régression:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11, 'bold')
        ).pack(anchor="w", pady=(0, 5))
        
        # Affichage fixe du type neuronal
        neural_frame = tk.Frame(regression_info_frame, bg=self.controller.colors["primary"], relief="ridge", bd=2)
        neural_frame.pack(fill=tk.X, pady=(0, 5))
        
        tk.Label(
            neural_frame,
            text="✅ RÉSEAU DE NEURONES (MLP)",
            bg=self.controller.colors["primary"],
            fg="white",
            font=('Helvetica', 12, 'bold'),
            padx=15,
            pady=8
        ).pack()
        
        # Avantages du réseau neuronal
        advantages_text = """
• Capture les relations complexes et non-linéaires
• Optimal pour l'immobilier (interactions surface×pièces×localisation)
• Performances supérieures aux méthodes linéaires
• Architecture adaptable selon la complexité des données"""
        
        tk.Label(
            regression_info_frame,
            text=advantages_text,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 9),
            justify="left",
            anchor="w"
        ).pack(anchor="w", pady=(5, 0))
        
        # Stocker le type de régression (toujours neuronal)
        self.regression_type_var = tk.StringVar(value="Neuronal")
        
        # Section 4: Architecture du réseau de neurones
        network_section = tk.LabelFrame(
            form_frame,
            text="Architecture du réseau de neurones",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        network_section.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Frame principal pour l'architecture (vertical : contrôles puis visualisation)
        architecture_main_frame = tk.Frame(network_section, bg=self.controller.colors["bg_white"])
        architecture_main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Frame pour les contrôles (en haut)
        controls_frame = tk.Frame(architecture_main_frame, bg=self.controller.colors["bg_white"])
        controls_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 20))        # Contrôles pour les couches
        tk.Label(
            controls_frame,
            text="Configuration des couches:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11, 'bold')
        ).pack(anchor="w", pady=(10, 10))
        
      
        # Séparateur
        separator = tk.Frame(controls_frame, height=2, bg=self.controller.colors["primary"])
        separator.pack(fill=tk.X, pady=(10, 15))
        
        # Contrôles pour la couche d'entrée
        input_frame = tk.Frame(controls_frame, bg=self.controller.colors["bg_white"])
        input_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            input_frame,
            text="Neurones d'entrée:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT)
        
        self.input_neurons_var = tk.IntVar(value=self.layers_config[0])
        input_spinbox = tk.Spinbox(
            input_frame,
            from_=1,
            to=20,
            textvariable=self.input_neurons_var,
            width=5,
            command=self.update_input_neurons,
            font=('Helvetica', 10)
        )
        input_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        input_spinbox.bind('<Return>', lambda e: self.update_input_neurons())
        input_spinbox.bind('<FocusOut>', lambda e: self.update_input_neurons())
        
        # Contrôles pour la couche de sortie
        output_frame = tk.Frame(controls_frame, bg=self.controller.colors["bg_white"])
        output_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(
            output_frame,
            text="Neurones de sortie:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        ).pack(side=tk.LEFT)
        
        self.output_neurons_var = tk.IntVar(value=self.layers_config[-1])
        output_spinbox = tk.Spinbox(
            output_frame,
            from_=1,
            to=10,
            textvariable=self.output_neurons_var,
            width=5,
            command=self.update_output_neurons,            font=('Helvetica', 10)
        )
        output_spinbox.pack(side=tk.LEFT, padx=(10, 0))
        output_spinbox.bind('<Return>', lambda e: self.update_output_neurons())
        output_spinbox.bind('<FocusOut>', lambda e: self.update_output_neurons())
        
        # Label pour les couches cachées
        tk.Label(
            controls_frame,
            text="Couches cachées:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10, 'bold')
        ).pack(anchor="w", pady=(10, 5))
        
        # Frame pour la liste des couches
        layers_list_frame = tk.Frame(controls_frame, bg=self.controller.colors["bg_white"])
        layers_list_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Scrollable frame pour les couches
        self.layers_canvas = tk.Canvas(layers_list_frame, height=200, bg=self.controller.colors["bg_white"])
        self.layers_scrollbar = ttk.Scrollbar(layers_list_frame, orient="vertical", command=self.layers_canvas.yview)
        self.layers_frame = tk.Frame(self.layers_canvas, bg=self.controller.colors["bg_white"])
        
        self.layers_frame.bind(
            "<Configure>",
            lambda e: self.layers_canvas.configure(scrollregion=self.layers_canvas.bbox("all"))
        )
        
        self.layers_canvas.create_window((0, 0), window=self.layers_frame, anchor="nw")
        self.layers_canvas.configure(yscrollcommand=self.layers_scrollbar.set)
        
        self.layers_canvas.pack(side="left", fill="both", expand=True)
        self.layers_scrollbar.pack(side="right", fill="y")
        
        # Boutons de contrôle des couches
        buttons_frame = tk.Frame(controls_frame, bg=self.controller.colors["bg_white"])
        buttons_frame.pack(fill=tk.X, pady=(0, 10))
        
        add_layer_btn = ttk.Button(
            buttons_frame,
            text="+ Ajouter couche",
            command=self.add_layer,
            style='Add.TButton',
            cursor="hand2"
        )
        add_layer_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        remove_layer_btn = ttk.Button(
            buttons_frame,
            text="- Supprimer couche",
            command=self.remove_layer,
            cursor="hand2"
        )
        remove_layer_btn.pack(side=tk.LEFT)
        
        # Bouton pour réinitialiser l'architecture
        reset_btn = ttk.Button(
            controls_frame,
            text="Réinitialiser architecture",
            command=self.reset_architecture,
            cursor="hand2"
        )
        reset_btn.pack(pady=(10, 0))
        
        # Frame pour la visualisation (en bas, occupe toute la largeur)
        viz_frame = tk.Frame(architecture_main_frame, bg=self.controller.colors["bg_white"])
        viz_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        tk.Label(
            viz_frame,
            text="Visualisation du réseau:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11, 'bold')
        ).pack(anchor="w", pady=(10, 5))
        
        # Frame pour la visualisation matplotlib avec hauteur minimale
        self.network_viz_frame = tk.Frame(viz_frame, bg=self.controller.colors["bg_white"], relief=tk.SUNKEN, bd=1)
        self.network_viz_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.network_viz_frame.configure(height=520)  # Hauteur minimale pour assurer la visibilité
        
        # Initialiser la visualisation et l'interface des couches
        self.update_layers_interface()
        self.update_network_visualization()
        
        # Boutons d'action
        button_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Bouton Annuler
        cancel_button = ttk.Button(
            button_frame,
            text="Annuler",
            command=lambda: self.controller.show_page("HomePage"),
            style='TButton',
            cursor="hand2"
        )
        cancel_button.pack(side=tk.RIGHT, padx=(10, 0))
        
        # Bouton Créer le modèle
        create_button = ttk.Button(
            button_frame,
            text="Prévisualiser les données",
            command=self.preview_data,
            style='Add.TButton',
            cursor="hand2"        )
        create_button.pack(side=tk.RIGHT)

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            title="Sélectionner un fichier de données",
            filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
            filename = os.path.basename(file_path)
            try:
                file_size = os.path.getsize(file_path)
                size_mb = file_size / (1024 * 1024)
                self.file_info_label.config(
                    text=f"Fichier: {filename}\nTaille: {size_mb:.2f} MB",
                    fg=self.controller.colors["success"]
                )
            except:
                self.file_info_label.config(
                    text=f"Fichier: {filename}",
                    fg=self.controller.colors["success"]
                )

    def preview_data(self):
        """Aller à la prévisualisation des données"""
        model_name = self.name_entry.get().strip()
        file_path = self.file_path_var.get().strip()
        
        if not model_name:
            messagebox.showwarning("Attention", "Veuillez entrer un nom pour le modèle")
            return
            
        if not file_path:
            messagebox.showwarning("Attention", "Veuillez sélectionner un fichier de données")
            return
        
        # Vérifier si le nom existe déjà
        for model in self.controller.model_list:
            # model_list contient des chaînes de caractères (noms de modèles)
            if isinstance(model, str):
                if model.lower() == model_name.lower():
                    messagebox.showwarning("Attention", "Un modèle avec ce nom existe déjà")
                    return
            elif isinstance(model, dict) and "name" in model:
                if model["name"].lower() == model_name.lower():
                    messagebox.showwarning("Attention", "Un modèle avec ce nom existe déjà")
                    return        # Préparer les informations du modèle
        model_info = {
            "name": model_name,
            "data_file": os.path.basename(file_path),
            "full_path": file_path,
            "description": self.description_text.get("1.0", tk.END).strip(),
            "regression_type": self.regression_type_var.get(),
            "network_architecture": self.layers_config.copy(),
            "total_parameters": self._calculate_total_parameters(),
            "accuracy": "En cours..."
        }
        
        # Debug: afficher l'architecture envoyée
        print(f"🔍 DEBUG: Architecture envoyée depuis create_model_page: {self.layers_config}")
        
        # Aller à la page de prévisualisation
        self.controller.show_data_preview(file_path, model_info)
    
    def create_model(self):
        model_name = self.name_entry.get().strip()
        file_path = self.file_path_var.get().strip()
        
        if not model_name:
            messagebox.showwarning("Attention", "Veuillez entrer un nom pour le modèle")
            return
            
        if not file_path:
            messagebox.showwarning("Attention", "Veuillez sélectionner un fichier de données")
            return
          # Vérifier si le nom existe déjà
        for model in self.controller.model_list:
            # model_list contient des chaînes de caractères (noms de modèles)
            if isinstance(model, str):
                if model.lower() == model_name.lower():
                    messagebox.showwarning("Attention", "Un modèle avec ce nom existe déjà")
                    return
            elif isinstance(model, dict) and "name" in model:
                if model["name"].lower() == model_name.lower():
                    messagebox.showwarning("Attention", "Un modèle avec ce nom existe déjà")
                    return
          # Créer le nouveau modèle
        new_model = {
            "name": model_name,
            "data_file": os.path.basename(file_path),
            "full_path": file_path,
            "description": self.description_text.get("1.0", tk.END).strip(),
            "regression_type": self.regression_type_var.get(),
            "accuracy": "En cours...",
            "network_architecture": self.layers_config.copy()  # Sauvegarder l'architecture configurée
        }
        
        self.controller.model_list.append(new_model)
        self.controller.status_label.config(text=f"Modèle '{model_name}' créé avec succès")
        
        # Retourner à l'accueil
        self.controller.show_page("HomePage")
          # Réinitialiser le formulaire
        self.name_entry.delete(0, tk.END)
        self.description_text.delete("1.0", tk.END)
        self.file_path_var.set("")
        self.file_info_label.config(text="Aucun fichier sélectionné", fg=self.controller.colors["text"])
    
    def add_layer(self):
        """Ajouter une nouvelle couche cachée au réseau"""
        # Insérer avant la couche de sortie (dernière)
        self.layers_config.insert(-1, 5)  # Ajouter une couche avec 5 neurones par défaut
        self.update_layers_interface()
        self.update_network_visualization()
    
    def apply_preset_architecture(self, architecture):
        """Appliquer une architecture prédéfinie"""
        self.layers_config = architecture.copy()
        
        # Mettre à jour les variables des spinbox d'entrée et sortie
        self.input_neurons_var.set(self.layers_config[0])
        self.output_neurons_var.set(self.layers_config[-1])
        
        # Mettre à jour l'interface et la visualisation
        self.update_layers_interface()
        self.update_network_visualization()
        
        # Message de confirmation
        arch_str = " → ".join(map(str, architecture))
        self.controller.update_status(f"Architecture appliquée: {arch_str}")
    
    def remove_layer(self):
        """Supprimer la dernière couche cachée"""
        if len(self.layers_config) > 2:  # Garder au moins une couche d'entrée et une de sortie
            self.layers_config.pop(-2)  # Supprimer l'avant-dernière couche (dernière couche cachée)
            self.update_layers_interface()
            self.update_network_visualization()
    
    def reset_architecture(self):
        """Réinitialiser l'architecture par défaut"""
        self.layers_config = [4, 8, 6, 1]
        self.update_layers_interface()
        self.update_network_visualization()
    
    def update_layers_interface(self):
        """Mettre à jour l'interface de configuration des couches"""
        # Mettre à jour les variables des spinbox d'entrée et de sortie
        if hasattr(self, 'input_neurons_var'):
            self.input_neurons_var.set(self.layers_config[0])
        if hasattr(self, 'output_neurons_var'):
            self.output_neurons_var.set(self.layers_config[-1])
        
        # Nettoyer le frame des couches
        for widget in self.layers_frame.winfo_children():
            widget.destroy()
          # Créer les contrôles pour chaque couche CACHÉE seulement
        hidden_layers = self.layers_config[1:-1]  # Exclure entrée et sortie
        
        if len(hidden_layers) == 0:
            # Afficher un message si aucune couche cachée
            no_hidden_label = tk.Label(
                self.layers_frame,
                text="Aucune couche cachée",
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text"],
                font=('Helvetica', 10, 'italic')
            )
            no_hidden_label.pack(pady=10)
        else:
            for i, neurons in enumerate(hidden_layers):
                layer_frame = tk.Frame(self.layers_frame, bg=self.controller.colors["bg_white"])
                layer_frame.pack(fill=tk.X, pady=2)
                
                # Label du type de couche (Cachée 1, Cachée 2, etc.)
                type_label = tk.Label(
                    layer_frame,
                    text=f"Cachée {i + 1}:",
                    bg=self.controller.colors["bg_white"],
                    fg=self.controller.colors["text"],
                    font=('Helvetica', 10),
                    width=12,
                    anchor="w"
                )
                type_label.pack(side=tk.LEFT, padx=(0, 5))
                
                # Spinbox pour modifier le nombre de neurones
                neurons_var = tk.StringVar(value=str(neurons))
                spinbox = tk.Spinbox(
                    layer_frame,
                    from_=1,
                    to=100,
                    textvariable=neurons_var,
                    width=8,
                    font=('Helvetica', 10),
                    command=lambda idx=i+1, var=neurons_var: self.update_layer_neurons(idx, var)
                )
                spinbox.pack(side=tk.LEFT, padx=(0, 5))
                
                # Bind pour la modification manuelle
                spinbox.bind('<KeyRelease>', lambda e, idx=i+1, var=neurons_var: self.update_layer_neurons(idx, var))
                
                # Label "neurones"
                tk.Label(
                    layer_frame,
                    text="neurones",
                    bg=self.controller.colors["bg_white"],
                    fg=self.controller.colors["text"],
                    font=('Helvetica', 10)
                ).pack(side=tk.LEFT)
    
    def update_layer_neurons(self, layer_index, neurons_var):
        """Mettre à jour le nombre de neurones d'une couche"""
        try:
            new_neurons = int(neurons_var.get())
            if new_neurons > 0:
                self.layers_config[layer_index] = new_neurons
                self.update_network_visualization()
        except ValueError:
            pass  # Ignorer les valeurs invalides
    
    def update_input_neurons(self):
        """Mettre à jour le nombre de neurones d'entrée"""
        try:
            new_neurons = int(self.input_neurons_var.get())
            if new_neurons > 0:
                self.layers_config[0] = new_neurons
                self.update_layers_interface()
                self.update_network_visualization()
        except ValueError:
            pass  # Ignorer les valeurs invalides
    
    def update_output_neurons(self):
        """Mettre à jour le nombre de neurones de sortie"""
        try:
            new_neurons = int(self.output_neurons_var.get())
            if new_neurons > 0:
                self.layers_config[-1] = new_neurons
                self.update_layers_interface()
                self.update_network_visualization()
        except ValueError:
            pass  # Ignorer les valeurs invalides
    
    def update_network_visualization(self):
        """Mettre à jour la visualisation du réseau de neurones"""
        # Nettoyer le frame de visualisation
        for widget in self.network_viz_frame.winfo_children():
            widget.destroy()
        
        # Créer une nouvelle figure matplotlib (plus large pour occuper toute la largeur)
        self.network_fig, self.network_ax = plt.subplots(figsize=(14, 8))
        self.network_fig.patch.set_facecolor('white')
        
        # Dessiner le réseau
        self._draw_network()
        
        # Ajuster l'espacement pour utiliser tout l'espace
        plt.tight_layout()
        
        # Intégrer dans tkinter avec une taille minimale
        self.network_canvas = FigureCanvasTkAgg(self.network_fig, master=self.network_viz_frame)
        self.network_canvas.draw()
        canvas_widget = self.network_canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        canvas_widget.configure(height=500)  # Hauteur minimale garantie
    
    def _draw_network(self):
        """Dessiner l'architecture du réseau de neurones"""
        self.network_ax.clear()
        self.network_ax.set_title("Architecture du réseau de neurones", fontsize=14, fontweight='bold', pad=20)
        
        layers_config = self.layers_config
        n_layers = len(layers_config)
        max_neurons = max(layers_config) if layers_config else 1
        
        # Paramètres de dessin
        layer_spacing = 2.0
        neuron_radius = 0.3
        max_display_neurons = 8  # Limite pour éviter l'encombrement
        
        # Couleurs pour chaque type de couche
        colors = {
            'input': '#4CAF50',     # Vert pour l'entrée
            'hidden': '#2196F3',    # Bleu pour les couches cachées
            'output': '#FF9800'     # Orange pour la sortie
        }
        
        # Calculer les positions des couches
        total_width = (n_layers - 1) * layer_spacing
        layer_positions = [i * layer_spacing - total_width/2 for i in range(n_layers)]
        
        # Dessiner chaque couche
        for layer_idx, (x_pos, n_neurons) in enumerate(zip(layer_positions, layers_config)):
            # Déterminer la couleur de la couche
            if layer_idx == 0:
                color = colors['input']
                layer_name = 'Entrée'
            elif layer_idx == n_layers - 1:
                color = colors['output']
                layer_name = 'Sortie'
            else:
                color = colors['hidden']
                layer_name = f'Cachée {layer_idx}'
            
            # Calculer les positions des neurones dans la couche
            display_neurons = min(n_neurons, max_display_neurons)
            if display_neurons < n_neurons:
                # Si trop de neurones, afficher seulement quelques-uns avec "..."
                neuron_spacing = 0.8
                total_height = (display_neurons - 1) * neuron_spacing
                neuron_positions = [i * neuron_spacing - total_height/2 for i in range(display_neurons)]
            else:
                neuron_spacing = 0.8 if display_neurons > 1 else 0
                total_height = (display_neurons - 1) * neuron_spacing
                neuron_positions = [i * neuron_spacing - total_height/2 for i in range(display_neurons)]
            
            # Dessiner les neurones
            for neuron_idx, y_pos in enumerate(neuron_positions):
                if display_neurons < n_neurons and neuron_idx == display_neurons - 1:
                    # Dessiner "..." pour indiquer qu'il y a plus de neurones
                    self.network_ax.text(x_pos, y_pos, '...', ha='center', va='center', 
                                       fontsize=12, fontweight='bold')
                else:
                    circle = plt.Circle((x_pos, y_pos), neuron_radius, color=color, alpha=0.8)
                    self.network_ax.add_patch(circle)
            
            # Ajouter le label de la couche
            label_y = max(neuron_positions) + 0.8 if neuron_positions else 0
            self.network_ax.text(x_pos, label_y, f'{layer_name}\n({n_neurons})', 
                               ha='center', va='bottom', fontsize=10, fontweight='bold')
              # Dessiner les connexions vers la couche suivante
            if layer_idx < n_layers - 1:
                next_layer_neurons = min(layers_config[layer_idx + 1], max_display_neurons)
                next_layer_spacing = 0.8 if next_layer_neurons > 1 else 0
                next_total_height = (next_layer_neurons - 1) * next_layer_spacing
                next_neuron_positions = [i * next_layer_spacing - next_total_height/2 for i in range(next_layer_neurons)]
                
                next_x_pos = layer_positions[layer_idx + 1]
                
                # Dessiner TOUTES les connexions entre neurones adjacents
                connection_alpha = 0.4 if max(len(neuron_positions), len(next_neuron_positions)) <= 6 else 0.2
                line_width = 0.8 if max(len(neuron_positions), len(next_neuron_positions)) <= 6 else 0.4
                
                for i, y1 in enumerate(neuron_positions):
                    for j, y2 in enumerate(next_neuron_positions):
                        # Ne pas dessiner de connexions depuis ou vers "..."
                        if (display_neurons < n_neurons and i == len(neuron_positions) - 1) or \
                           (next_layer_neurons < layers_config[layer_idx + 1] and j == len(next_neuron_positions) - 1):
                            continue
                        
                        # Calculer l'intensité de la connexion (simulation visuelle)
                        connection_color = 'darkblue' if (i + j) % 3 == 0 else 'gray'
                        
                        self.network_ax.plot([x_pos + neuron_radius, next_x_pos - neuron_radius], 
                                           [y1, y2], connection_color, alpha=connection_alpha, 
                                           linewidth=line_width, zorder=1)
                
                # Si il y a des neurones cachés (représentés par "..."), dessiner quelques connexions vers eux
                if display_neurons < n_neurons or next_layer_neurons < layers_config[layer_idx + 1]:
                    # Ajouter quelques connexions en pointillés pour suggérer les connexions cachées
                    for i in range(min(2, len(neuron_positions))):
                        for j in range(min(2, len(next_neuron_positions))):
                            if i < len(neuron_positions) and j < len(next_neuron_positions):
                                y1 = neuron_positions[i]
                                y2 = next_neuron_positions[j]
                                self.network_ax.plot([x_pos + neuron_radius, next_x_pos - neuron_radius], 
                                                   [y1, y2], 'lightgray', alpha=0.3, 
                                                   linewidth=0.3, linestyle='--', zorder=1)
        
        # Configuration des axes
        self.network_ax.set_xlim(-total_width/2 - 1, total_width/2 + 1)
        
        if layers_config:
            max_height = max([max(0.8 * (min(n, max_display_neurons) - 1) / 2, 1) for n in layers_config])
            self.network_ax.set_ylim(-max_height - 1, max_height + 2)
        else:
            self.network_ax.set_ylim(-2, 2)
        
        self.network_ax.set_aspect('equal')
        self.network_ax.axis('off')
        
        # Ajouter des informations sur l'architecture
        info_text = f"Couches: {n_layers} | Paramètres totaux: {self._calculate_total_parameters()}"
        self.network_ax.text(0, -max_height - 0.8, info_text, ha='center', va='top', 
                           fontsize=10, style='italic')
        
        plt.tight_layout()
    
    def _calculate_total_parameters(self):
        """Calculer le nombre total de paramètres du réseau"""
        total_params = 0
        for i in range(len(self.layers_config) - 1):
            # Poids: couche_actuelle * couche_suivante
            # Biais: couche_suivante
            weights = self.layers_config[i] * self.layers_config[i + 1]
            biases = self.layers_config[i + 1]
            total_params += weights + biases
        return total_params
