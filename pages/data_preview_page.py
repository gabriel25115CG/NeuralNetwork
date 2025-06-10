import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import os

class DataPreviewPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.df = None
        self.target_column = None
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
            command=lambda: self.controller.show_page("CreateModelPage"),
            style='TButton'
        ).pack(side=tk.LEFT)
        
        self.title_label = tk.Label(
            main_frame,
            text="Prévisualisation des données",
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
        
        # Ajuster la largeur
        def on_canvas_configure(event):
            canvas_width = event.width
            self.main_canvas.itemconfig(self.main_canvas.find_all()[0], width=canvas_width)
        
        self.main_canvas.bind('<Configure>', on_canvas_configure)
        
        # Pack canvas et scrollbar
        self.main_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Contenu principal
        content_frame = tk.Frame(self.scrollable_content, bg=self.controller.colors["bg_white"], padx=30, pady=20)
        content_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Section informations
        self.create_info_section(content_frame)
        
        # Section sélection variables
        self.create_variables_section(content_frame)
        
        # Section aide
        self.create_help_section(content_frame)
        
        # Section statistiques
        self.create_stats_section(content_frame)
        
        # Section preview
        self.create_preview_section(content_frame)
        
        # Boutons
        self.create_buttons(main_frame)
        
        # Variables
        self.feature_vars = {}
    
    def create_info_section(self, parent):
        info_section = tk.LabelFrame(
            parent,
            text="Informations du fichier",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=15, pady=10
        )
        info_section.pack(fill=tk.X, pady=(0, 15))
        
        self.info_label = tk.Label(
            info_section,
            text="Aucune donnée chargée",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11),
            justify="left"
        )
        self.info_label.pack(anchor="w", pady=5)
    
    def create_variables_section(self, parent):
        variables_section = tk.LabelFrame(
            parent,
            text="Sélection des variables",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=15, pady=10
        )
        variables_section.pack(fill=tk.X, pady=(0, 15))
        
        # Variable cible
        tk.Label(
            variables_section,
            text="🎯 Variable cible :",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11, 'bold')
        ).pack(anchor="w", pady=(5, 2))
        
        self.target_var = tk.StringVar()
        self.target_combo = ttk.Combobox(
            variables_section,
            textvariable=self.target_var,
            state="readonly",
            font=('Helvetica', 11),
            width=30
        )
        self.target_combo.pack(anchor="w", pady=(0, 10))
        self.target_combo.bind("<<ComboboxSelected>>", self.on_target_selected)
        
        # Variables explicatives
        tk.Label(
            variables_section,
            text="📊 Variables explicatives :",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11, 'bold')
        ).pack(anchor="w", pady=(5, 2))
        
        # Frame pour checkboxes avec limite de hauteur
        self.features_frame = tk.Frame(variables_section, bg=self.controller.colors["bg_white"], height=80)
        self.features_frame.pack(fill=tk.X, pady=(0, 10))
        self.features_frame.pack_propagate(False)
        
        # Canvas pour scroll des checkboxes
        self.features_canvas = tk.Canvas(self.features_frame, bg=self.controller.colors["bg_white"], height=80)
        features_scrollbar = ttk.Scrollbar(self.features_frame, orient="vertical", command=self.features_canvas.yview)
        self.scrollable_features = tk.Frame(self.features_canvas, bg=self.controller.colors["bg_white"])
        
        self.scrollable_features.bind(
            "<Configure>",
            lambda e: self.features_canvas.configure(scrollregion=self.features_canvas.bbox("all"))
        )
        
        self.features_canvas.create_window((0, 0), window=self.scrollable_features, anchor="nw")
        self.features_canvas.configure(yscrollcommand=features_scrollbar.set)
        
        self.features_canvas.pack(side="left", fill="both", expand=True)
        features_scrollbar.pack(side="right", fill="y")
        
        # Boutons de sélection
        buttons_frame = tk.Frame(variables_section, bg=self.controller.colors["bg_white"])
        buttons_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(buttons_frame, text="Tout sélectionner", command=self.select_all_features).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="Désélectionner", command=self.deselect_all_features).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="Auto (numériques)", command=self.auto_select_features).pack(side=tk.LEFT)
    
    def create_help_section(self, parent):
        help_section = tk.LabelFrame(
            parent,
            text="💡 Aide",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["primary"],
            padx=15, pady=10
        )
        help_section.pack(fill=tk.X, pady=(0, 15))
        
        help_text = """🎯 Variable cible : Ce que vous voulez prédire
📊 Variables explicatives : Les données utilisées pour la prédiction
📈 Statistiques : Résumé de vos données (moyenne, min, max...)"""
        
        tk.Label(
            help_section,
            text=help_text,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10),
            justify="left"
        ).pack(anchor="w", pady=5)
    
    def create_stats_section(self, parent):
        stats_section = tk.LabelFrame(
            parent,
            text="Statistiques descriptives",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=15, pady=10
        )
        stats_section.pack(fill=tk.X, pady=(0, 15))
        
        # Treeview pour statistiques
        columns = ("Colonne", "Type", "Moyenne", "Min", "Max", "Manquantes")
        self.stats_tree = ttk.Treeview(stats_section, columns=columns, show="headings", height=6)
        
        for col in columns:
            self.stats_tree.heading(col, text=col)
            self.stats_tree.column(col, width=100, anchor="center")
        
        self.stats_tree.pack(fill=tk.X, pady=5)
        self.stats_tree.tag_configure("target", background="#e8f5e8")
    
    def create_preview_section(self, parent):
        preview_section = tk.LabelFrame(
            parent,
            text="Aperçu des données (5 premières lignes)",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=15, pady=10
        )
        preview_section.pack(fill=tk.X, pady=(0, 15))
        
        # Frame avec scrollbars
        preview_frame = tk.Frame(preview_section, bg=self.controller.colors["bg_white"])
        preview_frame.pack(fill=tk.X, pady=5)
        
        self.preview_tree = ttk.Treeview(preview_frame, show="headings", height=6)
        
        v_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        h_scroll = ttk.Scrollbar(preview_frame, orient="horizontal", command=self.preview_tree.xview)
        
        self.preview_tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        self.preview_tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")
        
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
    
    def create_buttons(self, parent):
        button_frame = tk.Frame(parent, bg=self.controller.colors["bg_light"])
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        ttk.Button(
            button_frame,
            text="Annuler",
            command=lambda: self.controller.show_page("CreateModelPage"),
            style='TButton'
        ).pack(side=tk.RIGHT, padx=(5, 0))
        
        self.continue_button = ttk.Button(
            button_frame,
            text="Entraîner le modèle",
            command=self.continue_training,
            style='Add.TButton',
            state="disabled"
        )
        self.continue_button.pack(side=tk.RIGHT)
    
    def load_data(self, file_path, model_info):
        try:
            self.df = pd.read_csv(file_path, low_memory=False)
            self.model_info = model_info
            
            filename = os.path.basename(file_path)
            self.title_label.config(text=f"Prévisualisation - {filename}")
            
            rows, cols = self.df.shape
            size_kb = self.df.memory_usage(deep=True).sum() / 1024
            info_text = f"📊 {rows} lignes × {cols} colonnes\n📁 {filename}\n💾 {size_kb:.1f} KB"
            
            # Analyse de qualité des données
            self.analyze_data_quality()
            
            # Colonnes numériques avec validation renforcée
            numeric_columns = []
            non_numeric_info = []
            
            for col in self.df.columns:
                try:
                    # Essayer de convertir en numérique
                    numeric_data = pd.to_numeric(self.df[col], errors='coerce')
                    valid_numeric_count = len(numeric_data.dropna())
                    total_count = len(self.df[col])
                    
                    # Considérer comme numérique si au moins 80% des valeurs sont convertibles
                    if valid_numeric_count > 0 and (valid_numeric_count / total_count) >= 0.8:
                        numeric_columns.append(col)
                    else:
                        non_numeric_count = total_count - valid_numeric_count
                        non_numeric_info.append(f"{col}: {non_numeric_count} valeurs non-numériques")
                except:
                    non_numeric_info.append(f"{col}: erreur de conversion")
                    continue
            
            # Ajouter info sur les colonnes non-numériques
            quality_warnings = []
            if non_numeric_info:
                info_text += f"\n⚠️ Colonnes avec données non-numériques:\n" + "\n".join(non_numeric_info[:3])
                if len(non_numeric_info) > 3:
                    info_text += f"\n... et {len(non_numeric_info)-3} autres"
                quality_warnings.append(f"{len(non_numeric_info)} colonnes non-numériques")
            
            # Analyser la corrélation pour alerter l'utilisateur
            if len(numeric_columns) >= 2:
                try:
                    correlation_matrix = self.df[numeric_columns].corr()
                    low_correlation_pairs = []
                    for i, col1 in enumerate(numeric_columns):
                        for j, col2 in enumerate(numeric_columns[i+1:], i+1):
                            corr_value = abs(correlation_matrix.loc[col1, col2])
                            if corr_value < 0.1:
                                low_correlation_pairs.append((col1, col2, corr_value))
                    
                    if len(low_correlation_pairs) > len(numeric_columns) * 0.7:
                        quality_warnings.append("Faible corrélation entre variables")
                except:
                    pass
            
            if quality_warnings:
                info_text += f"\n\n🚨 ALERTES QUALITÉ:\n• " + "\n• ".join(quality_warnings)
            
            self.info_label.config(text=info_text)
            
            if not numeric_columns:
                messagebox.showwarning(
                    "Attention", 
                    "Aucune colonne numérique détectée dans ce fichier.\n"
                    "Les modèles de régression nécessitent des données numériques."
                )
                return
            
            self.target_combo['values'] = numeric_columns
            if numeric_columns:
                # Choisir la dernière colonne numérique comme cible par défaut
                self.target_var.set(numeric_columns[-1])
                self.on_target_selected()
            
            self.create_feature_checkboxes()
            self.auto_select_features()
            self.update_statistics()
            self.update_preview()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le fichier:\n{str(e)}")
    
    def create_feature_checkboxes(self):
        for widget in self.scrollable_features.winfo_children():
            widget.destroy()
        
        self.feature_vars = {}
        
        if self.df is not None:
            for column in self.df.columns:
                if column != self.target_column:
                    var = tk.BooleanVar()
                    self.feature_vars[column] = var
                    
                    frame = tk.Frame(self.scrollable_features, bg=self.controller.colors["bg_white"])
                    frame.pack(fill=tk.X, pady=1)
                    
                    checkbox = tk.Checkbutton(
                        frame,
                        text=column,
                        variable=var,
                        bg=self.controller.colors["bg_white"],
                        fg=self.controller.colors["text"],
                        font=('Helvetica', 10),
                        anchor="w"
                    )
                    checkbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
                    
                    # Indicateur de type
                    try:
                        numeric_data = pd.to_numeric(self.df[column], errors='coerce')
                        icon = "📊" if not numeric_data.dropna().empty else "📝"
                        tk.Label(frame, text=icon, bg=self.controller.colors["bg_white"], font=('Helvetica', 10)).pack(side=tk.RIGHT)
                    except:
                        pass
    
    def select_all_features(self):
        for var in self.feature_vars.values():
            var.set(True)
    
    def deselect_all_features(self):
        for var in self.feature_vars.values():
            var.set(False)
    
    def auto_select_features(self):
        self.deselect_all_features()
        if self.df is not None:
            for column in self.df.columns:
                if column != self.target_column and column in self.feature_vars:
                    try:
                        numeric_data = pd.to_numeric(self.df[column], errors='coerce')
                        if not numeric_data.dropna().empty:
                            self.feature_vars[column].set(True)
                    except:
                        pass
    
    def get_selected_features(self):
        return [col for col, var in self.feature_vars.items() if var.get()]
    
    def update_statistics(self):
        for item in self.stats_tree.get_children():
            self.stats_tree.delete(item)
        
        for column in self.df.columns:
            col_data = self.df[column]
            missing_count = col_data.isnull().sum()
            
            try:
                numeric_data = pd.to_numeric(col_data, errors='coerce')
                if not numeric_data.dropna().empty:
                    mean_val = f"{numeric_data.mean():.2f}"
                    min_val = f"{numeric_data.min():.2f}"
                    max_val = f"{numeric_data.max():.2f}"
                    col_type = "Numérique"
                else:
                    mean_val = "N/A"
                    min_val = str(col_data.dropna().min()) if not col_data.dropna().empty else "N/A"
                    max_val = str(col_data.dropna().max()) if not col_data.dropna().empty else "N/A"
                    col_type = "Texte"
            except:
                mean_val = min_val = max_val = "N/A"
                col_type = "Mixte"
            
            tags = ("target",) if column == self.target_column else ()
            self.stats_tree.insert("", "end", values=(column, col_type, mean_val, min_val, max_val, missing_count), tags=tags)
    
    def update_preview(self):
        self.preview_tree["columns"] = ()
        columns = list(self.df.columns)
        self.preview_tree["columns"] = columns
        
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=120, anchor="center")
        
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        for _, row in self.df.head(5).iterrows():
            values = [str(row[col]) for col in columns]
            self.preview_tree.insert("", "end", values=values)
    
    def on_target_selected(self, event=None):
        self.target_column = self.target_var.get()
        self.update_statistics()
        self.create_feature_checkboxes()
        self.auto_select_features()
        self.continue_button.config(state="normal")
    
    def continue_training(self):
        if not self.target_column:
            messagebox.showwarning("Attention", "Veuillez sélectionner une colonne cible")
            return
        
        selected_features = self.get_selected_features()
        if not selected_features:
            messagebox.showwarning("Attention", "Veuillez sélectionner au moins une variable explicative")
            return
        
        # Recommandations basées sur les résultats d'analyse
        if len(selected_features) < 2:
            result = messagebox.askyesno(
                "Recommandation", 
                "⚠️ ATTENTION : Vous n'avez sélectionné qu'une seule variable explicative.\n\n"
                "Pour de meilleurs résultats, il est recommandé d'utiliser plusieurs variables.\n"
                "Souhaitez-vous continuer quand même ?"
            )
            if not result:
                return
        
        # Valider que les colonnes sélectionnées sont bien numériques
        non_numeric_features = []
        for feature in selected_features:
            try:
                numeric_data = pd.to_numeric(self.df[feature], errors='coerce')
                valid_count = len(numeric_data.dropna())
                total_count = len(self.df[feature])
                if valid_count == 0 or (valid_count / total_count) < 0.5:
                    non_numeric_features.append(feature)
            except:
                non_numeric_features.append(feature)
        
        if non_numeric_features:
            messagebox.showwarning(
                "Attention", 
                f"Les variables suivantes contiennent trop de données non-numériques:\n" +
                "\n".join(non_numeric_features[:5]) +
                ("\n..." if len(non_numeric_features) > 5 else "") +
                "\n\nVeuillez désélectionner ces variables ou nettoyer vos données."
            )
            return
        
        # Valider la variable cible
        try:
            target_numeric = pd.to_numeric(self.df[self.target_column], errors='coerce')
            if len(target_numeric.dropna()) < len(self.df[self.target_column]) * 0.8:
                messagebox.showwarning(
                    "Attention", 
                    f"La variable cible '{self.target_column}' contient trop de valeurs non-numériques.\n"
                    "Veuillez choisir une autre variable cible."
                )
                return
        except:
            messagebox.showwarning("Attention", f"Impossible de valider la variable cible '{self.target_column}'")
            return
        
        self.model_info["target_column"] = self.target_column
        self.model_info["feature_columns"] = selected_features
        self.model_info["data_shape"] = self.df.shape
        
        # Aller à la page d'entraînement au lieu de créer directement le modèle
        self.controller.show_model_training(self.model_info["full_path"], self.model_info)
    
    def analyze_data_quality(self):
        """Analyse la qualité des données et détecte les valeurs aberrantes."""
        if self.df is not None:
            # Exemple simple : marquer comme aberrantes les valeurs en dehors de 3 écarts-types
            numeric_cols = self.df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            for col in numeric_cols:
                try:
                    mean = self.df[col].mean()
                    std = self.df[col].std()
                    threshold_upper = mean + 3 * std
                    threshold_lower = mean - 3 * std
                    
                    # Marquer les valeurs aberrantes
                    outliers = self.df[(self.df[col] > threshold_upper) | (self.df[col] < threshold_lower)]
                    if not outliers.empty:
                        self.df.loc[outliers.index, col] = None  # Remplacer par NaN ou une autre méthode
                except:
                    continue
            
            # Mettre à jour les statistiques après traitement des valeurs aberrantes
            self.update_statistics()
            self.update_preview()
