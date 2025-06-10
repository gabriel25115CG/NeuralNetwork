import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import os

class CreateModelPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.create_widgets()
        
    def create_widgets(self):
        # Cadre principal
        main_frame = tk.Frame(self, bg=self.controller.colors["bg_light"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
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
            text="Créer un nouveau modèle de régression",
            font=('Helvetica', 20, 'bold'),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["primary"]
        )
        title_label.pack(pady=(0, 30))
        
        # Cadre principal du formulaire
        form_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_white"], padx=40, pady=30)
        form_frame.pack(fill=tk.BOTH, expand=True)
        
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
        
        # Section 3: Paramètres du modèle
        params_section = tk.LabelFrame(
            form_frame,
            text="Paramètres du modèle",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        params_section.pack(fill=tk.X, pady=(0, 30))
        
        # Type de régression
        tk.Label(
            params_section,
            text="Type de régression:",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(10, 5))
        
        self.regression_type_var = tk.StringVar(value="Linéaire")
        regression_combo = ttk.Combobox(
            params_section,
            textvariable=self.regression_type_var,
            values=["Linéaire", "Polynomiale", "Ridge", "Lasso"],
            state="readonly",
            font=('Helvetica', 11),
            width=20
        )
        regression_combo.pack(anchor="w", pady=(0, 15))
        
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
            cursor="hand2"
        )
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
            if model["name"].lower() == model_name.lower():
                messagebox.showwarning("Attention", "Un modèle avec ce nom existe déjà")
                return
        
        # Préparer les informations du modèle
        model_info = {
            "name": model_name,
            "data_file": os.path.basename(file_path),
            "full_path": file_path,
            "description": self.description_text.get("1.0", tk.END).strip(),
            "regression_type": self.regression_type_var.get(),
            "accuracy": "En cours..."
        }
        
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
            "accuracy": "En cours..."
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
