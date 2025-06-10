import tkinter as tk
from tkinter import ttk, messagebox

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.create_widgets()
        
    def create_widgets(self):
        # Cadre principal
        main_frame = tk.Frame(self, bg=self.controller.colors["bg_light"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Titre animé
        from tkinter.font import Font
        self.controller.title_font = Font(family="Helvetica", size=22, weight="bold")
        self.controller.title_label = tk.Label(
            main_frame,
            text="ML Model Manager",
            font=self.controller.title_font,
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["primary"]
        )
        self.controller.title_label.pack(pady=(0, 10))
        
        # Barre de recherche en haut
        search_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        search_frame.pack(fill=tk.X, pady=(0, 15))
        
        search_label = tk.Label(
            search_frame, 
            text="Rechercher un modèle:", 
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["text"]
        )
        search_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_models)
        
        self.search_entry = ttk.Entry(
            search_frame, 
            textvariable=self.search_var,
            width=30,
            style="TEntry"
        )
        self.search_entry.pack(side=tk.LEFT)
        
        # Description de l'application
        description = """
        Bienvenue dans ML Model Manager, votre outil complet pour gérer vos modèles de régression.
        Créez, entraînez, testez et optimisez vos modèles en toute simplicité.
        """
        desc_label = tk.Label(
            main_frame,
            text=description,
            font=('Helvetica', 11),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["text"],
            wraplength=800,
            justify="center"
        )
        desc_label.pack(pady=(0, 20))
        
        # Cadre pour la liste des modèles
        list_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_white"], padx=15, pady=15)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # En-tête de section
        list_header = tk.Label(
            list_frame,
            text="Vos modèles de régression",
            font=('Helvetica', 14, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"]
        )
        list_header.pack(anchor="w", pady=(0, 10))
        
        # Création du Treeview pour afficher les modèles
        self.tree = ttk.Treeview(list_frame, columns=("name", "data_file", "accuracy"), show="headings")
        self.tree.heading("name", text="Nom du modèle")
        self.tree.heading("data_file", text="Fichier de données")
        self.tree.heading("accuracy", text="Précision")
        
        # Configuration des colonnes
        self.tree.column("name", width=250)
        self.tree.column("data_file", width=200)
        self.tree.column("accuracy", width=100)
        
        # Insertion des données ou message si vide
        self.update_model_display()
        
        # Scrollbar pour le Treeview
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Placement du Treeview et de la scrollbar
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Binding pour double-clic sur un modèle
        self.tree.bind("<Double-1>", self.on_model_select)
        
        # Cadre pour les boutons d'action
        button_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"], pady=15)
        button_frame.pack(fill=tk.X)
        
        # Bouton stylisé pour créer un nouveau modèle
        self.new_model_button = ttk.Button(
            button_frame,
            text="➕ Créer un nouveau modèle",
            command=lambda: self.controller.show_page("CreateModelPage"),
            style='Add.TButton',
            cursor="hand2"
        )
        self.new_model_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Boutons d'action réguliers
        self.edit_model_button = ttk.Button(
            button_frame,
            text="Modifier le modèle sélectionné",
            command=self.edit_model,
            style='TButton',
            cursor="hand2"
        )
        self.edit_model_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.delete_model_button = ttk.Button(
            button_frame,
            text="Supprimer le modèle",
            command=self.delete_model,
            style='TButton',
            cursor="hand2"
        )
        self.delete_model_button.pack(side=tk.LEFT)

    def filter_models(self, *args):
        search_text = self.search_var.get().lower()
        
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if not search_text:
            self.update_model_display()
        else:
            filtered_models = [model for model in self.controller.model_list 
                             if search_text in model["name"].lower()]
            
            if filtered_models:
                for model in filtered_models:
                    self.tree.insert("", tk.END, values=(model["name"], model["data_file"], model["accuracy"]))
            else:
                self.tree.insert("", tk.END, values=("Aucun modèle trouvé", "", ""))
        
        if search_text:
            self.controller.status_label.config(text=f"Recherche: '{search_text}'")
        else:
            self.controller.status_label.config(text="Prêt")

    def update_model_display(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        if self.controller.model_list:
            for model in self.controller.model_list:
                self.tree.insert("", tk.END, values=(model["name"], model["data_file"], model["accuracy"]))
        else:
            self.tree.insert("", tk.END, values=("Vous n'avez pas de modèle.", "Veuillez en ajouter un nouveau.", ""))

    def on_model_select(self, event):
        selected_item = self.tree.selection()
        if selected_item:
            model_name = self.tree.item(selected_item[0])['values'][0]
            
            # Ignorer si c'est un message d'aide
            if model_name in ["Vous n'avez pas de modèle.", "Aucun modèle trouvé"]:
                return
                
            # Trouver le modèle correspondant
            for model in self.controller.model_list:
                if model["name"] == model_name:
                    self.controller.show_model_details(model)
                    break
    
    def edit_model(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Attention", "Veuillez sélectionner un modèle à modifier")
            return
            
        model_name = self.tree.item(selected_item[0])['values'][0]
        self.controller.status_label.config(text=f"Modification du modèle : {model_name}")
        messagebox.showinfo("Modification", f"Modification du modèle '{model_name}'")
    
    def delete_model(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Attention", "Veuillez sélectionner un modèle à supprimer")
            return
            
        model_name = self.tree.item(selected_item[0])['values'][0]
        
        if model_name == "Vous n'avez pas de modèle.":
            return
        
        if messagebox.askyesno("Confirmation", f"Êtes-vous sûr de vouloir supprimer le modèle '{model_name}' ?"):
            self.controller.model_list = [model for model in self.controller.model_list if model["name"] != model_name]
            self.update_model_display()
            self.controller.status_label.config(text=f"Modèle '{model_name}' supprimé")
