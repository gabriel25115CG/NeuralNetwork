import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.font import Font
import os

class MLModelManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Model Manager")
        self.geometry("1920x1080")
        
        # Thème de couleurs
        self.colors = {
            "primary": "#3498db",       # Bleu principal
            "primary_dark": "#2980b9",  # Bleu foncé (hover)
            "bg_light": "#f5f5f5",      # Fond clair
            "bg_white": "#ffffff",      # Fond blanc
            "text": "#333333",          # Texte principal
            "accent": "#e74c3c",        # Accent (animations)
            "success": "#2ecc71",       # Vert (succès)
            "text_invisible": "#a3c7e3"  # Texte semi-transparent (presque invisible)
        }
        
        self.configure(bg=self.colors["bg_light"])
        
        # Exemple de liste de modèles (vide au départ)
        self.model_list = []
        
        # Animation
        self.animation_step = 0
        
        self.setup_styles()
        self.create_container()
        self.create_pages()
        self.show_page("HomePage")
        self.animate_title()
        
    def create_container(self):
        # Container principal pour toutes les pages
        self.container = tk.Frame(self, bg=self.colors["bg_light"])
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Dictionnaire pour stocker les pages
        self.pages = {}
        
    def create_pages(self):
        # Créer toutes les pages
        for PageClass in (HomePage, CreateModelPage):
            page = PageClass(self.container, self)
            self.pages[PageClass.__name__] = page
            page.grid(row=0, column=0, sticky="nsew")
            
    def show_page(self, page_name):
        # Afficher une page spécifique
        page = self.pages[page_name]
        page.tkraise()
        
        # Mettre à jour l'affichage si c'est la page d'accueil
        if page_name == "HomePage":
            page.update_model_display()

    def setup_styles(self):
        # Configuration des styles pour les widgets ttk
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Base theme
        
        # Style pour les boutons avec texte semi-invisible
        self.style.configure(
            'TButton',
            background=self.colors["primary"],
            foreground=self.colors["text_invisible"],  # Texte presque invisible
            font=('Helvetica', 10, 'bold'),
            padding=10
        )
        
        # Style pour le survol des boutons (texte devient blanc)
        self.style.map('TButton',
            background=[('active', self.colors["primary_dark"])],
            foreground=[('active', 'white')],  # Texte visible au survol
            relief=[('pressed', 'sunken')]
        )
        
        # Style pour les treeview
        self.style.configure(
            "Treeview",
            background=self.colors["bg_white"],
            foreground=self.colors["text"],
            rowheight=30,
            fieldbackground=self.colors["bg_white"]
        )
        self.style.map('Treeview', 
            background=[('selected', self.colors["primary"])]
        )
        
        # Style pour les en-têtes de Treeview
        self.style.configure(
            "Treeview.Heading",
            background=self.colors["primary"],
            foreground="white",
            font=('Helvetica', 10, 'bold')
        )
        
        # Style pour l'entrée de recherche
        self.style.configure(
            "TEntry",
            fieldbackground=self.colors["bg_white"],
            bordercolor=self.colors["primary"]
        )
        
        # Style spécial pour le bouton d'ajout
        self.style.configure(
            'Add.TButton',
            background=self.colors["success"],
            foreground="white",
            font=('Helvetica', 11, 'bold'),
            padding=(20, 12),
            relief="flat"
        )
        
        self.style.map('Add.TButton',
            background=[('active', '#27ae60'), ('pressed', '#1e8449')],
            foreground=[('active', 'white'), ('pressed', 'white')],
            relief=[('pressed', 'sunken')]
        )

    def create_widgets(self):
        # Cadre principal
        main_frame = tk.Frame(self, bg=self.colors["bg_light"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Titre animé
        self.title_font = Font(family="Helvetica", size=22, weight="bold")
        self.title_label = tk.Label(
            main_frame,
            text="ML Model Manager",
            font=self.title_font,
            bg=self.colors["bg_light"],
            fg=self.colors["primary"]
        )
        self.title_label.pack(pady=(0, 10))
        
        # Barre de recherche en haut
        search_frame = tk.Frame(main_frame, bg=self.colors["bg_light"])
        search_frame.pack(fill=tk.X, pady=(0, 15))
        
        search_label = tk.Label(
            search_frame, 
            text="Rechercher un modèle:", 
            bg=self.colors["bg_light"],
            fg=self.colors["text"]
        )
        search_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_models)  # Déclenche la recherche à chaque frappe
        
        self.search_entry = ttk.Entry(
            search_frame, 
            textvariable=self.search_var,
            width=30,
            style="TEntry"
        )
        self.search_entry.pack(side=tk.LEFT)
        
        # Description de l'application
        description = """
        Bienvenue dans ML Model Manager, votre outil complet pour gérer vos modèles de machine learning.
        Créez, entraînez, testez et optimisez vos modèles en toute simplicité.
        """
        desc_label = tk.Label(
            main_frame,
            text=description,
            font=('Helvetica', 11),
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            wraplength=800,
            justify="center"
        )
        desc_label.pack(pady=(0, 20))
        
        # Cadre pour la liste des modèles
        list_frame = tk.Frame(main_frame, bg=self.colors["bg_white"], padx=15, pady=15)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        # En-tête de section
        list_header = tk.Label(
            list_frame,
            text="Vos modèles de machine learning",
            font=('Helvetica', 14, 'bold'),
            bg=self.colors["bg_white"],
            fg=self.colors["text"]
        )
        list_header.pack(anchor="w", pady=(0, 10))
        
        # Création du Treeview pour afficher les modèles
        self.tree = ttk.Treeview(list_frame, columns=("name", "type", "accuracy"), show="headings")
        self.tree.heading("name", text="Nom du modèle")
        self.tree.heading("type", text="Type")
        self.tree.heading("accuracy", text="Précision")
        
        # Configuration des colonnes
        self.tree.column("name", width=250)
        self.tree.column("type", width=150)
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
        button_frame = tk.Frame(main_frame, bg=self.colors["bg_light"], pady=15)
        button_frame.pack(fill=tk.X)
        
        # Bouton stylisé pour créer un nouveau modèle (plus grand et proéminent)
        self.new_model_button = ttk.Button(
            button_frame,
            text="➕ Créer un nouveau modèle",
            command=self.create_new_model,
            style='Add.TButton',
            cursor="hand2"  # Pointeur main au survol
        )
        self.new_model_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Boutons d'action réguliers
        self.edit_model_button = ttk.Button(
            button_frame,
            text="Modifier le modèle sélectionné",
            command=self.edit_model,
            style='TButton',
            cursor="hand2"  # Pointeur main au survol
        )
        self.edit_model_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.delete_model_button = ttk.Button(
            button_frame,
            text="Supprimer le modèle",
            command=self.delete_model,
            style='TButton',
            cursor="hand2"  # Pointeur main au survol
        )
        self.delete_model_button.pack(side=tk.LEFT)
        
        # Barre de statut
        self.status_label = tk.Label(
            self,
            text="Prêt",
            bg=self.colors["primary"],
            fg="white",
            font=('Helvetica', 10),
            anchor="w",
            padx=10
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def filter_models(self, *args):
        # Fonction pour filtrer les modèles selon le texte de recherche
        search_text = self.search_var.get().lower()
        
        # Effacer tous les items actuels
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Si pas de recherche, afficher tous les modèles
        if not search_text:
            self.update_model_display()
        else:
            # Filtrer et réinsérer les modèles qui correspondent
            filtered_models = [model for model in self.model_list 
                             if search_text in model["name"].lower()]
            
            if filtered_models:
                for model in filtered_models:
                    self.tree.insert("", tk.END, values=(model["name"], model["type"], model["accuracy"]))
            else:
                # Aucun résultat trouvé
                self.tree.insert("", tk.END, values=("Aucun modèle trouvé", "", ""))
        
        # Mettre à jour le statut
        if search_text:
            self.status_label.config(text=f"Recherche: '{search_text}'")
        else:
            self.status_label.config(text="Prêt")

    def update_model_display(self):
        # Effacer le contenu actuel
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Afficher les modèles ou message si vide
        if self.model_list:
            for model in self.model_list:
                self.tree.insert("", tk.END, values=(model["name"], model["type"], model["accuracy"]))
        else:
            # Message quand aucun modèle n'est disponible
            self.tree.insert("", tk.END, values=("Vous n'avez pas de modèle.", "Veuillez en ajouter un nouveau.", ""))

    def animate_title(self):
        # Animation plus sophistiquée pour le titre
        colors = [
            self.colors["primary"],
            "#4aa3df",
            "#5dade2",
            "#5dade2",
            "#4aa3df",
            self.colors["primary"],
            self.colors["primary_dark"]
        ]
        
        self.animation_step = (self.animation_step + 1) % len(colors)
        self.title_label.config(fg=colors[self.animation_step])
        self.after(200, self.animate_title)

    def on_model_select(self, event):
        # Gestion du double-clic sur un modèle
        selected_item = self.tree.selection()
        if selected_item:
            model_name = self.tree.item(selected_item[0])['values'][0]
            self.status_label.config(text=f"Modèle sélectionné : {model_name}")
    
    def create_new_model(self):
        # Créer une fenêtre de dialogue pour la création de modèle
        dialog = tk.Toplevel(self)
        dialog.title("Créer un nouveau modèle")
        dialog.geometry("400x300")
        dialog.configure(bg=self.colors["bg_light"])
        dialog.grab_set()  # Fenêtre modale
        
        # Titre de la fenêtre
        title_label = tk.Label(
            dialog,
            text="Nouveau modèle",
            font=('Helvetica', 16, 'bold'),
            bg=self.colors["bg_light"],
            fg=self.colors["primary"]
        )
        title_label.pack(pady=(20, 10))
        
        # Cadre principal pour les champs
        form_frame = tk.Frame(dialog, bg=self.colors["bg_light"])
        form_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Nom du modèle
        tk.Label(
            form_frame,
            text="Nom du modèle:",
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(10, 5))
        
        name_entry = tk.Entry(
            form_frame,
            font=('Helvetica', 11),
            width=30,
            bg=self.colors["bg_white"]
        )
        name_entry.pack(fill=tk.X, pady=(0, 10))
        name_entry.focus()
        
        # Type de modèle
        tk.Label(
            form_frame,
            text="Type de modèle:",
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(10, 5))
        
        type_var = tk.StringVar(value="Classification")
        type_combo = ttk.Combobox(
            form_frame,
            textvariable=type_var,
            values=["Classification", "Régression", "Clustering", "Deep Learning"],
            state="readonly",
            font=('Helvetica', 11)
        )
        type_combo.pack(fill=tk.X, pady=(0, 10))
        
        # Précision (optionnelle)
        tk.Label(
            form_frame,
            text="Précision (optionnelle):",
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            font=('Helvetica', 11)
        ).pack(anchor="w", pady=(10, 5))
        
        accuracy_entry = tk.Entry(
            form_frame,
            font=('Helvetica', 11),
            width=30,
            bg=self.colors["bg_white"]
        )
        accuracy_entry.pack(fill=tk.X, pady=(0, 20))
        
        # Cadre pour les boutons
        button_frame = tk.Frame(dialog, bg=self.colors["bg_light"])
        button_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        def save_model():
            model_name = name_entry.get().strip()
            if not model_name:
                messagebox.showwarning("Attention", "Veuillez entrer un nom pour le modèle")
                return
            
            # Vérifier si le nom existe déjà
            for model in self.model_list:
                if model["name"].lower() == model_name.lower():
                    messagebox.showwarning("Attention", "Un modèle avec ce nom existe déjà")
                    return
            
            model_type = type_var.get()
            accuracy = accuracy_entry.get().strip()
            if not accuracy:
                accuracy = "N/A"
            elif not accuracy.endswith('%'):
                try:
                    float(accuracy)
                    accuracy = f"{accuracy}%"
                except ValueError:
                    accuracy = f"{accuracy}%"
            
            # Ajouter le nouveau modèle
            new_model = {
                "name": model_name,
                "type": model_type,
                "accuracy": accuracy
            }
            self.model_list.append(new_model)
            
            # Mettre à jour l'affichage
            self.update_model_display()
            self.status_label.config(text=f"Modèle '{model_name}' créé avec succès")
            
            dialog.destroy()
        
        def cancel():
            dialog.destroy()
        
        # Boutons
        ttk.Button(
            button_frame,
            text="Annuler",
            command=cancel,
            style='TButton'
        ).pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(
            button_frame,
            text="Créer le modèle",
            command=save_model,
            style='TButton'
        ).pack(side=tk.RIGHT)
        
        # Gestion de la touche Entrée
        dialog.bind('<Return>', lambda e: save_model())
        dialog.bind('<Escape>', lambda e: cancel())
    
    def edit_model(self):
        # Modifier le modèle sélectionné
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Attention", "Veuillez sélectionner un modèle à modifier")
            return
            
        model_name = self.tree.item(selected_item[0])['values'][0]
        self.status_label.config(text=f"Modification du modèle : {model_name}")
        messagebox.showinfo("Modification", f"Modification du modèle '{model_name}'")
    
    def delete_model(self):
        # Supprimer le modèle sélectionné
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showwarning("Attention", "Veuillez sélectionner un modèle à supprimer")
            return
            
        model_name = self.tree.item(selected_item[0])['values'][0]
        
        # Ne pas supprimer le message d'info quand la liste est vide
        if model_name == "Vous n'avez pas de modèle.":
            return
        
        if messagebox.askyesno("Confirmation", f"Êtes-vous sûr de vouloir supprimer le modèle '{model_name}' ?"):
            # Supprimer de la liste interne
            self.model_list = [model for model in self.model_list if model["name"] != model_name]
            
            # Mettre à jour l'affichage
            self.update_model_display()
            self.status_label.config(text=f"Modèle '{model_name}' supprimé")

if __name__ == "__main__":
    app = MLModelManagerApp()
    app.mainloop()