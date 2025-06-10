import tkinter as tk
from tkinter import ttk, messagebox
from tkinter.font import Font

class MLModelManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Model Manager")
        self.geometry("900x600")  # Écran plus grand
        
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
        
        # Exemple de liste de modèles
        self.model_list = [
            {"name": "Réseau de neurones", "type": "Classification", "accuracy": "92.5%"},
            {"name": "Random Forest", "type": "Régression", "accuracy": "88.7%"},
            {"name": "SVM", "type": "Classification", "accuracy": "90.1%"}
        ]
        
        # Animation
        self.animation_step = 0
        
        self.setup_styles()
        self.create_widgets()
        self.animate_title()

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
        
        # Insertion des données
        for model in self.model_list:
            self.tree.insert("", tk.END, values=(model["name"], model["type"], model["accuracy"]))
        
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
        
        # Boutons d'action avec curseur "hand2" (main)
        self.new_model_button = ttk.Button(
            button_frame,
            text="Créer un nouveau modèle",
            command=self.create_new_model,
            style='TButton',
            cursor="hand2"  # Pointeur main au survol
        )
        self.new_model_button.pack(side=tk.LEFT, padx=(0, 10))
        
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
        
        # Filtrer et réinsérer les modèles qui correspondent
        for model in self.model_list:
            if search_text in model["name"].lower():
                self.tree.insert("", tk.END, values=(model["name"], model["type"], model["accuracy"]))
        
        # Mettre à jour le statut
        if search_text:
            self.status_label.config(text=f"Recherche: '{search_text}'")
        else:
            self.status_label.config(text="Prêt")

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
        # Placeholder pour la création de modèle
        self.status_label.config(text="Création d'un nouveau modèle...")
        messagebox.showinfo("Création de modèle", "Fonctionnalité de création de modèle à implémenter")
    
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
        if messagebox.askyesno("Confirmation", f"Êtes-vous sûr de vouloir supprimer le modèle '{model_name}' ?"):
            self.tree.delete(selected_item[0])
            self.status_label.config(text=f"Modèle '{model_name}' supprimé")

if __name__ == "__main__":
    app = MLModelManagerApp()
    app.mainloop()