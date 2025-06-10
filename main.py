import tkinter as tk
from tkinter import ttk
from pages.home_page import HomePage
from pages.create_model_page import CreateModelPage
from pages.data_preview_page import DataPreviewPage
from pages.model_training_page import ModelTrainingPage
from pages.model_details_page import ModelDetailsPage

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
        
        # Liste de modèles (vide au départ)
        self.model_list = []
        
        # Animation
        self.animation_step = 0
        
        self.setup_styles()
        self.create_container()
        self.create_pages()
        self.create_status_bar()
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
        page_classes = {
            "HomePage": HomePage,
            "CreateModelPage": CreateModelPage,
            "DataPreviewPage": DataPreviewPage,
            "ModelTrainingPage": ModelTrainingPage,
            "ModelDetailsPage": ModelDetailsPage
        }
        
        for page_name, PageClass in page_classes.items():
            page = PageClass(self.container, self)
            self.pages[page_name] = page
            page.grid(row=0, column=0, sticky="nsew")
            
    def show_page(self, page_name):
        # Afficher la page spécifiée
        page = self.pages[page_name]
        page.tkraise()
        
        # Mettre à jour l'affichage si c'est la page d'accueil
        if page_name == "HomePage":
            page.update_model_display()
    
    def show_data_preview(self, file_path, model_info):
        """Aller à la page de prévisualisation avec les données CSV"""
        self.show_page("DataPreviewPage")
        self.pages["DataPreviewPage"].load_data(file_path, model_info)
        
    def show_model_training(self, file_path, model_info):
        """Aller à la page d'entraînement avec les données configurées"""
        self.show_page("ModelTrainingPage")
        self.pages["ModelTrainingPage"].load_training_data(file_path, model_info)
        
    def show_model_details(self, model_info):
        """Aller à la page de détails d'un modèle"""
        self.show_page("ModelDetailsPage")
        self.pages["ModelDetailsPage"].load_model_details(model_info)
        
    def create_status_bar(self):
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

    def setup_styles(self):
        # Configuration des styles pour les widgets ttk
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Style pour les boutons avec texte semi-invisible
        self.style.configure(
            'TButton',
            background=self.colors["primary"],
            foreground=self.colors["text_invisible"],
            font=('Helvetica', 10, 'bold'),
            padding=10
        )
        
        # Style pour le survol des boutons
        self.style.map('TButton',
            background=[('active', self.colors["primary_dark"])],
            foreground=[('active', 'white')],
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

    def animate_title(self):
        # Animation pour le titre (seulement sur la page d'accueil)
        if hasattr(self, 'title_label'):
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

if __name__ == "__main__":
    app = MLModelManagerApp()
    app.mainloop()