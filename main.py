import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.font import Font
import os

# Importer les pages
from pages.home_page import HomePage
from pages.create_model_page import CreateModelPage
from pages.data_preview_page import DataPreviewPage
from pages.model_training_page import ModelTrainingPage
from pages.model_details_page import ModelDetailsPage
from pages.model_manager_page import ModelManagerPage
from pages.network_visualization_page import NetworkVisualizationPage

# Importer les modules utilitaires
from neural_network.model_persistence import ModelPersistence

class MLModelManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ML Model Manager - Réseaux de Neurones")
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
        
        # Liste de modèles sauvegardés
        self.model_list = []        
        # Animation
        self.animation_step = 0
        
        # Label de statut (pour les messages)
        self.status_label = None
        
        self.setup_styles()
        self.create_container()
        self.create_pages()
        self.load_saved_models()
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
          # Label de statut en bas
        self.status_label = tk.Label(
            self,
            text="Prêt",
            bg=self.colors["bg_light"],
            fg=self.colors["text"],
            font=('Helvetica', 9),
            anchor="w"
        )
        self.status_label.pack(side="bottom", fill="x", padx=10, pady=5)
        
    def create_pages(self):
        # Créer toutes les pages
        page_classes = [
            HomePage,
            CreateModelPage, 
            DataPreviewPage,
            ModelTrainingPage,
            ModelDetailsPage,
            ModelManagerPage,
            NetworkVisualizationPage
        ]
        
        for PageClass in page_classes:
            page_name = PageClass.__name__
            page = PageClass(self.container, self)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[page_name] = page
        
    def show_page(self, page_name):
        # Afficher une page spécifique
        page = self.pages[page_name]
        page.tkraise()
        
        # Mettre à jour l'affichage si c'est la page d'accueil
        if page_name == "HomePage":
            page.update_model_display()
        elif page_name == "ModelManagerPage":
            page.refresh_model_list()

    def show_data_preview(self, file_path, model_info):
        """Afficher la page de prévisualisation des données"""
        data_preview_page = self.pages["DataPreviewPage"]
        data_preview_page.load_data(file_path, model_info)
        self.show_page("DataPreviewPage")

    def show_model_training(self, file_path, model_info):
        """Afficher la page d'entraînement du modèle"""
        training_page = self.pages["ModelTrainingPage"]
        training_page.load_training_data(file_path, model_info)
        self.show_page("ModelTrainingPage")

    def show_model_details(self, loaded_model):
        """Afficher la page de détails d'un modèle chargé"""
        details_page = self.pages["ModelDetailsPage"]
        details_page.load_model_details(loaded_model)
        self.show_page("ModelDetailsPage")

    def setup_styles(self):
        # Configuration des styles pour les widgets ttk
        self.style = ttk.Style()
        self.style.theme_use('clam')  # Base theme
        
        # Style pour les boutons avec texte semi-invisible
        self.style.configure(
            'Invisible.TButton',
            foreground=self.colors["text_invisible"],
            background=self.colors["bg_white"],
            borderwidth=0,
            relief="flat"
        )
        
        # Style pour les boutons normaux
        self.style.configure(
            'TButton',
            background=self.colors["bg_white"],
            foreground=self.colors["text"],
            borderwidth=1,
            relief="solid"
        )
        
        # Style pour les boutons d'action
        self.style.configure(
            'Add.TButton',
            background=self.colors["primary"],
            foreground="white",
            borderwidth=0,
            relief="flat"
        )
        
        # Effets hover
        self.style.map('TButton',
            background=[('active', self.colors["bg_light"])],
            relief=[('pressed', 'sunken')]
        )
        
        self.style.map('Add.TButton',
            background=[('active', self.colors["primary_dark"])],
            relief=[('pressed', 'sunken')]
        )
        
        # Style pour les entrées
        self.style.configure(
            'TEntry',
            fieldbackground=self.colors["bg_white"],
            borderwidth=1,
            relief="solid"
        )
          # Style pour les combobox
        self.style.configure(
            'TCombobox',
            fieldbackground=self.colors["bg_white"],
            borderwidth=1,
            relief="solid"
        )

    def animate_title(self):
        """Animation du titre avec changement de couleur"""
        # Animation du titre (si il existe)
        if hasattr(self, 'title_label') and self.title_label.winfo_exists():
            try:
                colors = ["#3498db", "#2980b9", "#5dade2", "#3498db"]
                current_color = colors[self.animation_step % len(colors)]
                self.title_label.config(fg=current_color)
                self.animation_step += 1
            except:
                pass          # Répéter l'animation
        self.after(2000, self.animate_title)

    def update_status(self, message):
        """Mettre à jour le message de statut"""
        if self.status_label:
            self.status_label.config(text=message)
            self.update_idletasks()

    def load_saved_models(self):
        """Charger la liste des modèles sauvegardés"""
        try:
            # Utiliser ModelPersistence pour obtenir la liste des modèles
            model_persistence = ModelPersistence()
            self.model_list = model_persistence.list_saved_models()
            self.update_status(f"Chargé {len(self.model_list)} modèle(s)")
            
        except Exception as e:
            self.update_status(f"Erreur lors du chargement des modèles: {str(e)}")
            self.model_list = []

    def refresh_all_pages(self):
        """Actualiser l'affichage de toutes les pages"""
        try:
            self.load_saved_models()
            if "HomePage" in self.pages:
                self.pages["HomePage"].update_model_display()
            if "ModelManagerPage" in self.pages:
                self.pages["ModelManagerPage"].refresh_model_list()
        except Exception as e:
            self.update_status(f"Erreur lors de l'actualisation: {str(e)}")

if __name__ == "__main__":
    app = MLModelManagerApp()
    app.mainloop()