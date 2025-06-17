import tkinter as tk
from tkinter import ttk, messagebox
import os
import json
from datetime import datetime

# Ajouter le chemin vers neural_network
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_network.model_persistence import ModelPersistence, format_file_size

class HomePage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.model_persistence = ModelPersistence()
        self.saved_models = []
        self.create_widgets()
        self.setup_keyboard_shortcuts()
        self.refresh_models()
        
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
        
        # Section de contrôle avec recherche et boutons d'action
        control_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Barre de recherche à gauche
        search_label = tk.Label(
            control_frame, 
            text="Rechercher:", 
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["text"]
        )
        search_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.search_var = tk.StringVar()
        self.search_var.trace("w", self.filter_models)
        
        self.search_entry = ttk.Entry(
            control_frame, 
            textvariable=self.search_var,
            width=25,
            style="TEntry"
        )
        self.search_entry.pack(side=tk.LEFT, padx=(0, 20))
        
        # Boutons d'action à droite
        ttk.Button(
            control_frame,
            text="🔄 Actualiser",
            command=self.refresh_models,
            style='TButton'
        ).pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(
            control_frame,
            text="📊 Voir Détails",
            command=self.view_model_details,
            style='TButton'
        ).pack(side=tk.RIGHT, padx=(10, 0))
        
        ttk.Button(
            control_frame,
            text="🗑️ Supprimer Sélectionnés",
            command=self.delete_selected_models,
            style='TButton'
        ).pack(side=tk.RIGHT, padx=(10, 0))
        
        # Section de la liste des modèles sauvegardés
        list_frame = tk.LabelFrame(
            main_frame,
            text="Modèles Sauvegardés",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        # Treeview pour la liste des modèles
        tree_frame = tk.Frame(list_frame, bg=self.controller.colors["bg_white"])
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Colonnes du tableau
        columns = ("nom", "date", "architecture", "taille", "dataset")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=12)
        
        # Configuration des en-têtes
        self.tree.heading("nom", text="Nom du Modèle")
        self.tree.heading("date", text="Date de création")
        self.tree.heading("architecture", text="Architecture")
        self.tree.heading("taille", text="Taille")
        self.tree.heading("dataset", text="Dataset")
        
        # Configuration des largeurs de colonnes
        self.tree.column("nom", width=200, minwidth=150)
        self.tree.column("date", width=150, minwidth=120)
        self.tree.column("architecture", width=120, minwidth=100)
        self.tree.column("taille", width=80, minwidth=60)
        self.tree.column("dataset", width=180, minwidth=150)
        
        # Scrollbars
        v_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        h_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        # Placement des widgets
        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
        
        # Events
        self.tree.bind("<Double-1>", self.on_tree_double_click)
        self.tree.bind("<ButtonRelease-1>", self.on_tree_click)
        
        # Section des statistiques
        stats_frame = tk.Frame(list_frame, bg=self.controller.colors["bg_white"])
        stats_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.stats_label = tk.Label(
            stats_frame,
            text="",
            font=('Helvetica', 9),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            anchor="w"
        )
        self.stats_label.pack(side=tk.LEFT)
        
        # Section des détails du modèle sélectionné
        details_frame = tk.LabelFrame(
            list_frame,
            text="Détails du Modèle Sélectionné",
            font=('Helvetica', 10, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=15,
            pady=10
        )
        details_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Zone de texte pour les détails
        details_text_frame = tk.Frame(details_frame, bg=self.controller.colors["bg_white"])
        details_text_frame.pack(fill=tk.BOTH, expand=True)
        
        self.details_text = tk.Text(
            details_text_frame,
            height=6,
            font=('Courier', 9),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["text"],
            wrap=tk.WORD,
            state=tk.DISABLED
        )
        details_scrollbar = ttk.Scrollbar(details_text_frame, orient=tk.VERTICAL, command=self.details_text.yview)
        self.details_text.configure(yscrollcommand=details_scrollbar.set)
        
        self.details_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Boutons principaux en bas
        main_buttons_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        main_buttons_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Bouton pour créer un nouveau modèle (proéminent)
        self.new_model_button = ttk.Button(
            main_buttons_frame,
            text="➕ Créer un Nouveau Modèle",
            command=lambda: self.controller.show_page("CreateModelPage"),
            style='Add.TButton',
            cursor="hand2"
        )
        self.new_model_button.pack(side=tk.LEFT, padx=(0, 15))
        
        # Bouton pour la visualisation des réseaux
        self.visualize_button = ttk.Button(
            main_buttons_frame,
            text="🧠 Visualisateur de Réseaux",
            command=lambda: self.controller.show_page("NetworkVisualizationPage"),
            style='TButton',
            cursor="hand2"
        )
        self.visualize_button.pack(side=tk.LEFT)
        
    def refresh_models(self):
        """Actualiser la liste des modèles sauvegardés"""
        try:
            self.saved_models = self.model_persistence.list_saved_models()
            self.update_model_display()
            self.update_statistics()
            self.controller.update_status(f"Actualisé - {len(self.saved_models)} modèle(s) trouvé(s)")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement des modèles: {str(e)}")
            
    def update_model_display(self):
        """Mettre à jour l'affichage de la liste des modèles"""
        # Effacer les éléments existants
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Filtrer les modèles selon la recherche
        search_text = self.search_var.get().lower() if hasattr(self, 'search_var') else ""
        filtered_models = []
        
        for model in self.saved_models:
            model_info = model.get('model_info', {})
            model_name = model_info.get('name', 'Sans nom').lower()
            if search_text == "" or search_text in model_name:
                filtered_models.append(model)
        
        # Afficher les modèles filtrés
        if filtered_models:
            for model in filtered_models:
                # Extraire les informations depuis la structure correcte
                model_info = model.get('model_info', {})
                network_arch = model.get('network_architecture', {})
                file_info = model.get('file_info', {})
                file_size_info = model.get('file_size', {})
                
                # Formatage des informations
                name = model_info.get('name', 'Sans nom')
                
                # Date de création depuis file_info
                date_str = file_info.get('saved_at', 'N/A')
                if date_str != 'N/A':
                    try:
                        date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                        date_str = date_obj.strftime('%d/%m/%Y %H:%M')
                    except:
                        # Essayer le format alternatif depuis model_info
                        date_str = model_info.get('created_date', 'N/A')
                
                # Architecture du réseau
                layers_config = network_arch.get('layers_config', [])
                if layers_config:
                    arch_str = f"{len(layers_config)} couches ({'-'.join(map(str, layers_config))})"
                else:
                    arch_str = "N/A"
                
                # Taille des fichiers
                total_size = file_size_info.get('total_size', 0)
                size_str = format_file_size(total_size)
                
                # Dataset utilisé
                dataset = model_info.get('data_file', 'N/A')
                if dataset and len(dataset) > 20:
                    dataset = "..." + dataset[-17:]
                
                # Insérer dans le tree
                self.tree.insert("", tk.END, values=(name, date_str, arch_str, size_str, dataset), tags=(model['filename'],))
        else:
            # Message si aucun modèle trouvé
            if search_text:
                self.tree.insert("", tk.END, values=("Aucun modèle trouvé pour cette recherche", "", "", "", ""))
            else:
                self.tree.insert("", tk.END, values=("Aucun modèle sauvegardé", "Créez votre premier modèle !", "", "", ""))

    def filter_models(self, *args):
        """Filtrer les modèles selon la recherche"""
        self.update_model_display()
        
    def on_tree_click(self, event):
        """Gestionnaire de clic sur le tree"""
        selection = self.tree.selection()
        if selection:
            item = self.tree.item(selection[0])
            values = item['values']
            if len(values) > 0 and values[0] not in ["Aucun modèle trouvé pour cette recherche", "Aucun modèle sauvegardé"]:
                # Trouver le modèle correspondant
                for model in self.saved_models:
                    model_info = model.get('model_info', {})
                    if model_info.get('name') == values[0]:
                        self.show_model_details(model)
                        break
                        
    def on_tree_double_click(self, event):
        """Gestionnaire de double-clic pour voir les détails"""
        self.view_model_details()
        
    def show_model_details(self, model):
        """Afficher les détails d'un modèle dans la zone de texte"""
        model_info = model.get('model_info', {})
        network_arch = model.get('network_architecture', {})
        file_info = model.get('file_info', {})
        file_size_info = model.get('file_size', {})
        
        self.update_details(f"""
INFORMATIONS DU MODÈLE : {model_info.get('name', 'Sans nom')}
{'='*50}

📊 ARCHITECTURE
- Couches : {network_arch.get('layers_config', 'N/A')}
- Paramètres totaux : {network_arch.get('total_layers', 'N/A')} couches
- Fonction d'activation : {network_arch.get('activation_function', 'N/A')}

📈 ENTRAÎNEMENT  
- Dataset : {model_info.get('data_file', 'N/A')}
- Taille des données : {model_info.get('data_shape', 'N/A')}
- Variables cibles : {model_info.get('target_column', 'N/A')}
- Variables d'entrée : {len(model_info.get('feature_columns', []))} variables

📁 FICHIERS
- Taille totale : {format_file_size(file_size_info.get('total_size', 0))}
- Date de création : {file_info.get('saved_at', 'N/A')}
- Nom de fichier : {model.get('filename', 'N/A')}

📝 DESCRIPTION
{model_info.get('accuracy', 'Aucune description disponible')}
        """.strip())
        
    def update_details(self, text):
        """Mettre à jour la zone de détails"""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, text)
        self.details_text.config(state=tk.DISABLED)
        
    def update_statistics(self):
        """Mettre à jour les statistiques"""
        total_models = len(self.saved_models)
        total_size = sum(model.get('file_size', {}).get('total_size', 0) for model in self.saved_models)
        
        stats_text = f"Total : {total_models} modèle(s) | Taille totale : {format_file_size(total_size)}"
        self.stats_label.config(text=stats_text)
        
    def delete_selected_models(self):
        """Supprimer les modèles sélectionnés"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Sélection", "Veuillez sélectionner un ou plusieurs modèles à supprimer.")
            return
            
        # Confirmer la suppression
        selected_names = []
        selected_filenames = []
        
        for item_id in selection:
            item = self.tree.item(item_id)
            values = item['values']
            if len(values) > 0 and values[0] not in ["Aucun modèle trouvé pour cette recherche", "Aucun modèle sauvegardé"]:
                selected_names.append(values[0])
                if item['tags']:
                    selected_filenames.append(item['tags'][0])
        
        if not selected_names:
            messagebox.showwarning("Sélection", "Aucun modèle valide sélectionné.")
            return
            
        # Confirmation
        if len(selected_names) == 1:
            message = f"Êtes-vous sûr de vouloir supprimer le modèle '{selected_names[0]}' ?"
        else:
            message = f"Êtes-vous sûr de vouloir supprimer les {len(selected_names)} modèles sélectionnés ?"
            
        if messagebox.askyesno("Confirmation", message):
            try:
                deleted_count = 0
                for filename in selected_filenames:
                    result = self.model_persistence.delete_model(filename)
                    if result.get('success', False):
                        deleted_count += 1
                        
                self.refresh_models()
                self.controller.refresh_all_pages()
                
                if deleted_count > 0:
                    messagebox.showinfo("Succès", f"{deleted_count} modèle(s) supprimé(s) avec succès.")
                else:
                    messagebox.showwarning("Attention", "Aucun modèle n'a pu être supprimé.")
                    
            except Exception as e:
                messagebox.showerror("Erreur", f"Erreur lors de la suppression : {str(e)}")
                
    def view_model_details(self):
        """Voir les détails complets d'un modèle sélectionné"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Sélection", "Veuillez sélectionner un modèle.")
            return
            
        item = self.tree.item(selection[0])
        values = item['values']
        
        if len(values) > 0 and values[0] not in ["Aucun modèle trouvé pour cette recherche", "Aucun modèle sauvegardé"]:
            # Trouver le modèle correspondant
            for model in self.saved_models:
                model_info = model.get('model_info', {})
                if model_info.get('name') == values[0]:
                    try:
                        # Charger le modèle complet
                        loaded_model = self.model_persistence.load_model(model['filename'])
                        if loaded_model.get('success', False):
                            self.controller.show_model_details(loaded_model)
                        else:
                            messagebox.showerror("Erreur", f"Erreur lors du chargement du modèle : {loaded_model.get('error', 'Erreur inconnue')}")
                    except Exception as e:
                        messagebox.showerror("Erreur", f"Erreur lors du chargement du modèle : {str(e)}")
                    break
        else:
            messagebox.showwarning("Sélection", "Veuillez sélectionner un modèle valide.")

    def setup_keyboard_shortcuts(self):
        """Configurer les raccourcis clavier pour la page"""
        # F5 pour rafraîchir
        self.bind_all("<F5>", lambda e: self.refresh_models())
        
        # Suppr pour supprimer le modèle sélectionné
        self.bind_all("<Delete>", lambda e: self.delete_selected_models())
        
        # Enter pour ouvrir les détails du modèle
        self.tree.bind("<Return>", self.on_tree_double_click)
        
        # Double-clic pour ouvrir les détails
        self.tree.bind("<Double-1>", self.on_tree_double_click)
