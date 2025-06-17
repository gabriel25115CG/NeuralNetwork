import tkinter as tk
from tkinter import ttk, messagebox
import sys
import os
from datetime import datetime

# Ajouter le chemin vers neural_network
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from neural_network.model_persistence import ModelPersistence, format_file_size

class ModelManagerPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=controller.colors["bg_light"])
        self.controller = controller
        self.model_persistence = ModelPersistence()
        self.saved_models = []
        self.create_widgets()
        self.refresh_models()
        
    def create_widgets(self):
        # Cadre principal
        main_frame = tk.Frame(self, bg=self.controller.colors["bg_light"])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # En-tête
        header_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        ttk.Button(
            header_frame,
            text="← Retour",
            command=lambda: self.controller.show_page("HomePage"),
            style='TButton'
        ).pack(side=tk.LEFT)
        
        tk.Label(
            main_frame,
            text="Gestionnaire de Modèles Sauvegardés",
            font=('Helvetica', 18, 'bold'),
            bg=self.controller.colors["bg_light"],
            fg=self.controller.colors["primary"]
        ).pack(pady=(0, 20))
        
        # Section de contrôle
        control_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Button(
            control_frame,
            text="🔄 Actualiser",
            command=self.refresh_models,
            style='TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            control_frame,
            text="🗑️ Supprimer Sélectionnés",
            command=self.delete_selected_models,
            style='TButton'
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(
            control_frame,
            text="📊 Voir Détails",
            command=self.view_model_details,
            style='TButton'
        ).pack(side=tk.LEFT)
        
        # Section de la liste des modèles
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
        
        # Colonnes du Treeview
        columns = ("name", "accuracy", "target", "features", "created", "size")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10)
        
        # Configuration des colonnes
        self.tree.heading("name", text="Nom du Modèle")
        self.tree.heading("accuracy", text="Précision")
        self.tree.heading("target", text="Variable Cible")
        self.tree.heading("features", text="Variables")
        self.tree.heading("created", text="Date de Création")
        self.tree.heading("size", text="Taille")
        
        self.tree.column("name", width=200)
        self.tree.column("accuracy", width=80)
        self.tree.column("target", width=150)
        self.tree.column("features", width=80)
        self.tree.column("created", width=130)
        self.tree.column("size", width=80)
        
        # Scrollbar pour le Treeview
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bind pour sélection multiple
        self.tree.bind('<Button-1>', self.on_tree_click)
        
        # Section d'informations détaillées
        details_frame = tk.LabelFrame(
            main_frame,
            text="Détails du Modèle Sélectionné",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        details_frame.pack(fill=tk.X, pady=(0, 20))
        
        self.details_text = tk.Text(
            details_frame,
            height=6,
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Courier', 9),
            state=tk.DISABLED,
            wrap=tk.WORD
        )
        self.details_text.pack(fill=tk.X, pady=(10, 10))
        
        # Section de statistiques
        stats_frame = tk.LabelFrame(
            main_frame,
            text="Statistiques",
            font=('Helvetica', 12, 'bold'),
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            padx=20,
            pady=15
        )
        stats_frame.pack(fill=tk.X)
        
        self.stats_label = tk.Label(
            stats_frame,
            text="Chargement des statistiques...",
            bg=self.controller.colors["bg_white"],
            fg=self.controller.colors["text"],
            font=('Helvetica', 10)
        )
        self.stats_label.pack(pady=10)
        
    def refresh_models(self):
        """Actualiser la liste des modèles sauvegardés"""
        try:
            # Charger les modèles depuis le système de persistence
            self.saved_models = self.model_persistence.list_saved_models()
            
            # Nettoyer le Treeview
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Remplir le Treeview
            for model in self.saved_models:
                model_info = model["model_info"]
                file_info = model["file_info"]
                file_sizes = model["file_size"]
                
                # Formater les données
                name = model_info.get("name", "Nom inconnu")
                accuracy = model_info.get("accuracy", "N/A")
                target = model_info.get("target_column", "N/A")
                feature_columns = model_info.get("feature_columns") or []
                n_features = len(feature_columns)
                created = model_info.get("created_date", "Date inconnue")
                size = format_file_size(file_sizes.get("total_size", 0))
                
                # Insérer dans le Treeview
                self.tree.insert("", "end", values=(name, accuracy, target, n_features, created, size))
            
            # Mettre à jour les statistiques
            self.update_statistics()
            
            # Nettoyer les détails
            self.update_details("")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger les modèles:\n{str(e)}")
    
    def on_tree_click(self, event):
        """Gérer le clic sur un élément du Treeview"""
        selection = self.tree.selection()
        if selection:
            # Récupérer l'index du modèle sélectionné
            item = selection[0]
            index = self.tree.index(item)
            
            if 0 <= index < len(self.saved_models):
                model = self.saved_models[index]
                self.show_model_details(model)
    
    def show_model_details(self, model):
        """Afficher les détails d'un modèle"""
        try:
            # Générer le résumé du modèle
            summary = self.model_persistence.export_model_summary(model["filename"])
            self.update_details(summary)
            
        except Exception as e:
            self.update_details(f"Erreur lors du chargement des détails:\n{str(e)}")
    
    def update_details(self, text):
        """Mettre à jour le texte des détails"""
        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, text)
        self.details_text.config(state=tk.DISABLED)
    
    def update_statistics(self):
        """Mettre à jour les statistiques"""
        try:
            total_models = len(self.saved_models)
            
            if total_models == 0:
                self.stats_label.config(text="📊 Aucun modèle sauvegardé")
                return
            
            # Calculer les statistiques
            total_size = sum(model["file_size"].get("total_size", 0) for model in self.saved_models)
            
            # Compter par type (tous sont des réseaux de neurones pour le moment)
            neural_networks = total_models
            
            # Trouver le modèle le plus récent
            try:
                most_recent = max(
                    self.saved_models,
                    key=lambda x: datetime.fromisoformat(x["file_info"].get("saved_at", "1970-01-01T00:00:00"))
                )
                recent_name = most_recent["model_info"].get("name", "Nom inconnu")
            except:
                recent_name = "Aucun"
            
            # Afficher les statistiques
            stats_text = f"""📊 {total_models} modèle(s) • 🧠 {neural_networks} réseau(x) de neurones • 💾 {format_file_size(total_size)} • 🕒 Plus récent: {recent_name}"""
            
            self.stats_label.config(text=stats_text)
            
        except Exception as e:
            self.stats_label.config(text=f"Erreur dans les statistiques: {str(e)}")
    
    def delete_selected_models(self):
        """Supprimer les modèles sélectionnés"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Attention", "Aucun modèle sélectionné")
            return
        
        # Confirmer la suppression
        count = len(selection)
        if not messagebox.askyesno(
            "Confirmer la suppression",
            f"Êtes-vous sûr de vouloir supprimer {count} modèle(s) ?\n"
            "Cette action est irréversible."
        ):
            return
        
        try:
            deleted_count = 0
            errors = []
            
            for item in selection:
                index = self.tree.index(item)
                if 0 <= index < len(self.saved_models):
                    model = self.saved_models[index]
                    filename = model["filename"]
                    
                    # Supprimer le modèle
                    result = self.model_persistence.delete_model(filename)
                    
                    if result["success"]:
                        deleted_count += 1
                    else:
                        errors.append(f"{model['model_info'].get('name', filename)}: {result['error']}")
            
            # Afficher le résultat
            if errors:
                messagebox.showwarning(
                    "Suppression partielle",
                    f"{deleted_count} modèle(s) supprimé(s) avec succès.\n\n"
                    f"Erreurs:\n" + "\n".join(errors)
                )
            else:
                messagebox.showinfo(
                    "Suppression réussie",
                    f"{deleted_count} modèle(s) supprimé(s) avec succès"
                )
            
            # Actualiser la liste
            self.refresh_models()
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la suppression:\n{str(e)}")
    
    def view_model_details(self):
        """Voir les détails complets d'un modèle dans une nouvelle fenêtre"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("Attention", "Aucun modèle sélectionné")
            return
        
        item = selection[0]
        index = self.tree.index(item)
        
        if 0 <= index < len(self.saved_models):
            model = self.saved_models[index]
            self.open_details_window(model)
    
    def open_details_window(self, model):
        """Ouvrir une fenêtre de détails pour un modèle"""
        try:
            # Créer une nouvelle fenêtre
            details_window = tk.Toplevel(self)
            details_window.title(f"Détails - {model['model_info'].get('name', 'Modèle')}")
            details_window.geometry("600x500")
            details_window.configure(bg=self.controller.colors["bg_light"])
            
            # Cadre principal
            main_frame = tk.Frame(details_window, bg=self.controller.colors["bg_light"])
            main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
            
            # Titre
            tk.Label(
                main_frame,
                text=f"Détails du Modèle: {model['model_info'].get('name', 'Inconnu')}",
                font=('Helvetica', 14, 'bold'),
                bg=self.controller.colors["bg_light"],
                fg=self.controller.colors["primary"]
            ).pack(pady=(0, 20))
            
            # Zone de texte avec scrollbar
            text_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            text_widget = tk.Text(
                text_frame,
                bg=self.controller.colors["bg_white"],
                fg=self.controller.colors["text"],
                font=('Courier', 10),
                wrap=tk.WORD,
                state=tk.DISABLED
            )
            
            scrollbar_details = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar_details.set)
            
            text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar_details.pack(side=tk.RIGHT, fill=tk.Y)
            
            # Charger et afficher les détails
            summary = self.model_persistence.export_model_summary(model["filename"])
            
            text_widget.config(state=tk.NORMAL)
            text_widget.insert(1.0, summary)
            text_widget.config(state=tk.DISABLED)
            
            # Boutons
            button_frame = tk.Frame(main_frame, bg=self.controller.colors["bg_light"])
            button_frame.pack(fill=tk.X, pady=(20, 0))
            
            ttk.Button(
                button_frame,
                text="Fermer",
                command=details_window.destroy,
                style='TButton'
            ).pack(side=tk.RIGHT)
            
            ttk.Button(
                button_frame,
                text="🔄 Charger ce Modèle",
                command=lambda: self.load_model_for_prediction(model, details_window),
                style='Add.TButton'
            ).pack(side=tk.RIGHT, padx=(0, 10))            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible d'ouvrir les détails:\n{str(e)}")
    
    def load_model_for_prediction(self, model, window):
        """Charger un modèle pour faire des prédictions"""
        try:
            # Charger le modèle complet
            loaded_model = self.model_persistence.load_model(model["filename"])
            
            if not loaded_model["success"]:
                messagebox.showerror("Erreur", f"Impossible de charger le modèle:\n{loaded_model['error']}")
                return
            
            # Stocker le modèle chargé dans le contrôleur
            self.controller.loaded_model = {
                "model_info": loaded_model["model_info"],
                "neural_network": loaded_model["neural_network"],
                "normalization_params": loaded_model["normalization_params"],
                "filename": model["filename"]
            }
            
            messagebox.showinfo(
                "Succès",
                f"Modèle '{loaded_model['model_info']['name']}' chargé avec succès!\n\n"
                "Redirection vers la page de détails avec visualisations..."
            )
            
            window.destroy()
            
            # Rediriger vers la page de détails avec le modèle chargé
            self.controller.show_model_details(loaded_model)
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger le modèle:\n{str(e)}")
            
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors du chargement:\n{str(e)}")
