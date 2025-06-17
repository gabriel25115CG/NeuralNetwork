"""
🧠 GUIDE POUR CHOISIR L'ARCHITECTURE D'UN RÉSEAU DE NEURONES
═══════════════════════════════════════════════════════════

📊 RÈGLES EMPIRIQUES BASÉES SUR LA TAILLE DU DATASET :

1. DATASET PETIT (< 100 lignes) :
   ├─ Architecture: [n_features, 1] (linéaire)
   ├─ Ou maximum: [n_features, n_features//2, 1]
   └─ Risque: Overfitting si trop complexe

2. DATASET MOYEN (100-1000 lignes) :
   ├─ Architecture: [n_features, n_features*2, n_features, 1]
   ├─ Exemple: [7, 14, 7, 1] ou [7, 10, 5, 1]
   └─ Sweet spot: 1-2 couches cachées

3. DATASET GRAND (1000+ lignes) :
   ├─ Architecture: [n_features, n_features*3, n_features*2, n_features, 1]
   ├─ Exemple: [7, 21, 14, 7, 1]
   └─ Permet: 3+ couches cachées

🎯 RÈGLES PAR COMPLEXITÉ DU PROBLÈME :

PROBLÈME LINÉAIRE (comme immobilier simple) :
├─ surface + pièces = prix ➜ Relation directe
├─ Architecture: [n_features, 1] ou [n_features, 3-5, 1]
└─ Éviter: Trop de couches

PROBLÈME NON-LINÉAIRE (reconnaissance, NLP) :
├─ Relations complexes, interactions multiples
├─ Architecture: [n_features, hidden1, hidden2, ..., 1]
└─ Besoin: Plus de couches

📏 FORMULES PRATIQUES :

Neurones couche cachée = n_features * 1.5 à 3
Nombre de couches = log2(n_samples/10)
Paramètres totaux < n_samples/5 (éviter overfitting)

🧪 MÉTHODE DE VALIDATION :

1. Tester 3-5 architectures différentes
2. Utiliser validation croisée 
3. Surveiller overfitting (loss test vs train)
4. Choisir le plus simple qui marche bien
"""
