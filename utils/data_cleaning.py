import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import statistics

class DataCleaner:
    """Module de nettoyage automatique des données"""
    
    def __init__(self):
        self.cleaning_report = {}
        self.normalization_params = {}
        self.outlier_info = {}
        
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse complète de la qualité des données"""
        report = {
            'missing_values': self._analyze_missing_values(df),
            'data_types': self._analyze_data_types(df),
            'outliers': self._detect_outliers(df),
            'statistics': self._calculate_statistics(df),
            'recommendations': []
        }
        
        # Générer des recommandations
        report['recommendations'] = self._generate_recommendations(report, df)
        
        self.cleaning_report = report
        return report
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyse des valeurs manquantes"""
        missing_info = {}
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100
            
            missing_info[column] = {
                'count': int(missing_count),
                'percentage': round(missing_percent, 2),
                'has_missing': missing_count > 0
            }
        
        return missing_info
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyse des types de données"""
        type_info = {}
        
        for column in df.columns:
            col_data = df[column].dropna()
            
            if col_data.dtype in ['int64', 'float64']:
                type_info[column] = 'numeric'
            elif col_data.dtype == 'object':
                # Vérifier si c'est numérique mais stocké comme string
                try:
                    pd.to_numeric(col_data.astype(str), errors='raise')
                    type_info[column] = 'numeric_string'
                except:
                    type_info[column] = 'categorical'
            else:
                type_info[column] = 'other'
        
        return type_info
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Détection des outliers avec méthode IQR"""
        outlier_info = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            
            if len(col_data) < 4:  # Pas assez de données
                outlier_info[column] = {
                    'outliers_count': 0,
                    'outliers_indices': [],
                    'method': 'insufficient_data'
                }
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (col_data < lower_bound) | (col_data > upper_bound)
            outliers_indices = col_data[outliers_mask].index.tolist()
            
            outlier_info[column] = {
                'outliers_count': len(outliers_indices),
                'outliers_indices': outliers_indices,
                'lower_bound': float(lower_bound),
                'upper_bound': float(upper_bound),
                'Q1': float(Q1),
                'Q3': float(Q3),
                'IQR': float(IQR),
                'method': 'IQR'
            }
        
        return outlier_info
    
    def _calculate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcul des statistiques descriptives"""
        stats = {}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                stats[column] = {'error': 'no_data'}
                continue
            
            stats[column] = {
                'mean': float(col_data.mean()),
                'median': float(col_data.median()),
                'std': float(col_data.std()) if len(col_data) > 1 else 0.0,
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'count': len(col_data),
                'skewness': float(col_data.skew()) if len(col_data) > 2 else 0.0
            }
        
        return stats
    
    def _generate_recommendations(self, report: Dict[str, Any], df: pd.DataFrame) -> List[str]:
        """Génère des recommandations de nettoyage"""
        recommendations = []
        
        # Recommandations pour les valeurs manquantes
        for column, info in report['missing_values'].items():
            if info['has_missing']:
                if info['percentage'] > 50:
                    recommendations.append(f"❌ Colonne '{column}': {info['percentage']:.1f}% de valeurs manquantes. Considérez la suppression de cette colonne.")
                elif info['percentage'] > 20:
                    recommendations.append(f"⚠️ Colonne '{column}': {info['percentage']:.1f}% de valeurs manquantes. Imputation recommandée.")
                else:
                    recommendations.append(f"💡 Colonne '{column}': {info['percentage']:.1f}% de valeurs manquantes. Imputation possible.")
        
        # Recommandations pour les outliers
        for column, info in report['outliers'].items():
            if info['outliers_count'] > 0:
                outlier_percent = (info['outliers_count'] / len(df)) * 100
                if outlier_percent > 10:
                    recommendations.append(f"⚠️ Colonne '{column}': {info['outliers_count']} outliers ({outlier_percent:.1f}%). Vérifiez les données.")
                else:
                    recommendations.append(f"💡 Colonne '{column}': {info['outliers_count']} outliers détectés.")
        
        # Recommandations pour les types de données
        for column, dtype in report['data_types'].items():
            if dtype == 'numeric_string':
                recommendations.append(f"🔧 Colonne '{column}': Données numériques stockées comme texte. Conversion recommandée.")
        
        return recommendations
    
    def clean_data(self, df: pd.DataFrame, 
                  handle_missing: str = 'auto',
                  handle_outliers: str = 'keep',
                  normalize: bool = True,
                  convert_types: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Nettoyage automatique des données
        
        Args:
            df: DataFrame à nettoyer
            handle_missing: 'auto', 'drop', 'mean', 'median', 'mode', 'keep'
            handle_outliers: 'keep', 'remove', 'cap'
            normalize: Normaliser les données numériques
            convert_types: Convertir les types de données automatiquement
        """
        df_cleaned = df.copy()
        cleaning_log = {
            'actions_performed': [],
            'original_shape': df.shape,
            'columns_processed': []
        }
        
        # 1. Conversion des types de données
        if convert_types:
            df_cleaned, type_log = self._convert_data_types(df_cleaned)
            cleaning_log['actions_performed'].extend(type_log)
        
        # 2. Gestion des valeurs manquantes
        if handle_missing != 'keep':
            df_cleaned, missing_log = self._handle_missing_values(df_cleaned, handle_missing)
            cleaning_log['actions_performed'].extend(missing_log)
        
        # 3. Gestion des outliers
        if handle_outliers != 'keep':
            df_cleaned, outlier_log = self._handle_outliers(df_cleaned, handle_outliers)
            cleaning_log['actions_performed'].extend(outlier_log)
        
        # 4. Normalisation
        if normalize:
            df_cleaned, norm_log = self._normalize_data(df_cleaned)
            cleaning_log['actions_performed'].extend(norm_log)
            cleaning_log['normalization_params'] = self.normalization_params
        
        cleaning_log['final_shape'] = df_cleaned.shape
        cleaning_log['rows_removed'] = df.shape[0] - df_cleaned.shape[0]
        
        return df_cleaned, cleaning_log
    
    def _convert_data_types(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Conversion automatique des types de données"""
        log = []
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Essayer de convertir en numérique
                try:
                    # Supprimer les espaces et caractères spéciaux
                    clean_series = df[column].astype(str).str.strip()
                    clean_series = clean_series.replace(['', 'nan', 'NaN', 'null', 'NULL'], np.nan)
                    
                    # Tenter la conversion
                    numeric_series = pd.to_numeric(clean_series, errors='coerce')
                    
                    # Si au moins 80% des valeurs sont convertibles
                    valid_conversions = numeric_series.notna().sum()
                    total_non_null = clean_series.notna().sum()
                    
                    if total_non_null > 0 and (valid_conversions / total_non_null) >= 0.8:
                        df[column] = numeric_series
                        log.append(f"Converti '{column}' en numérique ({valid_conversions}/{total_non_null} valeurs)")
                
                except Exception as e:
                    log.append(f"Impossible de convertir '{column}': {str(e)}")
        
        return df, log
    
    def _handle_missing_values(self, df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, List[str]]:
        """Gestion des valeurs manquantes"""
        log = []
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
            
            original_count = len(df)
            
            if method == 'auto':
                # Stratégie automatique basée sur le type et le pourcentage
                missing_percent = (missing_count / len(df)) * 100
                
                if missing_percent > 50:
                    # Supprimer la colonne si trop de valeurs manquantes
                    df = df.drop(columns=[column])
                    log.append(f"Supprimé colonne '{column}' ({missing_percent:.1f}% manquant)")
                    continue
                elif df[column].dtype in ['int64', 'float64']:
                    # Imputation par la médiane pour les données numériques
                    median_val = df[column].median()
                    df[column] = df[column].fillna(median_val)
                    log.append(f"Imputé '{column}' avec médiane ({median_val:.2f}) - {missing_count} valeurs")
                else:
                    # Imputation par le mode pour les données catégorielles
                    mode_val = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else 'Unknown'
                    df[column] = df[column].fillna(mode_val)
                    log.append(f"Imputé '{column}' avec mode ({mode_val}) - {missing_count} valeurs")
            
            elif method == 'drop':
                df = df.dropna(subset=[column])
                log.append(f"Supprimé {original_count - len(df)} lignes avec valeurs manquantes dans '{column}'")
            
            elif method == 'mean' and df[column].dtype in ['int64', 'float64']:
                mean_val = df[column].mean()
                df[column] = df[column].fillna(mean_val)
                log.append(f"Imputé '{column}' avec moyenne ({mean_val:.2f}) - {missing_count} valeurs")
            
            elif method == 'median' and df[column].dtype in ['int64', 'float64']:
                median_val = df[column].median()
                df[column] = df[column].fillna(median_val)
                log.append(f"Imputé '{column}' avec médiane ({median_val:.2f}) - {missing_count} valeurs")
            
            elif method == 'mode':
                mode_val = df[column].mode().iloc[0] if len(df[column].mode()) > 0 else 'Unknown'
                df[column] = df[column].fillna(mode_val)
                log.append(f"Imputé '{column}' avec mode ({mode_val}) - {missing_count} valeurs")
        
        return df, log
    
    def _handle_outliers(self, df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, List[str]]:
        """Gestion des outliers"""
        log = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            
            if len(col_data) < 4:
                continue
            
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outliers_count = outliers_mask.sum()
            
            if outliers_count == 0:
                continue
            
            if method == 'remove':
                df = df[~outliers_mask]
                log.append(f"Supprimé {outliers_count} outliers de '{column}'")
            
            elif method == 'cap':
                df.loc[df[column] < lower_bound, column] = lower_bound
                df.loc[df[column] > upper_bound, column] = upper_bound
                log.append(f"Écrêté {outliers_count} outliers de '{column}' (bornes: {lower_bound:.2f} - {upper_bound:.2f})")
        
        return df, log
    
    def _normalize_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Normalisation des données numériques (Min-Max Scaling)"""
        log = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            col_data = df[column].dropna()
            
            if len(col_data) == 0:
                continue
            
            min_val = col_data.min()
            max_val = col_data.max()
            
            if min_val == max_val:
                log.append(f"'{column}' non normalisé (valeurs constantes)")
                continue
            
            # Normalisation Min-Max
            df[column] = (df[column] - min_val) / (max_val - min_val)
            
            # Sauvegarder les paramètres pour la dénormalisation
            self.normalization_params[column] = {
                'min': float(min_val),
                'max': float(max_val),
                'method': 'minmax'
            }
            
            log.append(f"Normalisé '{column}' (min: {min_val:.2f}, max: {max_val:.2f})")
        
        return df, log
    
    def denormalize_data(self, data: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Dénormalisation des données"""
        if not self.normalization_params:
            return data
        
        df_denorm = data.copy()
        
        columns_to_denorm = columns if columns else list(self.normalization_params.keys())
        
        for column in columns_to_denorm:
            if column in self.normalization_params and column in df_denorm.columns:
                params = self.normalization_params[column]
                if params['method'] == 'minmax':
                    df_denorm[column] = df_denorm[column] * (params['max'] - params['min']) + params['min']
        
        return df_denorm
    
    def denormalize_value(self, value: float, column: str) -> float:
        """Dénormalisation d'une valeur unique"""
        if column not in self.normalization_params:
            return value
        
        params = self.normalization_params[column]
        if params['method'] == 'minmax':
            return value * (params['max'] - params['min']) + params['min']
        
        return value
    
    def get_cleaning_summary(self) -> str:
        """Résumé textuel du nettoyage"""
        if not self.cleaning_report:
            return "Aucune analyse effectuée"
        
        summary = []
        summary.append("=== Rapport de qualité des données ===\n")
        
        # Valeurs manquantes
        missing = self.cleaning_report['missing_values']
        total_missing = sum(info['count'] for info in missing.values())
        if total_missing > 0:
            summary.append(f"📊 Valeurs manquantes: {total_missing} au total")
            for col, info in missing.items():
                if info['has_missing']:
                    summary.append(f"  • {col}: {info['count']} ({info['percentage']:.1f}%)")
        else:
            summary.append("✅ Aucune valeur manquante détectée")
        
        summary.append("")
        
        # Outliers
        outliers = self.cleaning_report['outliers']
        total_outliers = sum(info['outliers_count'] for info in outliers.values())
        if total_outliers > 0:
            summary.append(f"📈 Outliers détectés: {total_outliers} au total")
            for col, info in outliers.items():
                if info['outliers_count'] > 0:
                    summary.append(f"  • {col}: {info['outliers_count']} outliers")
        else:
            summary.append("✅ Aucun outlier détecté")
        
        summary.append("")
        
        # Recommandations
        if self.cleaning_report['recommendations']:
            summary.append("📋 Recommandations:")
            for rec in self.cleaning_report['recommendations']:
                summary.append(f"  {rec}")
        
        return "\n".join(summary)
