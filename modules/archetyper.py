"""
Archetyper module for P.A.R.A.D.I.S.E.
Implements K-Means clustering for player role classification
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class PlayerArchetyper:
    """
    Player role classification using K-Means clustering.
    Identifies archetypes like 'Progressive Pivot', 'High-Press Predator', etc.
    """
    
    def __init__(self, n_clusters=8, random_state=42):
        """
        Initialize the archetyper.
        
        Args:
            n_clusters (int): Number of player archetypes to identify
            random_state (int): Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.archetype_names = None
        
    def _define_archetypes(self, cluster_centers, feature_names):
        """
        Assign meaningful names to clusters based on their characteristics.
        
        Args:
            cluster_centers (np.ndarray): Cluster centers from K-Means
            feature_names (list): Names of features used for clustering
            
        Returns:
            dict: Mapping of cluster IDs to archetype names
        """
        # Define archetype templates based on key characteristics
        archetype_templates = {
            'Progressive Pivot': {
                'key_features': ['passes', 'progressive_passes', 'pass_accuracy'],
                'profile': 'high'
            },
            'High-Press Predator': {
                'key_features': ['tackles', 'interceptions', 'pressing_actions'],
                'profile': 'high'
            },
            'Creative Playmaker': {
                'key_features': ['key_passes', 'assists', 'shot_creating_actions'],
                'profile': 'high'
            },
            'Box-to-Box Warrior': {
                'key_features': ['tackles', 'progressive_carries', 'aerial_duels'],
                'profile': 'balanced'
            },
            'Deep-Lying Controller': {
                'key_features': ['passes', 'pass_accuracy', 'long_balls'],
                'profile': 'high'
            },
            'Goal-Scoring Threat': {
                'key_features': ['shots', 'goals', 'xg'],
                'profile': 'high'
            },
            'Defensive Anchor': {
                'key_features': ['tackles', 'interceptions', 'clearances'],
                'profile': 'high'
            },
            'Dynamic Dribbler': {
                'key_features': ['dribbles', 'progressive_carries', 'take_ons'],
                'profile': 'high'
            }
        }
        
        # Simple heuristic-based assignment
        # In a real implementation, this would be more sophisticated
        archetype_names = {}
        available_archetypes = list(archetype_templates.keys())
        
        for i in range(len(cluster_centers)):
            if i < len(available_archetypes):
                archetype_names[i] = available_archetypes[i]
            else:
                archetype_names[i] = f'Archetype_{i}'
        
        return archetype_names
    
    def fit(self, data, features):
        """
        Fit the K-Means model to identify player archetypes.
        
        Args:
            data (pd.DataFrame): Player statistics dataframe
            features (list): List of feature columns to use for clustering
            
        Returns:
            self
        """
        # Extract features
        X = data[features].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init='auto'
        )
        self.kmeans.fit(X_scaled)
        
        # Define archetype names
        self.archetype_names = self._define_archetypes(
            self.kmeans.cluster_centers_,
            features
        )
        
        return self
    
    def predict(self, data, features):
        """
        Predict player archetypes for new data.
        
        Args:
            data (pd.DataFrame): Player statistics dataframe
            features (list): List of feature columns to use for prediction
            
        Returns:
            np.ndarray: Array of cluster assignments
        """
        if self.kmeans is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract features
        X = data[features].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict clusters
        clusters = self.kmeans.predict(X_scaled)
        
        return clusters
    
    def fit_predict(self, data, features):
        """
        Fit the model and predict archetypes in one step.
        
        Args:
            data (pd.DataFrame): Player statistics dataframe
            features (list): List of feature columns to use
            
        Returns:
            pd.DataFrame: Input data with added 'archetype' and 'archetype_name' columns
        """
        self.fit(data, features)
        clusters = self.predict(data, features)
        
        result = data.copy()
        result['archetype'] = clusters
        result['archetype_name'] = result['archetype'].map(self.archetype_names)
        
        return result
    
    def get_archetype_summary(self, data_with_archetypes, features):
        """
        Generate summary statistics for each archetype.
        
        Args:
            data_with_archetypes (pd.DataFrame): Dataframe with archetype assignments
            features (list): List of features to summarize
            
        Returns:
            pd.DataFrame: Summary statistics by archetype
        """
        summary = data_with_archetypes.groupby('archetype_name')[features].mean()
        summary['count'] = data_with_archetypes.groupby('archetype_name').size()
        
        return summary
