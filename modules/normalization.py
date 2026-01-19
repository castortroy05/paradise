"""
Normalization module for P.A.R.A.D.I.S.E.
Implements Z-score normalization with league coefficients
"""
import pandas as pd
import numpy as np


class ZScoreNormalizer:
    """
    Z-score normalization with league coefficients.
    Formula: (x - mean) / std * league_coeff
    """
    
    def __init__(self, league_coefficients=None):
        """
        Initialize the normalizer with league coefficients.
        
        Args:
            league_coefficients (dict): Dictionary mapping league names to coefficients.
                                       Defaults to common leagues if not provided.
        """
        if league_coefficients is None:
            self.league_coefficients = {
                'Premier League': 1.0,
                'La Liga': 0.95,
                'Serie A': 0.93,
                'Bundesliga': 0.92,
                'Ligue 1': 0.88,
                'Eredivisie': 0.75,
                'Championship': 0.70,
                'Other': 0.65
            }
        else:
            self.league_coefficients = league_coefficients
    
    def normalize(self, data, columns, league_column='league'):
        """
        Apply Z-score normalization with league coefficients.
        
        Args:
            data (pd.DataFrame): Input dataframe with player statistics
            columns (list): List of column names to normalize
            league_column (str): Name of the column containing league information
            
        Returns:
            pd.DataFrame: Dataframe with normalized columns (suffixed with '_norm')
        """
        result = data.copy()
        
        for col in columns:
            if col not in data.columns:
                print(f"Warning: Column '{col}' not found in data")
                continue
            
            # Calculate mean and std
            mean = data[col].mean()
            std = data[col].std()
            
            # Avoid division by zero
            if std == 0:
                result[f'{col}_norm'] = 0
                continue
            
            # Calculate Z-score
            z_score = (data[col] - mean) / std
            
            # Apply league coefficient if league column exists
            if league_column in data.columns:
                league_coeff = data[league_column].map(
                    lambda x: self.league_coefficients.get(x, self.league_coefficients['Other'])
                )
                result[f'{col}_norm'] = z_score * league_coeff
            else:
                result[f'{col}_norm'] = z_score
        
        return result
    
    def normalize_by_position(self, data, columns, position_column='position', league_column='league'):
        """
        Apply Z-score normalization grouped by position with league coefficients.
        
        Args:
            data (pd.DataFrame): Input dataframe with player statistics
            columns (list): List of column names to normalize
            position_column (str): Name of the column containing position information
            league_column (str): Name of the column containing league information
            
        Returns:
            pd.DataFrame: Dataframe with normalized columns (suffixed with '_norm')
        """
        result = data.copy()
        
        for col in columns:
            if col not in data.columns:
                print(f"Warning: Column '{col}' not found in data")
                continue
            
            normalized_values = []
            
            for position in data[position_column].unique():
                position_mask = data[position_column] == position
                position_data = data[position_mask]
                
                # Calculate mean and std for position
                mean = position_data[col].mean()
                std = position_data[col].std()
                
                # Avoid division by zero
                if std == 0:
                    z_score = pd.Series(0, index=position_data.index)
                else:
                    z_score = (position_data[col] - mean) / std
                
                # Apply league coefficient
                if league_column in data.columns:
                    league_coeff = position_data[league_column].map(
                        lambda x: self.league_coefficients.get(x, self.league_coefficients['Other'])
                    )
                    normalized = z_score * league_coeff
                else:
                    normalized = z_score
                
                normalized_values.append(normalized)
            
            result[f'{col}_norm'] = pd.concat(normalized_values).sort_index()
        
        return result
