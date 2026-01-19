#!/usr/bin/env python3
"""
Example usage of P.A.R.A.D.I.S.E. modules
Demonstrates normalization and archetyping functionality
"""
import pandas as pd
import numpy as np
from modules.normalization import ZScoreNormalizer
from modules.archetyper import PlayerArchetyper


def main():
    print("=" * 70)
    print("P.A.R.A.D.I.S.E. - Example Usage")
    print("=" * 70)
    
    # Create sample player data
    print("\n1. Creating sample player dataset...")
    np.random.seed(42)
    
    players_data = {
        'name': ['De Bruyne', 'Rodri', 'Casemiro', 'Bruno', 'Haaland', 
                 'Kane', 'Salah', 'Saka', 'Van Dijk', 'Dias'],
        'position': ['CM', 'CDM', 'CDM', 'CAM', 'ST', 
                     'ST', 'RW', 'RW', 'CB', 'CB'],
        'league': ['Premier League', 'Premier League', 'Premier League', 
                   'Premier League', 'Premier League', 'Premier League', 
                   'Premier League', 'Premier League', 'Premier League', 'Premier League'],
        'tackles': [40, 95, 88, 35, 20, 25, 30, 45, 80, 75],
        'interceptions': [35, 80, 75, 30, 15, 20, 25, 40, 90, 85],
        'passes': [2200, 2500, 1800, 1900, 600, 900, 1200, 1400, 1800, 2000],
        'pass_accuracy': [88.5, 91.2, 87.3, 85.6, 72.4, 78.9, 82.1, 84.3, 89.7, 90.1],
        'progressive_passes': [180, 140, 100, 160, 50, 70, 90, 110, 130, 120],
        'key_passes': [85, 40, 25, 90, 20, 45, 70, 60, 15, 12],
        'shots': [60, 20, 15, 80, 150, 120, 140, 90, 10, 8],
        'goals': [8, 2, 1, 10, 35, 28, 25, 15, 2, 1],
        'dribbles': [70, 25, 20, 80, 40, 35, 110, 95, 15, 12],
        'progressive_carries': [95, 60, 45, 100, 55, 50, 120, 105, 35, 30],
    }
    
    df = pd.DataFrame(players_data)
    print(f"Created dataset with {len(df)} players")
    print(df[['name', 'position', 'tackles', 'passes', 'goals']].to_string(index=False))
    
    # Normalization
    print("\n" + "=" * 70)
    print("2. Applying Z-Score Normalization with League Coefficients")
    print("=" * 70)
    
    normalizer = ZScoreNormalizer()
    
    # Normalize key metrics
    metrics_to_normalize = ['tackles', 'passes', 'progressive_passes', 'shots', 'goals']
    normalized_df = normalizer.normalize(df, metrics_to_normalize, 'league')
    
    print("\nNormalized Statistics (sample):")
    norm_cols = ['name', 'position'] + [f'{m}_norm' for m in metrics_to_normalize]
    print(normalized_df[norm_cols].head().to_string(index=False))
    
    # Archetyping
    print("\n" + "=" * 70)
    print("3. Player Archetyping with K-Means Clustering")
    print("=" * 70)
    
    # Features for clustering
    clustering_features = [
        'tackles', 'interceptions', 'passes', 'progressive_passes',
        'key_passes', 'shots', 'goals', 'dribbles', 'progressive_carries'
    ]
    
    archetyper = PlayerArchetyper(n_clusters=5, random_state=42)
    archetyped_df = archetyper.fit_predict(df, clustering_features)
    
    print("\nPlayer Archetypes:")
    result_cols = ['name', 'position', 'archetype_name']
    print(archetyped_df[result_cols].to_string(index=False))
    
    print("\n" + "=" * 70)
    print("4. Archetype Summary Statistics")
    print("=" * 70)
    
    summary = archetyper.get_archetype_summary(archetyped_df, clustering_features)
    print("\nAverage stats by archetype:")
    summary_display = summary[['tackles', 'passes', 'shots', 'goals', 'count']].round(1)
    print(summary_display)
    
    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)
    print("\nTo run the interactive dashboard:")
    print("  streamlit run app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
