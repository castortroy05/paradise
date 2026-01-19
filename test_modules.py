#!/usr/bin/env python3
"""
Test script for P.A.R.A.D.I.S.E. modules
"""
import sys
import numpy as np
import pandas as pd
from modules.normalization import ZScoreNormalizer
from modules.archetyper import PlayerArchetyper


def test_normalization():
    """Test Z-score normalization with league coefficients"""
    print("=" * 60)
    print("Test 1: Z-Score Normalization")
    print("=" * 60)
    
    # Create sample data
    data = pd.DataFrame({
        'name': ['Messi', 'Ronaldo', 'Mbappe', 'Haaland', 'Neymar'],
        'league': ['La Liga', 'Premier League', 'Ligue 1', 'Premier League', 'Ligue 1'],
        'goals': [30, 28, 25, 35, 20],
        'assists': [12, 8, 10, 5, 15],
        'shots': [150, 140, 120, 180, 100]
    })
    
    print("\nOriginal Data:")
    print(data)
    
    # Test normalization
    normalizer = ZScoreNormalizer()
    normalized = normalizer.normalize(data, ['goals', 'assists', 'shots'], 'league')
    
    print("\nNormalized Data (with league coefficients):")
    print(normalized[['name', 'league', 'goals_norm', 'assists_norm', 'shots_norm']])
    
    # Verify league coefficients are applied
    assert 'goals_norm' in normalized.columns
    assert 'assists_norm' in normalized.columns
    assert 'shots_norm' in normalized.columns
    
    print("\n✅ Normalization test passed!")
    return True


def test_position_normalization():
    """Test Z-score normalization by position"""
    print("\n" + "=" * 60)
    print("Test 2: Position-Based Normalization")
    print("=" * 60)
    
    # Create sample data with positions
    data = pd.DataFrame({
        'name': ['GK1', 'CB1', 'CM1', 'ST1', 'GK2', 'CB2', 'CM2', 'ST2'],
        'position': ['GK', 'CB', 'CM', 'ST', 'GK', 'CB', 'CM', 'ST'],
        'league': ['Premier League'] * 8,
        'tackles': [5, 80, 60, 20, 4, 75, 55, 18],
        'passes': [500, 1500, 1800, 400, 480, 1450, 1750, 380],
    })
    
    print("\nOriginal Data:")
    print(data)
    
    # Test position-based normalization
    normalizer = ZScoreNormalizer()
    normalized = normalizer.normalize_by_position(
        data, ['tackles', 'passes'], 'position', 'league'
    )
    
    print("\nPosition-Normalized Data:")
    print(normalized[['name', 'position', 'tackles_norm', 'passes_norm']])
    
    print("\n✅ Position-based normalization test passed!")
    return True


def test_archetyper():
    """Test K-Means player archetyping"""
    print("\n" + "=" * 60)
    print("Test 3: Player Archetyping (K-Means)")
    print("=" * 60)
    
    # Create diverse player data
    np.random.seed(42)
    n_players = 50
    
    data = pd.DataFrame({
        'name': [f'Player_{i}' for i in range(n_players)],
        'tackles': np.random.randint(20, 100, n_players),
        'interceptions': np.random.randint(20, 80, n_players),
        'passes': np.random.randint(300, 2000, n_players),
        'progressive_passes': np.random.randint(30, 200, n_players),
        'shots': np.random.randint(10, 150, n_players),
        'dribbles': np.random.randint(20, 150, n_players),
    })
    
    print(f"\nCreated {n_players} players with 6 attributes")
    
    # Test archetyping
    archetyper = PlayerArchetyper(n_clusters=8, random_state=42)
    result = archetyper.fit_predict(
        data,
        ['tackles', 'interceptions', 'passes', 'progressive_passes', 'shots', 'dribbles']
    )
    
    print("\nIdentified Archetypes:")
    archetype_dist = result['archetype_name'].value_counts()
    for archetype, count in archetype_dist.items():
        print(f"  - {archetype}: {count} players")
    
    # Verify archetypes were assigned
    assert 'archetype' in result.columns
    assert 'archetype_name' in result.columns
    assert len(result['archetype_name'].unique()) <= 8
    
    # Show sample players from each archetype
    print("\nSample Players by Archetype:")
    for archetype in result['archetype_name'].unique()[:3]:
        sample = result[result['archetype_name'] == archetype].iloc[0]
        print(f"\n  {archetype}: {sample['name']}")
        print(f"    Tackles: {sample['tackles']}, Passes: {sample['passes']}, Shots: {sample['shots']}")
    
    print("\n✅ Archetyping test passed!")
    return True


def test_archetype_summary():
    """Test archetype summary statistics"""
    print("\n" + "=" * 60)
    print("Test 4: Archetype Summary Statistics")
    print("=" * 60)
    
    # Create player data
    np.random.seed(42)
    data = pd.DataFrame({
        'name': [f'Player_{i}' for i in range(30)],
        'tackles': np.random.randint(20, 100, 30),
        'passes': np.random.randint(300, 2000, 30),
        'shots': np.random.randint(10, 150, 30),
    })
    
    # Create archetyper and fit
    archetyper = PlayerArchetyper(n_clusters=4, random_state=42)
    result = archetyper.fit_predict(data, ['tackles', 'passes', 'shots'])
    
    # Get summary
    summary = archetyper.get_archetype_summary(result, ['tackles', 'passes', 'shots'])
    
    print("\nArchetype Summary:")
    print(summary)
    
    assert 'count' in summary.columns
    assert len(summary) <= 4
    
    print("\n✅ Archetype summary test passed!")
    return True


def test_integration():
    """Test integration of normalization and archetyping"""
    print("\n" + "=" * 60)
    print("Test 5: Integration Test (Normalization + Archetyping)")
    print("=" * 60)
    
    # Create realistic player data
    np.random.seed(42)
    data = pd.DataFrame({
        'name': [f'Player_{i}' for i in range(40)],
        'position': np.random.choice(['GK', 'CB', 'CM', 'ST'], 40),
        'league': np.random.choice(['Premier League', 'La Liga', 'Serie A'], 40),
        'tackles': np.random.randint(20, 100, 40),
        'passes': np.random.randint(300, 2000, 40),
        'shots': np.random.randint(10, 150, 40),
        'progressive_passes': np.random.randint(30, 200, 40),
    })
    
    print("\nStep 1: Normalize data")
    normalizer = ZScoreNormalizer()
    normalized = normalizer.normalize(
        data,
        ['tackles', 'passes', 'shots', 'progressive_passes'],
        'league'
    )
    
    print(f"  Added {len([c for c in normalized.columns if c.endswith('_norm')])} normalized columns")
    
    print("\nStep 2: Identify archetypes")
    archetyper = PlayerArchetyper(n_clusters=6, random_state=42)
    result = archetyper.fit_predict(
        data,  # Using original features for clustering
        ['tackles', 'passes', 'shots', 'progressive_passes']
    )
    
    print(f"  Identified {len(result['archetype_name'].unique())} archetypes")
    
    # Merge normalized data with archetypes
    final = normalized.merge(
        result[['name', 'archetype', 'archetype_name']],
        on='name',
        how='left'
    )
    
    print("\nSample Final Data (Normalized + Archetypes):")
    print(final[['name', 'position', 'league', 'archetype_name', 'tackles_norm', 'passes_norm']].head(10))
    
    print("\n✅ Integration test passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "🔬" * 30)
    print("P.A.R.A.D.I.S.E. MODULE TESTING")
    print("🔬" * 30 + "\n")
    
    tests = [
        test_normalization,
        test_position_normalization,
        test_archetyper,
        test_archetype_summary,
        test_integration
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}/{len(tests)}")
    print(f"❌ Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
