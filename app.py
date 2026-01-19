"""
P.A.R.A.D.I.S.E. Streamlit Application
Player Analysis, Recruitment, Archetyping & Data-driven Identification for Squad Efficiency
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import sys
import os

# Add modules to path
sys.path.insert(0, os.path.dirname(__file__))

from modules.normalization import ZScoreNormalizer
from modules.archetyper import PlayerArchetyper

# Try to import mplsoccer, but provide fallback
try:
    from mplsoccer import Radar, FontManager
    MPLSOCCER_AVAILABLE = True
except ImportError:
    MPLSOCCER_AVAILABLE = False


def create_sample_data():
    """
    Create sample player data for demonstration.
    """
    np.random.seed(42)
    
    players = []
    positions = ['GK', 'CB', 'LB', 'RB', 'CDM', 'CM', 'CAM', 'LW', 'RW', 'ST']
    leagues = ['Premier League', 'La Liga', 'Serie A', 'Bundesliga', 'Ligue 1']
    
    for i in range(100):
        position = np.random.choice(positions)
        league = np.random.choice(leagues)
        
        # Generate stats based on position
        if position in ['GK']:
            player = {
                'name': f'Player_{i}',
                'position': position,
                'league': league,
                'saves': np.random.randint(50, 150),
                'clean_sheets': np.random.randint(5, 20),
                'passes': np.random.randint(300, 800),
                'pass_accuracy': np.random.uniform(60, 95),
            }
        elif position in ['CB']:
            player = {
                'name': f'Player_{i}',
                'position': position,
                'league': league,
                'tackles': np.random.randint(30, 80),
                'interceptions': np.random.randint(40, 100),
                'clearances': np.random.randint(80, 200),
                'aerial_duels': np.random.randint(50, 150),
                'passes': np.random.randint(800, 2000),
                'pass_accuracy': np.random.uniform(80, 95),
                'progressive_passes': np.random.randint(50, 200),
            }
        elif position in ['LB', 'RB']:
            player = {
                'name': f'Player_{i}',
                'position': position,
                'league': league,
                'tackles': np.random.randint(40, 90),
                'interceptions': np.random.randint(30, 80),
                'progressive_carries': np.random.randint(50, 150),
                'crosses': np.random.randint(20, 100),
                'passes': np.random.randint(600, 1500),
                'pass_accuracy': np.random.uniform(70, 90),
                'progressive_passes': np.random.randint(40, 150),
            }
        elif position in ['CDM', 'CM']:
            player = {
                'name': f'Player_{i}',
                'position': position,
                'league': league,
                'tackles': np.random.randint(40, 100),
                'interceptions': np.random.randint(30, 90),
                'passes': np.random.randint(1000, 2500),
                'pass_accuracy': np.random.uniform(75, 92),
                'progressive_passes': np.random.randint(80, 250),
                'progressive_carries': np.random.randint(40, 120),
                'key_passes': np.random.randint(10, 60),
                'pressing_actions': np.random.randint(100, 300),
            }
        elif position in ['CAM', 'LW', 'RW']:
            player = {
                'name': f'Player_{i}',
                'position': position,
                'league': league,
                'key_passes': np.random.randint(30, 120),
                'assists': np.random.randint(2, 15),
                'shot_creating_actions': np.random.randint(50, 200),
                'dribbles': np.random.randint(50, 200),
                'progressive_carries': np.random.randint(80, 250),
                'shots': np.random.randint(30, 120),
                'goals': np.random.randint(3, 25),
                'passes': np.random.randint(600, 1800),
                'pass_accuracy': np.random.uniform(70, 88),
            }
        else:  # ST
            player = {
                'name': f'Player_{i}',
                'position': position,
                'league': league,
                'shots': np.random.randint(80, 200),
                'goals': np.random.randint(10, 35),
                'xg': np.random.uniform(8, 30),
                'assists': np.random.randint(1, 10),
                'aerial_duels': np.random.randint(50, 200),
                'progressive_carries': np.random.randint(30, 100),
                'passes': np.random.randint(300, 1000),
                'pass_accuracy': np.random.uniform(60, 80),
            }
        
        players.append(player)
    
    return pd.DataFrame(players)


def draw_player_radar(player_data, attributes, ax=None):
    """
    Draw a radar chart for a player using matplotlib.
    Fallback implementation when mplsoccer is not available.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Number of attributes
    num_attrs = len(attributes)
    angles = np.linspace(0, 2 * np.pi, num_attrs, endpoint=False).tolist()
    
    # Get values
    values = [player_data.get(attr, 0) for attr in attributes]
    
    # Close the plot
    angles += angles[:1]
    values += values[:1]
    
    # Plot
    ax.plot(angles, values, 'o-', linewidth=2, label=player_data.get('name', 'Player'))
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    ax.set_ylim(0, max(values) * 1.2 if max(values) > 0 else 1)
    ax.grid(True)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return ax


def draw_mplsoccer_radar(player_data, attributes, player_name):
    """
    Draw a radar chart using mplsoccer library.
    """
    # Get values and normalize to 0-100 scale
    values = [player_data.get(attr, 0) for attr in attributes]
    max_val = max(values) if max(values) > 0 else 1
    values_normalized = [v / max_val * 100 for v in values]
    
    # Create radar
    radar = Radar(attributes, [[0, 100]] * len(attributes))
    fig, ax = radar.setup_axis()
    rings_inner = radar.draw_circles(ax=ax, facecolor='#ffb2b2', edgecolor='#fc5f5f')
    radar_output = radar.draw_radar(values_normalized, ax=ax, kwargs_radar={'facecolor': '#00f2c1', 'alpha': 0.6})
    radar_poly, rings_outer, vertices = radar_output
    range_labels = radar.draw_range_labels(ax=ax, fontsize=10)
    param_labels = radar.draw_param_labels(ax=ax, fontsize=12)
    
    ax.set_title(player_name, fontsize=16, pad=20)
    
    return fig


def draw_formation_433(selected_players):
    """
    Draw a 4-3-3 formation visualization with selected players.
    """
    fig, ax = plt.subplots(figsize=(10, 14))
    
    # Draw pitch
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)
    ax.set_aspect('equal')
    
    # Pitch background
    pitch_color = '#2d5f3f'
    ax.set_facecolor(pitch_color)
    
    # Draw lines
    ax.plot([0, 100], [70, 70], 'white', linewidth=2)  # Halfway line
    ax.plot([50, 50], [0, 140], 'white', linewidth=2)  # Center line
    
    # Center circle
    circle = Circle((50, 70), 10, fill=False, edgecolor='white', linewidth=2)
    ax.add_patch(circle)
    
    # Define 4-3-3 positions (x, y, position_name)
    positions_433 = {
        'GK': (50, 10),
        'LB': (20, 30),
        'CB1': (40, 30),
        'CB2': (60, 30),
        'RB': (80, 30),
        'CDM': (50, 55),
        'CM1': (30, 65),
        'CM2': (70, 65),
        'LW': (20, 100),
        'ST': (50, 110),
        'RW': (80, 100),
    }
    
    # Draw players
    for pos_key, (x, y) in positions_433.items():
        player = selected_players.get(pos_key, {})
        player_name = player.get('name', pos_key)
        archetype = player.get('archetype_name', '')
        
        # Draw circle for player
        color = '#00f2c1' if player else '#cccccc'
        circle = Circle((x, y), 3, color=color, ec='white', linewidth=2, zorder=10)
        ax.add_patch(circle)
        
        # Add text
        ax.text(x, y - 8, player_name, ha='center', va='top', fontsize=9, 
                color='white', weight='bold', zorder=11)
        
        if archetype:
            ax.text(x, y - 12, archetype, ha='center', va='top', fontsize=7, 
                    color='#ffeb3b', style='italic', zorder=11)
    
    ax.axis('off')
    ax.set_title('4-3-3 Formation - Squad Synergy', fontsize=16, color='white', pad=20)
    
    return fig


def calculate_synergy_score(selected_players):
    """
    Calculate synergy score for the 4-3-3 formation.
    """
    if len(selected_players) < 11:
        return 0.0, "Incomplete formation"
    
    # Simple synergy calculation based on archetype compatibility
    synergy_matrix = {
        'Progressive Pivot': ['Creative Playmaker', 'Box-to-Box Warrior', 'Dynamic Dribbler'],
        'High-Press Predator': ['Box-to-Box Warrior', 'Defensive Anchor', 'Progressive Pivot'],
        'Creative Playmaker': ['Goal-Scoring Threat', 'Dynamic Dribbler', 'Progressive Pivot'],
        'Box-to-Box Warrior': ['Progressive Pivot', 'Creative Playmaker', 'High-Press Predator'],
        'Deep-Lying Controller': ['Progressive Pivot', 'Box-to-Box Warrior', 'Creative Playmaker'],
        'Goal-Scoring Threat': ['Creative Playmaker', 'Dynamic Dribbler'],
        'Defensive Anchor': ['High-Press Predator', 'Progressive Pivot'],
        'Dynamic Dribbler': ['Creative Playmaker', 'Goal-Scoring Threat'],
    }
    
    total_synergy = 0
    max_synergy = 0
    
    # Check adjacent position synergies
    adjacencies = [
        ('CB1', 'CB2'), ('LB', 'CB1'), ('RB', 'CB2'),
        ('CDM', 'CB1'), ('CDM', 'CB2'),
        ('CM1', 'CDM'), ('CM2', 'CDM'),
        ('CM1', 'LW'), ('CM2', 'RW'),
        ('ST', 'LW'), ('ST', 'RW'),
    ]
    
    for pos1, pos2 in adjacencies:
        player1 = selected_players.get(pos1, {})
        player2 = selected_players.get(pos2, {})
        
        arch1 = player1.get('archetype_name', '')
        arch2 = player2.get('archetype_name', '')
        
        max_synergy += 1
        
        if arch1 and arch2:
            compatible = synergy_matrix.get(arch1, [])
            if arch2 in compatible:
                total_synergy += 1
    
    synergy_score = (total_synergy / max_synergy * 100) if max_synergy > 0 else 0
    
    if synergy_score >= 70:
        feedback = "Excellent synergy! This formation has great compatibility."
    elif synergy_score >= 50:
        feedback = "Good synergy. Some improvements possible."
    elif synergy_score >= 30:
        feedback = "Moderate synergy. Consider adjusting player roles."
    else:
        feedback = "Low synergy. Significant tactical adjustments needed."
    
    return synergy_score, feedback


def main():
    st.set_page_config(page_title="P.A.R.A.D.I.S.E.", layout="wide", page_icon="⚽")
    
    st.title("⚽ P.A.R.A.D.I.S.E.")
    st.markdown("**Player Analysis, Recruitment, Archetyping & Data-driven Identification for Squad Efficiency**")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Load or generate data
    if 'player_data' not in st.session_state:
        st.session_state.player_data = create_sample_data()
        st.session_state.normalized_data = None
        st.session_state.archetyped_data = None
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Player Analysis", "🎯 Archetyping", "⚽ Formation Synergy"])
    
    with tab1:
        st.header("Player Statistics & Normalization")
        
        # Show raw data
        st.subheader("Raw Player Data")
        st.dataframe(st.session_state.player_data.head(20))
        
        # Normalization
        st.subheader("Z-Score Normalization")
        
        if st.button("Apply Normalization"):
            normalizer = ZScoreNormalizer()
            
            # Get numeric columns
            numeric_cols = st.session_state.player_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                st.session_state.normalized_data = normalizer.normalize(
                    st.session_state.player_data,
                    numeric_cols,
                    league_column='league'
                )
                st.success("Normalization applied successfully!")
        
        if st.session_state.normalized_data is not None:
            st.subheader("Normalized Data")
            norm_cols = [col for col in st.session_state.normalized_data.columns if col.endswith('_norm')]
            display_cols = ['name', 'position', 'league'] + norm_cols
            st.dataframe(st.session_state.normalized_data[display_cols].head(20))
    
    with tab2:
        st.header("Player Archetyping with K-Means")
        
        n_clusters = st.slider("Number of Archetypes", min_value=4, max_value=12, value=8)
        
        if st.button("Identify Archetypes"):
            # Get feature columns
            numeric_cols = st.session_state.player_data.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 3:
                archetyper = PlayerArchetyper(n_clusters=n_clusters)
                st.session_state.archetyped_data = archetyper.fit_predict(
                    st.session_state.player_data,
                    numeric_cols
                )
                st.success(f"Identified {n_clusters} player archetypes!")
                
                # Show archetype distribution
                st.subheader("Archetype Distribution")
                archetype_counts = st.session_state.archetyped_data['archetype_name'].value_counts()
                st.bar_chart(archetype_counts)
                
                # Show archetype summary
                st.subheader("Archetype Summary")
                summary = archetyper.get_archetype_summary(st.session_state.archetyped_data, numeric_cols)
                st.dataframe(summary)
            else:
                st.error("Not enough numeric features for clustering")
        
        # Player radar
        if st.session_state.archetyped_data is not None:
            st.subheader("Player Radar Chart")
            
            player_names = st.session_state.archetyped_data['name'].tolist()
            selected_player = st.selectbox("Select Player", player_names)
            
            if selected_player:
                player_row = st.session_state.archetyped_data[
                    st.session_state.archetyped_data['name'] == selected_player
                ].iloc[0]
                
                st.write(f"**Position:** {player_row['position']}")
                st.write(f"**League:** {player_row['league']}")
                st.write(f"**Archetype:** {player_row['archetype_name']}")
                
                # Get numeric attributes
                numeric_attrs = st.session_state.player_data.select_dtypes(include=[np.number]).columns.tolist()
                
                if numeric_attrs:
                    # Choose relevant attributes based on position
                    if len(numeric_attrs) > 8:
                        numeric_attrs = numeric_attrs[:8]
                    
                    player_dict = player_row.to_dict()
                    
                    if MPLSOCCER_AVAILABLE:
                        fig = draw_mplsoccer_radar(player_dict, numeric_attrs, selected_player)
                    else:
                        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
                        draw_player_radar(player_dict, numeric_attrs, ax)
                    
                    st.pyplot(fig)
    
    with tab3:
        st.header("4-3-3 Formation Synergy Checker")
        
        if st.session_state.archetyped_data is not None:
            st.subheader("Select Players for Each Position")
            
            positions_433 = ['GK', 'LB', 'CB1', 'CB2', 'RB', 'CDM', 'CM1', 'CM2', 'LW', 'ST', 'RW']
            
            selected_squad = {}
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**Defense**")
                for pos in ['GK', 'LB', 'CB1', 'CB2', 'RB']:
                    players = st.session_state.archetyped_data['name'].tolist()
                    selected = st.selectbox(f"{pos}", [''] + players, key=pos)
                    if selected:
                        player_row = st.session_state.archetyped_data[
                            st.session_state.archetyped_data['name'] == selected
                        ].iloc[0]
                        selected_squad[pos] = player_row.to_dict()
            
            with col2:
                st.markdown("**Midfield**")
                for pos in ['CDM', 'CM1', 'CM2']:
                    players = st.session_state.archetyped_data['name'].tolist()
                    selected = st.selectbox(f"{pos}", [''] + players, key=pos)
                    if selected:
                        player_row = st.session_state.archetyped_data[
                            st.session_state.archetyped_data['name'] == selected
                        ].iloc[0]
                        selected_squad[pos] = player_row.to_dict()
            
            with col3:
                st.markdown("**Attack**")
                for pos in ['LW', 'ST', 'RW']:
                    players = st.session_state.archetyped_data['name'].tolist()
                    selected = st.selectbox(f"{pos}", [''] + players, key=pos)
                    if selected:
                        player_row = st.session_state.archetyped_data[
                            st.session_state.archetyped_data['name'] == selected
                        ].iloc[0]
                        selected_squad[pos] = player_row.to_dict()
            
            # Visualize formation
            st.subheader("Formation Visualization")
            fig = draw_formation_433(selected_squad)
            st.pyplot(fig)
            
            # Calculate synergy
            if len(selected_squad) == 11:
                synergy_score, feedback = calculate_synergy_score(selected_squad)
                
                st.subheader("Synergy Analysis")
                st.metric("Synergy Score", f"{synergy_score:.1f}%")
                st.info(feedback)
            else:
                st.warning(f"Please select all 11 players. Currently selected: {len(selected_squad)}/11")
        else:
            st.warning("Please run archetyping first in the 'Archetyping' tab.")


if __name__ == "__main__":
    main()
