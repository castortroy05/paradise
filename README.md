# ⚽ P.A.R.A.D.I.S.E.

**Player Analysis, Recruitment, Archetyping & Data-driven Identification for Squad Efficiency**

An advanced football analytics utility that leverages event data and league-normalization algorithms to identify undervalued talent and optimize lineup synergy.

## Features

- **Z-Score Normalization**: Normalize player statistics across leagues using league coefficients
- **Player Archetyping**: K-Means clustering to classify players into roles like 'Progressive Pivot', 'High-Press Predator', etc.
- **Interactive Dashboard**: Streamlit-based UI with player radars and formation synergy checker
- **4-3-3 Formation Analysis**: Visual formation builder with tactical synergy scoring

## Tech Stack

- **Python** - Core language
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning (K-Means clustering)
- **Mplsoccer** - Football visualization
- **Streamlit** - Interactive dashboard

## Installation

```bash
pip install -r requirements.txt
```

## Project Initialization

Run the initialization script to set up a virtual environment and install dependencies:

```bash
./scripts/init.sh
```

## Usage

### Run the Streamlit Dashboard

```bash
streamlit run app.py
```

### Use Modules Programmatically

#### Normalization

```python
from modules.normalization import ZScoreNormalizer
import pandas as pd

# Create normalizer
normalizer = ZScoreNormalizer()

# Normalize player data
data = pd.DataFrame({
    'name': ['Player1', 'Player2'],
    'league': ['Premier League', 'La Liga'],
    'goals': [20, 25],
    'assists': [10, 8]
})

normalized = normalizer.normalize(data, ['goals', 'assists'], 'league')
print(normalized[['name', 'goals_norm', 'assists_norm']])
```

#### Archetyping

```python
from modules.archetyper import PlayerArchetyper
import pandas as pd

# Create archetyper
archetyper = PlayerArchetyper(n_clusters=8)

# Classify players
data = pd.DataFrame({
    'name': ['Player1', 'Player2'],
    'tackles': [80, 40],
    'passes': [1500, 2000],
    'shots': [50, 100]
})

result = archetyper.fit_predict(data, ['tackles', 'passes', 'shots'])
print(result[['name', 'archetype_name']])
```

## Project Structure

```
paradise/
├── app.py                    # Streamlit dashboard
├── modules/
│   ├── __init__.py
│   ├── normalization.py      # Z-score normalization
│   └── archetyper.py         # K-Means clustering
├── data/                     # Player data files
├── requirements.txt          # Python dependencies
└── test_modules.py           # Test suite
```

## Testing

Run the test suite:

```bash
python test_modules.py
```

## Key Concepts

### Z-Score Normalization with League Coefficients

Formula: `(x - mean) / std * league_coefficient`

Accounts for different league strengths when comparing players:
- Premier League: 1.0
- La Liga: 0.95
- Serie A: 0.93
- Bundesliga: 0.92
- Ligue 1: 0.88

### Player Archetypes

The K-Means clustering identifies player roles such as:
- **Progressive Pivot**: High passing volume with progressive passes
- **High-Press Predator**: Aggressive pressing and tackles
- **Creative Playmaker**: Key passes and shot-creating actions
- **Box-to-Box Warrior**: Balanced attacking and defensive contributions
- **Deep-Lying Controller**: High pass accuracy and long balls
- **Goal-Scoring Threat**: High shots and goals
- **Defensive Anchor**: Tackles, interceptions, and clearances
- **Dynamic Dribbler**: Dribbles and progressive carries

### Formation Synergy

The 4-3-3 formation checker analyzes tactical compatibility between players based on their archetypes and positions, providing a synergy score and recommendations.
