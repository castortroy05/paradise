# paradise
P.A.R.A.D.I.S.E. (Player Analysis, Recruitment, Archetyping &amp; Data-driven Identification for Squad Efficiency) is an advanced football analytics utility. It leverages event data and league-normalization algorithms to identify undervalued talent and optimize lineup synergy. 


# P.A.R.A.D.I.S.E. 🍀
**Player Analysis, Recruitment, Archetyping & Data-driven Identification for Squad Efficiency**

![Version](https://img.shields.io/badge/version-1.0.0-green)
![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

## 📋 Overview
P.A.R.A.D.I.S.E. is a high-performance data analysis utility inspired by the "Tony Bloom" model of football recruitment. Unlike traditional scouting platforms, P.A.R.A.D.I.S.E. focuses on **process over outcome**, using league-weighted Z-Scores and tactical archetyping to identify undervalued players and optimize squad lineups.

The suite is designed to move beyond raw stats (goals/assists) to find the "hidden signals" that predict long-term success and tactical fit.

---

## 🚀 Key Features

### 1. Market Arbitrage & Normalization
* **League Coefficients:** Automatically adjusts player stats based on the strength of the league (e.g., Eredivisie vs. Premier League) using UEFA and historical transition data.
* **Z-Score Benchmarking:** Measures players against league averages to identify elite outliers.

### 2. Tactical Archetyping
Instead of generic positions, the engine categorizes players into functional roles:
* **The Pivot:** High-volume, press-resistant deep-lying midfielders.
* **The Inverted Creative:** Wingers who prioritize chance creation in half-spaces.
* **The High-Press Predator:** Strikers weighted by defensive regains and npxG.

### 3. Squad & Lineup Optimizer
* **Synergy Scoring:** Evaluates how player archetypes interact (e.g., ensures an Inverted Winger is paired with an Overlapping Fullback).
* **Opponent-Adaptive Formations:** Analyzes opponent weaknesses (e.g., low PPDA, aerial vulnerability) and suggests the optimal starting XI and formation.

---

## 🛠️ Technical Stack
* **Engine:** Python (Pandas, NumPy, Scikit-learn)
* **Visualization:** Mplsoccer, Plotly, Streamlit
* **Database:** PostgreSQL (for historical performance tracking)
* **Algorithms:** K-Nearest Neighbors (Player Similarity), Integer Programming (Lineup Optimization)

---

## 📂 Project Structure
```text
├── data/               # Raw and processed event data
├── modules/
│   ├── ingestion.py    # Data fetching and cleaning
│   ├── normalization.py # League weighting & Z-Score logic
│   ├── archetyper.py   # Clustering and role assignment
│   └── optimizer.py    # Lineup & Formation synergy logic
├── notebooks/          # Exploratory Data Analysis (EDA)
├── app/                # Streamlit dashboard source code
└── README.md
