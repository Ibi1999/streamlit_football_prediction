from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import to_hex
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from scipy.stats import zscore
from collections import defaultdict
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
import math



def train_and_predict_league_table_tensorFlow(features, n_runs=10, epochs=120, batch_size=128, learning_rate=0.001, verbose=False):
    all_tables = []
    all_match_predictions = []

    for run in range(n_runs):
        print(f"Run {run+1}/{n_runs}")

        # Prepare training and full datasets
        features_train = features.dropna(subset=['home_goals', 'away_goals']).copy()
        features_all = features.copy()

        feature_cols = [
            'match_week', 'is_weekend_match', 'match_month', 'match_dayofweek',
            'home_avg_goals_scored_last_5', 'home_avg_goals_conceded_last_5',
            'away_avg_goals_scored_last_5', 'away_avg_goals_conceded_last_5',
            'home_avg_xg_scored_last_5', 'home_avg_xg_conceded_last_5',
            'away_avg_xg_scored_last_5', 'away_avg_xg_conceded_last_5',
        ]

        # Encode teams for training
        home_enc = pd.get_dummies(features_train['home'], prefix='home')
        away_enc = pd.get_dummies(features_train['away'], prefix='away')
        X_train = pd.concat([features_train[feature_cols].fillna(0), home_enc, away_enc], axis=1)
        y_train = features_train[['home_goals', 'away_goals']].astype(float)

        # Build model
        model = models.Sequential([
            layers.Dense(90, activation='relu', input_shape=(X_train.shape[1],)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(48, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(30, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(2)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=int(verbose))

        if run == 0:
            plt.plot(history.history['loss'], label='train loss')
            plt.plot(history.history['val_loss'], label='val loss')
            plt.legend()
            plt.title("Loss Over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()

        # Prepare full feature set for prediction
        home_enc_all = pd.get_dummies(features_all['home'], prefix='home')
        away_enc_all = pd.get_dummies(features_all['away'], prefix='away')
        X_all = pd.concat([features_all[feature_cols].fillna(0), home_enc_all, away_enc_all], axis=1)

        # Make sure columns match training set
        for col in X_train.columns:
            if col not in X_all.columns:
                X_all[col] = 0
        X_all = X_all[X_train.columns]

        predicted_scores = model.predict(X_all, verbose=0)
        features_all['pred_home_goals'] = predicted_scores[:, 0]
        features_all['pred_away_goals'] = predicted_scores[:, 1]

        # Assign points based on predicted scores
        points = features_all.apply(assign_points, axis=1)
        features_all = pd.concat([features_all, points], axis=1)

        match_stats = features_all[['home', 'away', 'pred_home_goals', 'pred_away_goals', 'home_points', 'away_points']].copy()
        all_match_predictions.append(match_stats)

        # Aggregate home stats
        home_stats = features_all.groupby('home').agg(
            points_sum=('home_points', 'sum'),
            wins_home=('home_points', lambda x: (x == 3).sum()),
            draws_home=('home_points', lambda x: (x == 1).sum()),
            losses_home=('home_points', lambda x: (x == 0).sum()),
            goals_for=('pred_home_goals', 'sum'),
            goals_against=('pred_away_goals', 'sum'),
        )

        # Aggregate away stats
        away_stats = features_all.groupby('away').agg(
            points_sum=('away_points', 'sum'),
            wins_away=('away_points', lambda x: (x == 3).sum()),
            draws_away=('away_points', lambda x: (x == 1).sum()),
            losses_away=('away_points', lambda x: (x == 0).sum()),
            goals_for=('pred_away_goals', 'sum'),
            goals_against=('pred_home_goals', 'sum'),
        )

        # Combine home and away stats
        league_table = pd.DataFrame()
        league_table['points'] = home_stats['points_sum'] + away_stats['points_sum']
        league_table['wins'] = home_stats['wins_home'] + away_stats['wins_away']
        league_table['draws'] = home_stats['draws_home'] + away_stats['draws_away']
        league_table['losses'] = home_stats['losses_home'] + away_stats['losses_away']
        league_table['goals_for'] = home_stats['goals_for'] + away_stats['goals_for']
        league_table['goals_against'] = home_stats['goals_against'] + away_stats['goals_against']

        league_table['goal_difference'] = league_table['goals_for'] - league_table['goals_against']

        # Count games played
        games_played = features_all['home'].value_counts().add(features_all['away'].value_counts(), fill_value=0)
        league_table['played'] = games_played

        league_table.reset_index(inplace=True)
        league_table.rename(columns={'home': 'team'}, inplace=True)

        all_tables.append(league_table)

    # Average the results from all runs
    final_table = pd.concat(all_tables).groupby('team').mean()

    # Round and convert to int
    final_table = final_table.round(0).astype(int)

    # Reorder columns exactly as requested
    final_table = final_table[['played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against', 'goal_difference', 'points']]

    # Sort by points desc then goal difference desc
    final_table = final_table.sort_values(by=['points', 'goal_difference'], ascending=False).reset_index()

    match_predictions = pd.concat(all_match_predictions).groupby(['home', 'away']).mean().reset_index()

    return final_table, match_predictions

def round_down_to_nearest_half(x):
    return math.floor(x * 2) / 2

def round_up_to_nearest_half(x):
    return math.ceil(x * 2) / 2

def get_match_result(home_goals, away_goals):
    if home_goals > away_goals:
        return "Home Win"
    elif home_goals < away_goals:
        return "Away Win"
    else:
        return "Draw"

def calculate_prediction_confidence(pred_home_goals, pred_away_goals):
    home_lower = round_down_to_nearest_half(pred_home_goals)
    home_upper = round_up_to_nearest_half(pred_home_goals)
    away_lower = round_down_to_nearest_half(pred_away_goals)
    away_upper = round_up_to_nearest_half(pred_away_goals)

    possible_results = set()
    for home_score in [home_lower, home_upper]:
        for away_score in [away_lower, away_upper]:
            possible_results.add(get_match_result(home_score, away_score))

    distinct_count = len(possible_results)
    if distinct_count == 1:
        return "High"
    elif distinct_count == 2:
        return "Medium"
    else:
        return "Low"

def compare_prediction_with_actual(home_team, away_team, features, match_predictions):
    # Find actual match in features
    actual_match = features[
        (features['home'] == home_team) & 
        (features['away'] == away_team)
    ]
    
    if actual_match.empty:
        return f"No match found for {home_team} vs {away_team}."

    actual_match = actual_match.iloc[0]
    
    # Get predicted result using your existing function
    prediction = predict_match_score(home_team, away_team, match_predictions)

    if isinstance(prediction, str):
        return prediction  # Error message from predict_match_score
    
    # Check if actual goals are available (match played)
    if pd.isna(actual_match['home_goals']) or pd.isna(actual_match['away_goals']):
        # Match not played yet
        confidence = calculate_prediction_confidence(
            prediction["pred_home_goals"], prediction["pred_away_goals"]
        )
        return {
            "home_team": home_team,
            "away_team": away_team,
            "actual_home_goals": None,
            "actual_away_goals": None,
            "actual_result": None,
            "pred_home_goals": prediction["pred_home_goals"],
            "pred_away_goals": prediction["pred_away_goals"],
            "pred_home_goals_rounded": round(prediction["pred_home_goals"]),
            "pred_away_goals_rounded": round(prediction["pred_away_goals"]),
            "predicted_result": prediction["predicted_result"],
            "prediction_confidence": confidence
        }

    actual_home_goals = int(actual_match['home_goals'])
    actual_away_goals = int(actual_match['away_goals'])

    if actual_home_goals > actual_away_goals:
        actual_result = "Home Win"
    elif actual_home_goals < actual_away_goals:
        actual_result = "Away Win"
    else:
        actual_result = "Draw"

    confidence = calculate_prediction_confidence(
        prediction["pred_home_goals"], prediction["pred_away_goals"]
    )

    # Return combined actual and predicted result with confidence and rounded scores
    return {
        "home_team": home_team,
        "away_team": away_team,
        "actual_home_goals": actual_home_goals,
        "actual_away_goals": actual_away_goals,
        "actual_result": actual_result,
        "pred_home_goals": prediction["pred_home_goals"],
        "pred_away_goals": prediction["pred_away_goals"],
        "pred_home_goals_rounded": round(prediction["pred_home_goals"]),
        "pred_away_goals_rounded": round(prediction["pred_away_goals"]),
        "predicted_result": prediction["predicted_result"],
        "prediction_confidence": confidence
    }

def predict_match_score(home_team, away_team, match_predictions):
    match = match_predictions[
        (match_predictions['home'] == home_team) &
        (match_predictions['away'] == away_team)
    ]

    if match.empty:
        return f"No prediction found for {home_team} vs {away_team}."
    
    match = match.iloc[0]
    
    h_goals = round(match['pred_home_goals'])
    a_goals = round(match['pred_away_goals'])
    
    if h_goals > a_goals:
        result = "Home Win"
    elif h_goals < a_goals:
        result = "Away Win"
    else:
        result = "Draw"
    
    return {
        "home_team": home_team,
        "away_team": away_team,
        "pred_home_goals": round(match['pred_home_goals'], 2),
        "pred_away_goals": round(match['pred_away_goals'], 2),
        "predicted_result": result
    }


def evaluate_model_accuracy(features, match_predictions):
    # Merge actual and predicted results
    merged = pd.merge(
        features[['home', 'away', 'home_goals', 'away_goals']],
        match_predictions[['home', 'away', 'pred_home_goals', 'pred_away_goals']],
        on=['home', 'away'],
        how='inner'
    )

    # ❗ Exclude matches with no actual result
    merged = merged.dropna(subset=['home_goals', 'away_goals'])

    # Round predicted goals to compare with actual goals
    merged['pred_home_goals_rounded'] = merged['pred_home_goals'].round().astype(int)
    merged['pred_away_goals_rounded'] = merged['pred_away_goals'].round().astype(int)

    # Compute goal accuracy
    home_goal_accuracy = (merged['pred_home_goals_rounded'] == merged['home_goals']).mean()
    away_goal_accuracy = (merged['pred_away_goals_rounded'] == merged['away_goals']).mean()

    # Actual and predicted results
    def get_result(hg, ag):
        if hg > ag:
            return "Home Win"
        elif hg < ag:
            return "Away Win"
        else:
            return "Draw"

    merged['actual_result'] = merged.apply(lambda row: get_result(row['home_goals'], row['away_goals']), axis=1)
    merged['predicted_result'] = merged.apply(lambda row: get_result(row['pred_home_goals_rounded'], row['pred_away_goals_rounded']), axis=1)

    result_accuracy = (merged['actual_result'] == merged['predicted_result']).mean()

    return {
        "home_goals_accuracy": round(home_goal_accuracy * 100, 2),
        "away_goals_accuracy": round(away_goal_accuracy * 100, 2),
        "result_accuracy": round(result_accuracy * 100, 2)
    }


def get_prediction_distribution(match_predictions):
    def classify_result(row):
        h = round(row['pred_home_goals'])
        a = round(row['pred_away_goals'])
        if h > a:
            return "Home Win"
        elif h < a:
            return "Away Win"
        else:
            return "Draw"

    match_predictions = match_predictions.copy()
    match_predictions['predicted_result'] = match_predictions.apply(classify_result, axis=1)

    distribution = match_predictions['predicted_result'].value_counts(normalize=True) * 100
    return distribution.round(2).to_dict()


def assign_points(row):
    home_goals = row['pred_home_goals']
    away_goals = row['pred_away_goals']
    # Round predicted goals to nearest integer for points calculation
    home_goals = round(home_goals)
    away_goals = round(away_goals)

    if home_goals > away_goals:
        return pd.Series({'home_points': 3, 'away_points': 0})
    elif home_goals < away_goals:
        return pd.Series({'home_points': 0, 'away_points': 3})
    else:
        return pd.Series({'home_points': 1, 'away_points': 1})


def compute_team_rolling_stats(df, team_col, goals_for_col, goals_against_col, prefix):
    
    team_stats = []

    for team in df[team_col].unique():
        team_matches = df[(df[team_col] == team)].copy()
        team_matches = team_matches.sort_values('date')

        team_matches[f'{prefix}_avg_goals_scored_last_5'] = team_matches[goals_for_col].shift().rolling(5, min_periods=1).mean()
        team_matches[f'{prefix}_avg_goals_conceded_last_5'] = team_matches[goals_against_col].shift().rolling(5, min_periods=1).mean()

        team_stats.append(team_matches[[f'{prefix}_avg_goals_scored_last_5', f'{prefix}_avg_goals_conceded_last_5']])

    return pd.concat(team_stats).sort_index()


def result(row):
    if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
        return None
    if row['home_goals'] > row['away_goals']:
        return 1
    elif row['home_goals'] < row['away_goals']:
        return -1
    else:
        return 0
    

def create_actual_league_table(features_df):
    # Create home stats
    home_df = features_df[['home', 'home_goals', 'away_goals']].copy()
    home_df.rename(columns={
        'home': 'team',
        'home_goals': 'goals_for',
        'away_goals': 'goals_against'
    }, inplace=True)
    home_df['win'] = home_df['goals_for'] > home_df['goals_against']
    home_df['draw'] = home_df['goals_for'] == home_df['goals_against']
    home_df['loss'] = home_df['goals_for'] < home_df['goals_against']

    # Create away stats
    away_df = features_df[['away', 'away_goals', 'home_goals']].copy()
    away_df.rename(columns={
        'away': 'team',
        'away_goals': 'goals_for',
        'home_goals': 'goals_against'
    }, inplace=True)
    away_df['win'] = away_df['goals_for'] > away_df['goals_against']
    away_df['draw'] = away_df['goals_for'] == away_df['goals_against']
    away_df['loss'] = away_df['goals_for'] < away_df['goals_against']

    # Combine home and away stats
    full_df = pd.concat([home_df, away_df], ignore_index=True)

    # Aggregate per team
    table = full_df.groupby('team').agg(
        played=('goals_for', 'count'),
        wins=('win', 'sum'),
        draws=('draw', 'sum'),
        losses=('loss', 'sum'),
        goals_for=('goals_for', 'sum'),
        goals_against=('goals_against', 'sum')
    ).reset_index()

    table['goal_difference'] = table['goals_for'] - table['goals_against']
    table['points'] = table['wins'] * 3 + table['draws']

    # Convert relevant columns to int (no decimals)
    cols_to_int = ['played', 'wins', 'draws', 'losses', 'goals_for', 'goals_against', 'goal_difference', 'points']
    table[cols_to_int] = table[cols_to_int].round(0).astype(int)

    # Sort and reset index
    table = table.sort_values(
        by=['points', 'goal_difference', 'goals_for'],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    return table


