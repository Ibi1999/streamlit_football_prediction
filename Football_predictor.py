import streamlit as st
import pandas as pd
from predictor_func import *
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
features = pd.read_pickle('features.pkl')

# --- Sidebar ---
with st.sidebar:
    st.markdown("<small>Created by Ibrahim Oksuzoglu</small>", unsafe_allow_html=True)
    st.title("🤖 Machine learning")
    model_type = st.selectbox("🧠 Select prediction Model type:", ("TensorFlow", "GLM - Poisson"))

    with st.expander("❓ What this app does", expanded=False): 
        st.markdown("""
        ### What this app does:
        This app uses various Machine Learning models to predict football scores and outcomes. 
        You can also look at the predicted outcomes of specific games.
        """)

# --- Main Content ---
if model_type in ("TensorFlow", "GLM - Poisson"):
    emoji_map = {
        "TensorFlow": "🚀",      # Orange diamond
        "GLM - Poisson": "📐"    # Bar chart (or choose another suitable emoji)
    }
    st.title(f"{emoji_map[model_type]} {model_type} Regression Model")


    iteration_count = st.selectbox("🔁 Select Number of Simulations", ("10", "50", "100", "500"))

    @st.cache_data
    def load_final_table(model_type, iteration_count):
        file_map = {
            "TensorFlow": {
                "10": "final_table_10.pkl",
                "50": "final_table_50.pkl",
                "100": "final_table_100.pkl",
                "500": "final_table_500.pkl"
            },
            "GLM - Poisson": {
                "10": "final_table_10_ps.pkl",
                "50": "final_table_50_ps.pkl",
                "100": "final_table_100_ps.pkl",
                "500": "final_table_500_ps.pkl"
            }
        }
        return pd.read_pickle(file_map[model_type][iteration_count])

    @st.cache_data
    def load_predicted_results(model_type, iteration_count):
        file_map = {
            "TensorFlow": {
                "10": "match_predictions_10.pkl",
                "50": "match_predictions_50.pkl",
                "100": "match_predictions_100.pkl",
                "500": "match_predictions_500.pkl"
            },
            "GLM - Poisson": {
                "10": "match_predictions_10_ps.pkl",
                "50": "match_predictions_50_ps.pkl",
                "100": "match_predictions_100_ps.pkl",
                "500": "match_predictions_500_ps.pkl"
            }
        }
        return pd.read_pickle(file_map[model_type][iteration_count])

    final_table_df = load_final_table(model_type, iteration_count)
    match_predictions = load_predicted_results(model_type, iteration_count)
    accuracy_scores = evaluate_model_accuracy(features, match_predictions)
    actual_table_df = create_actual_league_table(features)

    # --- Show Accuracy Metrics Just After Simulation Selection ---
    st.markdown("### 📈 Prediction Accuracy Metrics")
    col_acc1, col_acc2, col_acc3 = st.columns(3)
    with col_acc1:
        st.metric("🎯 Match Result Accuracy", f"{accuracy_scores['result_accuracy']}%")
    with col_acc2:
        st.metric("🏠 Home Goals Accuracy", f"{accuracy_scores['home_goals_accuracy']}%")
    with col_acc3:
        st.metric("🚪 Away Goals Accuracy", f"{accuracy_scores['away_goals_accuracy']}%")

    # --- Match Result Comparison ---
    st.markdown("---")
    st.subheader("🔍 Compare Prediction with Actual Match Result")

    team_list = sorted(features['home'].unique().tolist())

    col1, col2 = st.columns(2)
    with col1:
        selected_home_team = st.selectbox("Select Home Team", team_list)
    with col2:
        selected_away_team = st.selectbox("Select Away Team", team_list)

    if selected_home_team and selected_away_team:
        if selected_home_team == selected_away_team:
            st.warning("Home and away teams must be different.")
        else:
            comparison_result = compare_prediction_with_actual(
                selected_home_team, selected_away_team, features, match_predictions
            )

            if isinstance(comparison_result, str):
                st.error(comparison_result)
            else:
                st.success(f"Showing predicted vs actual match outcome of {selected_home_team} vs {selected_away_team}")

                col_pred, col_actual = st.columns(2)

                with col_pred:
                    st.markdown("#### 📊 Predicted Result")
                    st.metric(
                        label="Score",
                        value=(f"{comparison_result['pred_home_goals_rounded']} - {comparison_result['pred_away_goals_rounded']} "
                               f"({comparison_result['pred_home_goals']:.2f} - {comparison_result['pred_away_goals']:.2f})")
                    )
                    st.metric(label="Outcome", value=comparison_result['predicted_result'])
                    st.metric(label="Prediction Confidence", value=comparison_result['prediction_confidence'])

                with col_actual:
                    st.markdown("#### ✅ Actual Result")

                    if comparison_result['actual_home_goals'] is None or comparison_result['actual_away_goals'] is None:
                        st.info("Game has not been played yet")
                    else:
                        st.metric(label="Score", value=f"{comparison_result['actual_home_goals']} - {comparison_result['actual_away_goals']}")
                        st.metric(label="Outcome", value=comparison_result['actual_result'])

    st.markdown("---")
    # --- Layout: Three Columns for Tables ---
    col1, col2 = st.columns([1.5, 1.5])

    with col1:
        st.subheader("📊 Predicted League Table")

        # Convert float columns to integers for cleaner display
        int_cols = final_table_df.select_dtypes(include='number').columns
        final_table_display = final_table_df.copy()
        final_table_display[int_cols] = final_table_display[int_cols].round(0).astype(int)

        table_height = len(final_table_display) * 37
        st.dataframe(
            final_table_display.style.hide(axis="index"),
            use_container_width=True,
            height=table_height
        )

    with col2:
        st.subheader("✅ Actual Premier League Table")
        actual_table_height = len(actual_table_df) * 37
        st.dataframe(
            actual_table_df.style.hide(axis="index"),
            use_container_width=True,
            height=actual_table_height
        )
