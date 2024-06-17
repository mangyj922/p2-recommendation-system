import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
import pandas as pd


st.set_page_config(layout="wide",page_title="Recommendation System")

with st.sidebar:
    selected = option_menu(
        menu_title = "Model",
        options = ["Wide & Deep Learning Model", "K-Nearest Neighbors (KNN)", "Matrix Factorization"],
        icons = ["caret-right-square-fill", "caret-right-square-fill", "caret-right-square-fill"],
        menu_icon = "app",
        default_index = 0,
    )

def evaluation(evaluation_df, user_id):
    model_result = evaluation_df[evaluation_df.user_id == user_id].reset_index(drop=True)

    mae = model_result.iloc[0]['mae']
    mse = model_result.iloc[0]['mse']
    rmse = model_result.iloc[0]['rmse']

    st.write("**Mean Absolute Error (MAE)**: {mae:.4f}, **Mean Squared Error (MSE)**: {mse:.4f}, **Root Mean Squared Error (RMSE)**: {rmse:.4f}"
                .format(mae=mae, mse=mse, rmse=rmse))

def recommendation_page(recommendation_df, evaluation_df, selected):
    precision_recall_k = evaluation_df[evaluation_df.top_k == 30].reset_index(drop=True)

    precision_at_k = precision_recall_k.iloc[0]['precision']
    recall_at_k = precision_recall_k.iloc[0]['recall']
    
    st.write('**precision_at_k:** {precision_at_k:.4f}, **recall_at_k:** {recall_at_k:.4f}'.format(precision_at_k=precision_at_k, recall_at_k=recall_at_k))

    st.subheader("**User**")
    user_df = recommendation_df.drop_duplicates('user_id')[['user_id']].reset_index(drop=True)
    user_option = user_df['user_id'].to_list()

    user_selected = st.selectbox("Please select an user", options = user_option)

    if selected == "Wide & Deep Learning Model":
        wide_deep_metric = pd.read_parquet('./data/evaluation/wide_deep_metric.parquet')
        evaluation(wide_deep_metric, user_selected)

    elif selected == "Matrix Factorization":
        matrix_fact_metric = pd.read_parquet('./data/evaluation/matrix_fact_metric.parquet')
        evaluation(matrix_fact_metric, user_selected)

    i = 0
    
    recommend_item = recommendation_df[recommendation_df.user_id == user_selected]
    # st.dataframe(recommend_item)

    row1 = st.columns(5)
    row2 = st.columns(5)
    row3 = st.columns(5)
    row4 = st.columns(5)
    row5 = st.columns(5)
    row6 = st.columns(5)

    for col in row1 + row2 + row3 + row4 + row5 + row6:
        tile = col.container(border=True,height=500)

        tile.image(
            recommend_item.iloc[i]['images.large'][0], width=150,
        )

        tile.write(f"**{recommend_item.iloc[i]['title']}**")

        i+=1

if selected == "Wide & Deep Learning Model":
    st.title(f"{selected}")

    # evaluation metrics
    wide_deep_metric = pd.read_parquet('./data/evaluation/wide_deep_metric.parquet')
    df_evaluation_wide_deep_avg = pd.read_parquet('./data/evaluation/df_evaluation_wide_deep_avg.parquet')

    recommendation_df = pd.read_parquet('./data/recommendation/wide_deep_recommendation.parquet')
    
    st.subheader("**Model Evaluation**")
    evaluation(wide_deep_metric, "model")
    recommendation_page(recommendation_df, df_evaluation_wide_deep_avg, selected)


if selected == "K-Nearest Neighbors (KNN)":
    st.title(f"{selected}")

    # evaluation metrics
    # wide_deep_metric = pd.read_parquet('./data/evaluation/wide_deep_metric.parquet')
    df_evaluation_knn_avg = pd.read_parquet('./data/evaluation/df_evaluation_knn_avg.parquet')

    recommendation_df = pd.read_parquet('./data/recommendation/knn_recommendation.parquet')
    
    st.subheader("**Model Evaluation**")
    recommendation_page(recommendation_df, df_evaluation_knn_avg, selected)

if selected == "Matrix Factorization":
    st.title(f"{selected}")

    # evaluation metrics
    matrix_fact_metric = pd.read_parquet('./data/evaluation/matrix_fact_metric.parquet')
    df_evaluation_matrix_fact_avg = pd.read_parquet('./data/evaluation/df_evaluation_matric_fact_avg.parquet')

    recommendation_df = pd.read_parquet('./data/recommendation/matrix_fact_recommendation.parquet')
    
    st.subheader("**Model Evaluation**")
    evaluation(matrix_fact_metric, "model")
    recommendation_page(recommendation_df, df_evaluation_matrix_fact_avg, selected)

