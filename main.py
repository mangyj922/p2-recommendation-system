import streamlit as st
from streamlit_option_menu import option_menu

import numpy as np
import pandas as pd

st.set_page_config(layout="wide",page_title="Recommendation System")

user_evaluation_list = pd.read_parquet('./data/user_evaluation/user_evaluation_list.parquet')
user_evaluation_list = user_evaluation_list.drop(columns=['user_idx']).reset_index()
user_evaluation_list = user_evaluation_list.rename(columns = {'index':'user_idx'})
user_evaluation_list['user_idx'] = user_evaluation_list['user_idx'] + 1
user_evaluation_list['user_idx'] = user_evaluation_list['user_idx'].astype(str).str.zfill(4)

user_transactions = pd.read_parquet('./data/user_evaluation/user_transactions.parquet')
user_last5_txn = user_transactions.sort_values(['user_id', 'timestamp'],ascending=True).groupby(['user_id']).tail(5)

knn_recommend_items = pd.read_parquet('./data/recommendation/knn_recommend_items.parquet')
matrix_recommend_items = pd.read_parquet('./data/recommendation/matrix_recommend_items.parquet')
wide_deep_recommend_items = pd.read_parquet('./data/recommendation/wide_deep_recommend_items.parquet')

store_list = pd.read_parquet('./data/user_evaluation/store_list.parquet')

user = user_evaluation_list[user_evaluation_list['user_id'] != 'model']['user_idx'].dropna().to_list()
category = ['All Categories'] + sorted(store_list['main_category'].drop_duplicates().to_list())
store = ['All Stores'] + sorted(store_list['store'].drop_duplicates().to_list())

if 'user_list' not in st.session_state:   
    st.session_state['user_list'] = user
    
if 'category_list' not in st.session_state:   
    st.session_state['category_list'] = category

if 'store_list' not in st.session_state:   
    st.session_state['store_list'] = store

if 'clicked_store' not in st.session_state:   
    st.session_state['clicked_store'] = False

if 'clicked_category' not in st.session_state:   
    st.session_state['clicked_category'] = False

if 'store_list' not in st.session_state:   
    st.session_state['store_list'] = "All Stores"

if 'selected_user' not in st.session_state: 
    st.session_state['selected_user'] = '0001'

if 'selected_category' not in st.session_state:
    st.session_state['selected_category'] = 'All Categories'

if 'selected_store' not in st.session_state:
    st.session_state['selected_store'] = 'All Stores'

def update_store():
    st.session_state['clicked_category'] = True

    if st.session_state['clicked_store'] == False:
        if st.session_state['selected_category'] != "All Categories":
            temp_store_list = store_list[store_list['main_category'] == st.session_state['selected_category']]['store'].drop_duplicates().to_list()
            st.session_state['store_list'] = ['All Stores'] + sorted(temp_store_list)
        else:
            st.session_state['store_list'] = store
    else:
        pass

def update_category():
    st.session_state['clicked_store'] = True

    if st.session_state['clicked_category'] == False:
        if st.session_state['category_list'] != 'All Stores':
            temp_category_list = store_list[store_list['store'] == st.session_state['selected_store']]['main_category'].drop_duplicates().to_list()
            st.session_state['category_list'] = ['All Categories'] + sorted(temp_category_list)
        else:
            st.session_state['category_list'] = category
    else:
        # start select subset data
        pass

def show_recommended_items(recommended_df):
    recommended_df = recommended_df.reset_index(drop=True)
    # st.dataframe(recommended_df)
    if len(recommended_df) > 0:
        row_last_5txn = st.columns(len(recommended_df))

        for num, col in enumerate(row_last_5txn):
            tile = col.container(border=True,height=300)

            tile.image(
                recommended_df.iloc[num]['images.large'][0], width=100
            )

            if 'rating' in recommended_df.columns:
                tile.write(f"Rating: {'{0:.2f}'.format(recommended_df.iloc[num]['rating'])}")
            elif 'score' in recommended_df.columns:
                tile.write(f"Predicted Rating: {'{0:.2f}'.format(recommended_df.iloc[num]['score'])}")

            tile.write(f"**{recommended_df.iloc[num]['title']}**")
    else:
         st.error('No Product Found', icon="🚨")
        # st.error('No Product Found')

def clear_cache():
    keys = list(st.session_state.keys())
    for key in keys:
        st.session_state.pop(key)

    st.session_state['selected_user'] = '0001'
    st.session_state['selected_category'] = 'All Categories'
    st.session_state['selected_store'] = 'All Stores'
    st.session_state['clicked_category'] = False
    st.session_state['clicked_store'] = False
    st.session_state['user_list'] = user
    st.session_state['category_list'] = category
    st.session_state['store_list'] = store

with st.sidebar:
    selected_user = st.selectbox("User:", options = user, key="selected_user")
    selected_category = st.selectbox("Category:", options = st.session_state['category_list'], key="selected_category", on_change=update_store)
    selected_store = st.selectbox("Store Name:", options = st.session_state['store_list'], key="selected_store", on_change=update_category)

    st.button('Reset', on_click=clear_cache)

# Title
st.title("Retail Recommendation System")

st.subheader('Model Evaluation')

model_results = user_evaluation_list[user_evaluation_list['user_id'] == 'model'].reset_index(drop=True)

precision_recall_data = [['Wide & Deep Learning', 30, model_results.iloc[0]['wide_deep_precision'], model_results.iloc[0]['wide_deep_recall']], 
                        ['K-Nearest Neighbors (KNN)', 30, model_results.iloc[0]['knn_precision'], model_results.iloc[0]['knn_recall']],
                        ['Matrix Factorization', 30, model_results.iloc[0]['matrix_fact_precision'], model_results.iloc[0]['matrix_fact_recall']]]

precision_recall_df = pd.DataFrame(precision_recall_data, columns=['Model', 'Top K', 'Precision@K', 'Recall@K'])

metric_data = [['Wide & Deep Learning', (round(model_results.iloc[0]['wide_deep_mae'],4)).astype(str), (round(model_results.iloc[0]['wide_deep_mse'],4)).astype(str), (round(model_results.iloc[0]['wide_deep_rmse'],4)).astype(str)], 
                ['Matrix Factorization', (round(model_results.iloc[0]['matrix_fact_mae'],4)).astype(str), (round(model_results.iloc[0]['matrix_fact_mse'],4)).astype(str), (round(model_results.iloc[0]['matrix_fact_rmse'],4)).astype(str)],
                ['K-Nearest Neighbors (KNN)', '-', '-', '-']]

metric_df = pd.DataFrame(metric_data, columns=['Model', 'MAE', 'MSE', 'RMSE'])

df_rows = st.columns(2)
df_rows[0].dataframe(precision_recall_df)
df_rows[1].dataframe(metric_df)

# st.dataframe(precision_recall_df)
# st.dataframe(metric_df)

# Last 5 transaction of users
select_user_id = user_evaluation_list[user_evaluation_list.user_idx == st.session_state['selected_user']].reset_index(drop=True).iloc[0]['user_id']
selected_user_last5_txn = user_last5_txn[user_last5_txn['user_id'] == select_user_id]

st.subheader(f"Last 5 Transactions of User {st.session_state['selected_user']}")

show_recommended_items(selected_user_last5_txn)

if st.session_state['selected_category'] != 'All Categories':
    cat_last5_txn = user_transactions[user_transactions['main_category'] == st.session_state['selected_category']].sort_values(['user_id', 'timestamp'],ascending=True).groupby(['user_id']).tail(5)
    selected_user_cat_last5_txn = cat_last5_txn[cat_last5_txn['user_id'] == select_user_id].reset_index(drop=True)

    st.subheader(f"Last {len(selected_user_cat_last5_txn)} Transactions in {st.session_state['selected_category']} of User {st.session_state['selected_user']}")

    show_recommended_items(selected_user_cat_last5_txn)

selected_user_knn_top5 = knn_recommend_items[knn_recommend_items['user_id'] == select_user_id]
selected_user_matrix_top5 = matrix_recommend_items[matrix_recommend_items['user_id'] == select_user_id]
selected_user_wide_deep_top5 = wide_deep_recommend_items[wide_deep_recommend_items['user_id'] == select_user_id]

if (st.session_state['selected_category'] == 'All Categories' and st.session_state['selected_store'] == 'All Stores'):
    selected_user_knn_top5 = selected_user_knn_top5.sort_values(['user_id', 'score'],ascending=True).groupby(['user_id']).head(5).reset_index(drop=True)

    selected_user_matrix_top5 = selected_user_matrix_top5.sort_values(['user_id', 'score'],ascending=False).groupby(['user_id']).head(5).reset_index(drop=True)

    selected_user_wide_deep_top5 = selected_user_wide_deep_top5.sort_values(['user_id', 'score'],ascending=False).groupby(['user_id']).head(5).reset_index(drop=True)

elif st.session_state['selected_category'] == 'All Categories':
    selected_user_knn_top5 = selected_user_knn_top5[(knn_recommend_items['store'] == st.session_state['selected_store'])]
    selected_user_knn_top5 = selected_user_knn_top5.sort_values(['user_id', 'score'],ascending=True).groupby(['user_id']).head(5).reset_index(drop=True)

    selected_user_matrix_top5 = selected_user_matrix_top5[(selected_user_matrix_top5['store'] == st.session_state['selected_store'])]
    selected_user_matrix_top5 = selected_user_matrix_top5.sort_values(['user_id', 'score'],ascending=False).groupby(['user_id']).head(5).reset_index(drop=True)

    selected_user_wide_deep_top5 = selected_user_wide_deep_top5[(wide_deep_recommend_items['store'] == st.session_state['selected_store'])]
    selected_user_wide_deep_top5 = selected_user_wide_deep_top5.sort_values(['user_id', 'score'],ascending=False).groupby(['user_id']).head(5).reset_index(drop=True)

elif st.session_state['selected_store'] == 'All Stores':
    selected_user_knn_top5 = selected_user_knn_top5[(selected_user_knn_top5['main_category'] == st.session_state['selected_category'])]
    selected_user_knn_top5 = selected_user_knn_top5.sort_values(['user_id', 'score'],ascending=True).groupby(['user_id']).head(5).reset_index(drop=True)

    selected_user_matrix_top5 = selected_user_matrix_top5[(selected_user_matrix_top5['main_category'] == st.session_state['selected_category'])]
    selected_user_matrix_top5 = selected_user_matrix_top5.sort_values(['user_id', 'score'],ascending=False).groupby(['user_id']).head(5).reset_index(drop=True)

    selected_user_wide_deep_top5 = selected_user_wide_deep_top5[(selected_user_wide_deep_top5['main_category'] == st.session_state['selected_category'])]
    selected_user_wide_deep_top5 = selected_user_wide_deep_top5.sort_values(['user_id', 'score'],ascending=False).groupby(['user_id']).head(5).reset_index(drop=True)

else:
    selected_user_knn_top5 = selected_user_knn_top5[(knn_recommend_items['main_category'] == st.session_state['selected_category'])]
    selected_user_knn_top5 = selected_user_knn_top5[(selected_user_knn_top5['store'] == st.session_state['selected_store'])]
    selected_user_knn_top5 = selected_user_knn_top5.sort_values(['user_id', 'score'],ascending=True).groupby(['user_id']).head(5).reset_index(drop=True)

    selected_user_matrix_top5 = selected_user_matrix_top5[(selected_user_matrix_top5['main_category'] == st.session_state['selected_category'])]
    selected_user_matrix_top5 = selected_user_matrix_top5[(selected_user_matrix_top5['store'] == st.session_state['selected_store'])]
    selected_user_matrix_top5 = selected_user_matrix_top5.sort_values(['user_id', 'score'],ascending=False).groupby(['user_id']).head(5).reset_index(drop=True)

    selected_user_wide_deep_top5 = selected_user_wide_deep_top5[(selected_user_wide_deep_top5['main_category'] == st.session_state['selected_category'])]
    selected_user_wide_deep_top5 = selected_user_wide_deep_top5[(selected_user_wide_deep_top5['store'] == st.session_state['selected_store'])]
    selected_user_wide_deep_top5 = selected_user_wide_deep_top5.sort_values(['user_id', 'score'],ascending=False).groupby(['user_id']).head(5).reset_index(drop=True)

st.subheader(f"Recommended Products in {st.session_state['selected_category']} (Wide & Deep Learning)")

wide_deep_user_score = user_evaluation_list[user_evaluation_list['user_id'] == select_user_id].reset_index(drop=True)
wide_deep_mae = wide_deep_user_score.iloc[0]['wide_deep_mae']
wide_deep_mse = wide_deep_user_score.iloc[0]['wide_deep_mse']
wide_deep_rmse = wide_deep_user_score.iloc[0]['wide_deep_rmse']

st.write("**Mean Absolute Error (MAE)**: {mae:.4f}, **Mean Squared Error (MSE)**: {mse:.4f}, **Root Mean Squared Error (RMSE)**: {rmse:.4f}"
            .format(mae=wide_deep_mae, mse=wide_deep_mse, rmse=wide_deep_rmse))

show_recommended_items(selected_user_wide_deep_top5)

st.subheader(f"Recommended Products in {st.session_state['selected_category']} (KNN)")
selected_user_knn_top5_ = selected_user_knn_top5.drop(columns = 'score')
show_recommended_items(selected_user_knn_top5_)

st.subheader(f"Recommended Products in {st.session_state['selected_category']} (Matrix Factorization)")

matrix_fact_user_score = user_evaluation_list[user_evaluation_list['user_id'] == select_user_id].reset_index(drop=True)
matrix_fact_mae = matrix_fact_user_score.iloc[0]['matrix_fact_mae']
matrix_fact_mse = matrix_fact_user_score.iloc[0]['matrix_fact_mse']
matrix_fact_rmse = matrix_fact_user_score.iloc[0]['matrix_fact_rmse']

st.write("**Mean Absolute Error (MAE)**: {mae:.4f}, **Mean Squared Error (MSE)**: {mse:.4f}, **Root Mean Squared Error (RMSE)**: {rmse:.4f}"
            .format(mae=matrix_fact_mae, mse=matrix_fact_mse, rmse=matrix_fact_rmse))

show_recommended_items(selected_user_matrix_top5)


# model_result = evaluation_df[evaluation_df.user_id == user_id].reset_index(drop=True)

# mae = model_result.iloc[0]['mae']
# mse = model_result.iloc[0]['mse']
# rmse = model_result.iloc[0]['rmse']

# st.write("**Mean Absolute Error (MAE)**: {mae:.4f}, **Mean Squared Error (MSE)**: {mse:.4f}, **Root Mean Squared Error (RMSE)**: {rmse:.4f}"
#             .format(mae=mae, mse=mse, rmse=rmse))
