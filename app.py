import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder

# ---------------------------
# Load dataset
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("merged_df.csv")
    df['content'] = df['content'].fillna('')
    df['user_id'] = df['user_id'].fillna('unknown')
    df['news_id'] = df['news_id'].fillna('unknown')
    return df

merged_df = load_data()

# ---------------------------
# Content-Based Recommendation
# ---------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=5, ngram_range=(1,2))
tfidf_matrix = vectorizer.fit_transform(merged_df['content'])

def content_base_rec(title, top_n=5):
    title_clean = re.sub(r'\W+', ' ', title).lower().strip()
    title_vec = vectorizer.transform([title_clean])
    sim_scores = cosine_similarity(title_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[::-1][:top_n]
    return merged_df.loc[top_indices, ['news_id','title','category','subcategory','url','abstract']]

# ---------------------------
# Collaborative-Based Recommendation
# ---------------------------
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()
merged_df['user_enc'] = user_encoder.fit_transform(merged_df['user_id'])
merged_df['news_enc'] = item_encoder.fit_transform(merged_df['news_id'])
user_item_matrix = merged_df.pivot_table(index='user_enc', columns='news_enc', values='clicked', fill_value=0)
user_similarity = cosine_similarity(user_item_matrix)

def collaborative_base_rec(input_user, df, top_k=5):
    user_idx = user_encoder.transform([input_user])[0]
    sim_scores = user_similarity[user_idx]
    user_clicks = user_item_matrix.iloc[user_idx]
    weighted_scores = sim_scores @ user_item_matrix.values
    weighted_scores[user_clicks==1] = 0
    top_news_indices = weighted_scores.argsort()[::-1][:top_k]
    recommended_news_ids = item_encoder.inverse_transform(top_news_indices)
    recommended_news = df[df['news_id'].isin(recommended_news_ids)][
        ['news_id','title','category','subcategory','url','abstract']
    ].drop_duplicates(subset='news_id')
    return recommended_news.reset_index(drop=True)

# ---------------------------
# Hybrid Recommendation
# ---------------------------
def hybrid_recommendations(user_id, title, df, top_n=5):
    content_rec = content_base_rec(title)
    collab_rec = collaborative_base_rec(user_id, df)
    combine_rec = pd.concat([content_rec, collab_rec]).drop_duplicates().reset_index(drop=True)
    combine_rec = combine_rec.head(top_n)
    return combine_rec

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI News Recommender", page_icon="📰", layout="wide")
st.title("AI News Recommendation System")
st.write("Content-Based, Collaborative & Hybrid Recommendation Example")

user_id = st.text_input("Enter User ID", "U91836")
title = st.text_input("Enter News Title / Query", "Top investment strategies 2025")
top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

if st.button("Get Hybrid Recommendations"):
    results = hybrid_recommendations(user_id, title, merged_df, top_n=top_n)
    st.write(f"Showing top {top_n} recommended news:")
    st.dataframe(results)
