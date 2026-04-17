# ==============================
# model.py
# ==============================

import pandas as pd
import numpy as np
import joblib
import re
import string
from sklearn.metrics.pairwise import cosine_similarity

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("DATA SET.csv")

# ------------------------------
# TEXT CLEANING FUNCTION
# ------------------------------

def clean_text_function(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

# Create clean_text column (since CSV doesn't contain it)
df['clean_text'] = df['reviews_text'].apply(clean_text_function)

# Drop missing critical values
df = df.dropna(subset=[
    'reviews_username',
    'id',
    'reviews_rating',
    'clean_text',
    'name'
])

# ==============================
# LOAD SAVED SENTIMENT MODEL
# ==============================

sentiment_model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# ==============================
# CREATE PRODUCT ID → NAME MAP
# ==============================

product_id_name_map = (
    df[['id', 'name']]
    .drop_duplicates()
    .set_index('id')['name']
    .to_dict()
)

# ==============================
# CREATE USER-ITEM RATING MATRIX
# ==============================

rating_df = df[['reviews_username', 'id', 'reviews_rating']].drop_duplicates()

rating_matrix = rating_df.pivot_table(
    index='reviews_username',
    columns='id',
    values='reviews_rating'
)

# ==============================
# NORMALIZE RATINGS (User Mean Centering)
# ==============================

user_mean = rating_matrix.mean(axis=1)

rating_matrix_centered = rating_matrix.sub(user_mean, axis=0)
rating_matrix_centered = rating_matrix_centered.fillna(0)

# ==============================
# ITEM-ITEM COSINE SIMILARITY
# ==============================

item_similarity = cosine_similarity(rating_matrix_centered.T)

item_similarity_df = pd.DataFrame(
    item_similarity,
    index=rating_matrix.columns,
    columns=rating_matrix.columns
)
# ==============================
# FINAL HYBRID RECOMMEND FUNCTION
# ==============================

def recommend_products(username):

    # --------------------------
    # Check if user exists
    # --------------------------
    if username not in rating_matrix.index:
        return ["User not found"]

    # --------------------------
    # Step 1: Collaborative Filtering
    # --------------------------

    user_ratings = rating_matrix.loc[username]
    user_ratings_centered = user_ratings - user_mean[username]
    user_ratings_centered = user_ratings_centered.fillna(0)

    predicted_scores = item_similarity_df.dot(user_ratings_centered)

    # Remove already rated products
    already_rated = user_ratings[user_ratings.notna()].index
    predicted_scores = predicted_scores.drop(already_rated)

    # Get Top 20 recommended product IDs
    top_20_ids = predicted_scores.sort_values(ascending=False).head(20).index

    # --------------------------
    # Step 2: Sentiment Filtering
    # --------------------------

    filtered_df = df[df['id'].isin(top_20_ids)].copy()

    if filtered_df.empty:
        return ["No recommendations available"]

    # Predict sentiment using trained RF model
    X_text = vectorizer.transform(filtered_df['clean_text'])
    filtered_df['predicted_sentiment'] = sentiment_model.predict(X_text)

    # Calculate positive percentage per product
    sentiment_scores = (
        filtered_df.groupby('id')['predicted_sentiment']
        .mean()
        .sort_values(ascending=False)
    )

    # Take Top 5 based on positivity
    top_5_ids = sentiment_scores.head(5).index.tolist()

    # --------------------------
    # Step 3: Map IDs to Names
    # --------------------------

    recommended_products = [
        product_id_name_map.get(pid, "Unknown Product")
        for pid in top_5_ids
    ]

    return recommended_products