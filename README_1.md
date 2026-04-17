SENTIMENT-BASED PRODUCT RECOMMENDATION SYSTEM
An end-to-end hybrid recommendation system that combines collaborative filtering with sentiment-aware ranking to deliver more relevant and user-aligned product suggestions.

PROBLEM STATEMENT-
Traditional recommendation systems rely heavily on user-item interactions (ratings), often ignoring qualitative feedback like reviews. This project enhances recommendations by incorporating sentiment analysis from user reviews to refine results.

APPROACH :- 
1 - COLLABORATIVE FILTERING
 Implemented Item-Item Collaborative Filtering
 Used cosine similarity on mean-centered rating matrix
 Generated Top-N recommendations (Top-20)
2 - SENTIMENT ANALYSIS MODEL
  Preprocessed text (cleaning, tokenization, stopword removal)
 Extracted features using TF-IDF vectorization
 Handled class imbalance using class_weight
 Trained and compared:
      Logistic Regression
      Random Forest
      XGBoost
 Selected Random Forest via GridSearchCV tuning
3 - HYBRID RECOMENDATION LAYER
  Filtered Top-20 CF recommendations using:
     Sentiment positivity percentage
  Final output:
     Top-5 sentiment-refined product recommendations

TECH STACK:-
  Python
  Scikit-learn
  Pandas, NumPy
  TF-IDF (NLP)
  Flask (Deployment)

MODEL EVALUATION:-
  Collaborative Filtering: Evaluated using RMSE
  Classification Models: Compared using accuracy, precision, recall
  Selected model based on performance + generalization

DEPLOYMENT:-
  Built a Flask web app
  Users input a username
  System returns personalized product recommendations in real-time

KEY HIGHLIGHTS:-
  Hybrid system combining behavioral + textual intelligence
  Improved recommendation quality using sentiment filtering
  Demonstrates end-to-end ML pipeline:
     EDA → Feature Engineering → Modeling → Evaluation → Deployment

FUTURE IMPROVEMENTS:-
  Use deep learning models (BERT) for sentiment analysis
  Add user-based collaborative filtering
  Deploy using Docker + cloud (AWS/GCP)
  Build interactive frontend (React)
