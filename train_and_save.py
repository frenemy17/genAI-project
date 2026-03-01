import pandas as pd
import numpy as np
import re
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# 1. Feature Engineering function
def clean_and_engineer(data):
    df_processed = data.copy()
    
    # Handling unseen logic gracefully in production
    attempts = np.maximum(df_processed['num_students_attempted'], 1)
    df_processed['time_per_attempt'] = df_processed['time_taken_minutes'] / attempts
    df_processed['log_attempts'] = np.log1p(df_processed['num_students_attempted'])
    df_processed['question_length'] = df_processed['question_text'].astype(str).apply(lambda x: len(x.split()))
    
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^a-z0-9\\s]', '', text)
        return text
    
    df_processed['cleaned_text'] = df_processed['question_text'].apply(clean_text)
    return df_processed

def train_and_save_pipeline(csv_path="cognitive_dataset.csv", model_dir="models"):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    df_engineered = clean_and_engineer(df)
    
    X = df_engineered.drop(columns=['bloom_level', 'difficulty', 'question_text'])
    y_bloom = df_engineered['bloom_level']
    y_difficulty = df_engineered['difficulty']
    
    # 2. Pipeline Components
    text_col = 'cleaned_text'
    categorical_cols = ['subject', 'topic']
    numerical_cols = [
        'avg_score', 'correct_percentage', 'num_students_attempted', 
        'num_students_correct', 'time_taken_minutes', 'time_per_attempt',
        'log_attempts', 'question_length'
    ]

    text_transformer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=3, max_df=0.85, max_features=5000)
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('text', text_transformer, text_col),
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ]
    )
    
    # 3. Final Pipelines
    print("Training Bloom Level Model...")
    bloom_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
    ])
    bloom_pipeline.fit(X, y_bloom)
    
    print("Training Difficulty Model...")
    difficulty_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), # Preprocessor fits again, but on identical X so it's fine
        ('classifier', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
    ])
    difficulty_pipeline.fit(X, y_difficulty)
    
    # 4. Save Models
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    joblib.dump(bloom_pipeline, os.path.join(model_dir, 'bloom_pipeline.pkl'))
    joblib.dump(difficulty_pipeline, os.path.join(model_dir, 'difficulty_pipeline.pkl'))
    print(f"Models successfully saved to {model_dir}/")

if __name__ == "__main__":
    train_and_save_pipeline()
