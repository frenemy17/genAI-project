import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import json
# Apply the same engineering feature functions expected by the pipeline
def clean_and_engineer(data):
    df_processed = data.copy()
    import re
    
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

st.set_page_config(page_title="Intelligent Exam Question Analysis", layout="wide")

st.title("INTELLIGENT EXAM QUESTION ANALYSIS")

# Load models
@st.cache_resource
def load_models():
    bloom_path = "models/bloom_pipeline.pkl"
    diff_path = "models/difficulty_pipeline.pkl"
    
    if os.path.exists(bloom_path) and os.path.exists(diff_path):
        bloom_pipe = joblib.load(bloom_path)
        diff_pipe = joblib.load(diff_path)
        return bloom_pipe, diff_pipe
    return None, None

bloom_model, difficulty_model = load_models()

if not bloom_model:
    st.warning("Models not found. Please run `python train_and_save.py` first.")
    st.stop()

page = st.sidebar.radio("Navigation", ["📝 Predict Cognitive Load & Difficulty", "📊 Model Analytics & Accuracy"])

if page == "📝 Predict Cognitive Load & Difficulty":
    st.subheader("Predict Cognitive Load & Empirical Difficulty")
    st.sidebar.header("Input Metrics")
    
    # Academic Text
    question_text = st.text_area("Question Text", placeholder="E.g., Evaluate the ethical implications of genetic engineering...")
    
    col1, col2 = st.columns(2)
    with col1:
        subject = st.selectbox("Subject", ["Computer Science", "Biology", "Physics", "Mathematics", "Economics"])
    with col2:
        topic = st.text_input("Topic", "Ethics")
    
    st.markdown("---")
    st.markdown("### Empirical Student Data")
    st.markdown("These statistics provide the model with the necessary vectors to anchor its psychometric predictions.")
    
    col3, col4, col5 = st.columns(3)
    with col3:
        avg_score = st.slider("Average Score (%)", 0.0, 100.0, 75.0)
        correct_pct = st.slider("Correct Percentage", 0.0, 1.0, 0.75)
    with col4:
        attempted = st.number_input("Students Attempted", value=150, min_value=1)
        correct = st.number_input("Students Correct", value=112, min_value=0)
    with col5:
        time_taken = st.number_input("Total Time Taken (minutes)", value=300.0)
        
    if st.button("Analyze Question", type="primary"):
        if not question_text.strip():
            st.error("Please provide the question text.")
        else:
            # Construct DataFrame record
            input_dict = {
                "question_text": question_text,
                "subject": subject,
                "topic": topic,
                "avg_score": float(avg_score),
                "correct_percentage": float(correct_pct),
                "num_students_attempted": float(attempted),
                "num_students_correct": float(correct),
                "time_taken_minutes": float(time_taken)
            }
            
            input_df = pd.DataFrame([input_dict])
            engineered_df = clean_and_engineer(input_df)
            
            # Inference
            try:
                bloom_pred = bloom_model.predict(engineered_df)[0]
                diff_pred = difficulty_model.predict(engineered_df)[0]
                
                st.markdown("---")
                st.markdown("### Prediction Results")
                
                metric1, metric2 = st.columns(2)
                metric1.metric(label="Predicted Bloom's Taxonomy Level", value=bloom_pred)
                metric2.metric(label="Estimated Academic Difficulty", value=diff_pred)
                
                st.info("💡 **Model Confidence Note**: These classifications utilize purely Classical ML Ensembling mapped over TF-IDF n-grams (1-3) and raw psychometric dimensions.")
            except Exception as e:
                st.error(f"Inference Engine Failed: {e}. Please ensure the models were built using the same version of scikit-learn.")
                
elif page == "📊 Model Analytics & Accuracy":
        st.markdown("### Model Performance & Detailed Insights")
        
        metrics_path = "models/metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                pipeline_metrics = json.load(f)
                
            m1, m2 = st.columns(2)
            m1.metric(label="Bloom's Level Accuracy", value=f"{pipeline_metrics.get('bloom_accuracy', 'N/A')}%")
            m2.metric(label="Difficulty Level Accuracy", value=f"{pipeline_metrics.get('difficulty_accuracy', 'N/A')}%")
            
            st.divider()
            st.markdown("### 🔍 Dataset Analytical Insights")
            
            try:
                df_viz = pd.read_csv("cognitive_dataset.csv")
                
                # Insight 1: Time Taken vs Score Correlation
                st.markdown("**1. Correlation: Time Taken vs. Success Rate**")
                st.markdown("Does spending more time on a question correlate with higher student success?")
                # We'll use a native Streamlit scatter chart
                df_scatter = df_viz[['time_taken_minutes', 'correct_percentage']].rename(
                    columns={'time_taken_minutes': 'Time Spent (min)', 'correct_percentage': 'Success Rate'}
                )
                st.scatter_chart(df_scatter, x='Time Spent (min)', y='Success Rate')
                
                st.markdown("---")
                
                # Insight 2: Difficulty Impact on Attempts
                st.markdown("**2. Student Engagement by Empirical Difficulty**")
                st.markdown("Average number of student attempts broken down by the question's difficulty rating.")
                df_bar = df_viz.groupby('difficulty')['num_students_attempted'].mean().reset_index()
                df_bar = df_bar.set_index('difficulty')
                st.bar_chart(df_bar)
                
                st.markdown("---")
                
                # Insight 3: Bloom's Level Distribution
                st.markdown("**3. Question Frequency Across Bloom's Taxonomy**")
                st.markdown("The distribution of cognitive load requirements across the entire tested dataset.")
                df_bloom = df_viz['bloom_level'].value_counts().reset_index()
                df_bloom.columns = ['Bloom Taxonomy Level', 'Question Count']
                df_bloom = df_bloom.set_index('Bloom Taxonomy Level')
                st.bar_chart(df_bloom)
                
            except Exception as e:
                st.warning("Could not load internal dataset for deep visualizations.")
        else:
            st.warning("No metrics found. Please re-run the `train_and_save.py` script to generate evaluation scores.")
